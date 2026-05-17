//! `stryke-spark-helper` — bridge binary for the stryke `spark` package.
//!
//! Submits an embedded PySpark driver script via `spark-submit` for each
//! command. The driver runs the user's SQL against a SparkSession and
//! writes NDJSON rows to stdout, which we forward verbatim. Universal
//! across Spark 3.x — works wherever `spark-submit` is on PATH or
//! `$SPARK_HOME/bin/`.
//!
//! Output:
//!   query     → NDJSON rows on stdout (each row from `df.toJSON()`)
//!              `--columnar` emits one `{columns,num_rows,rows}` object
//!              `--with-meta` prepends a `{"meta":{columns:[…]}}` line
//!   execute   → `{"ok":true}` (or `{ok:false,error:…}` on driver failure)
//!   tables    → NDJSON `{"name":"...","database":"..."}`
//!   databases → NDJSON `{"name":"..."}`
//!   schema    → `{table, columns:[{name,type,nullable,comment,...}],
//!                 partitions:[...], properties:{...}}`
//!   ping      → `ok` on stdout, exit 0 on success
//!   submit    → forwards a user script through spark-submit verbatim

use std::ffi::OsStr;
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};

use anyhow::{anyhow, bail, Context, Result};
use clap::{Parser, Subcommand};

/* ------------------------------------------------------------------------- */
/* embedded PySpark driver                                                   */
/* ------------------------------------------------------------------------- */

const DRIVER_PY: &str = include_str!("driver.py");

/* ------------------------------------------------------------------------- */
/* CLI                                                                       */
/* ------------------------------------------------------------------------- */

#[derive(Parser)]
#[command(
    name = "stryke-spark-helper",
    version,
    about = "Apache Spark bridge for the stryke `spark` package"
)]
struct Cli {
    /// `spark://host:port`, `local[*]`, `yarn`, `k8s://…`. Defaults to
    /// `$SPARK_MASTER` or `local[*]`.
    #[arg(long, env = "SPARK_MASTER", global = true)]
    master: Option<String>,

    /// `$SPARK_HOME` (used to locate `spark-submit` when not on PATH).
    #[arg(long, env = "SPARK_HOME", global = true)]
    spark_home: Option<PathBuf>,

    /// Explicit path to `spark-submit` (overrides PATH and --spark-home).
    #[arg(long, global = true)]
    spark_submit: Option<PathBuf>,

    /// `spark.executor.memory=4g` style. Repeatable.
    #[arg(long = "conf", short = 'c', global = true, value_name = "K=V")]
    conf: Vec<String>,

    /// `--packages` Maven coordinates, comma-separated.
    #[arg(long, global = true)]
    packages: Option<String>,

    /// `--jars` paths, comma-separated.
    #[arg(long, global = true)]
    jars: Option<String>,

    /// Application name shown in Spark UI.
    #[arg(long, global = true, default_value = "stryke-spark")]
    app_name: String,

    /// Deploy mode: `client` (default) or `cluster`.
    #[arg(long, global = true)]
    deploy_mode: Option<String>,

    /// Default database to USE before running the command.
    #[arg(long, short = 'D', global = true)]
    database: Option<String>,

    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Run a SparkSQL SELECT and stream rows as NDJSON.
    Query {
        sql: String,
        #[arg(long)]
        columnar: bool,
        #[arg(long)]
        with_meta: bool,
        #[arg(long)]
        limit: Option<usize>,
    },
    /// Run a DDL / DML statement (CREATE TABLE, INSERT, etc.).
    Execute {
        sql: String,
    },
    /// `SELECT * FROM TABLE [WHERE w] [ORDER BY o] [LIMIT n]` shorthand.
    Dump {
        #[arg(long, short = 't')]
        table: String,
        #[arg(long)]
        columns: Option<String>,
        #[arg(long = "where", short = 'w')]
        where_clause: Option<String>,
        #[arg(long)]
        order_by: Option<String>,
        #[arg(long)]
        limit: Option<usize>,
    },
    /// List tables in the current database.
    Tables,
    /// List databases / catalogs.
    Databases,
    /// `DESCRIBE TABLE EXTENDED` for one table.
    Schema {
        #[arg(long, short = 't')]
        table: String,
    },
    /// Run `SELECT 1`. Exit 0 on success.
    Ping,
    /// Forward an arbitrary script through spark-submit. Use for jobs
    /// outside the SQL surface (`.py` / `.jar` / `.scala` w/ --class).
    Submit {
        script: PathBuf,
        /// Pass-through arguments after the script.
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<String>,
    },
}

/* ------------------------------------------------------------------------- */
/* main                                                                      */
/* ------------------------------------------------------------------------- */

fn main() {
    let cli = Cli::parse();
    if let Err(e) = run(cli) {
        eprintln!("stryke-spark-helper: {e:#}");
        std::process::exit(1);
    }
}

fn run(cli: Cli) -> Result<()> {
    match &cli.cmd {
        Cmd::Submit { script, args } => cmd_submit(&cli, script, args),
        _ => cmd_via_driver(&cli),
    }
}

/* ------------------------------------------------------------------------- */
/* `submit` — pass-through                                                   */
/* ------------------------------------------------------------------------- */

fn cmd_submit(cli: &Cli, script: &PathBuf, args: &[String]) -> Result<()> {
    let submit = locate_spark_submit(cli)?;
    let mut cmd = Command::new(&submit);
    apply_common_args(&mut cmd, cli);
    cmd.arg(script);
    cmd.args(args);
    cmd.stdin(Stdio::inherit())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit());
    let status = cmd
        .status()
        .with_context(|| format!("spawning {}", submit.display()))?;
    std::process::exit(status.code().unwrap_or(1));
}

/* ------------------------------------------------------------------------- */
/* `query/execute/...` — embedded driver                                     */
/* ------------------------------------------------------------------------- */

fn cmd_via_driver(cli: &Cli) -> Result<()> {
    // Write the embedded driver to a temp file so spark-submit can find it.
    let mut driver_file = tempfile::Builder::new()
        .prefix("stryke-spark-driver-")
        .suffix(".py")
        .tempfile()
        .context("creating temp driver file")?;
    driver_file
        .write_all(DRIVER_PY.as_bytes())
        .context("writing driver to temp file")?;
    driver_file.flush()?;
    let driver_path = driver_file.path().to_path_buf();

    let request = build_request_json(cli)?;

    let submit = locate_spark_submit(cli)?;
    let mut cmd = Command::new(&submit);
    apply_common_args(&mut cmd, cli);
    cmd.arg(&driver_path);
    // Pass the request as a single JSON arg to the driver.
    cmd.arg(&request);

    // Quiet spark-submit's banner so it doesn't pollute NDJSON stdout. The
    // driver writes its own status to stderr; spark-submit writes JVM warmup
    // chatter to stderr by default which we let through.
    cmd.stdin(Stdio::null())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit());

    let status = cmd
        .status()
        .with_context(|| format!("spawning {}", submit.display()))?;
    // Drop the temp file explicitly (NamedTempFile drops on Drop, but we
    // keep `driver_file` alive until here).
    drop(driver_file);
    std::process::exit(status.code().unwrap_or(1));
}

fn build_request_json(cli: &Cli) -> Result<String> {
    use serde_json::json;
    let mut req = serde_json::Map::new();

    if let Some(db) = &cli.database {
        req.insert("database".into(), json!(db));
    }

    match &cli.cmd {
        Cmd::Query {
            sql,
            columnar,
            with_meta,
            limit,
        } => {
            req.insert("cmd".into(), json!("query"));
            req.insert("sql".into(), json!(sql));
            req.insert("columnar".into(), json!(columnar));
            req.insert("with_meta".into(), json!(with_meta));
            if let Some(l) = limit {
                req.insert("limit".into(), json!(l));
            }
        }
        Cmd::Execute { sql } => {
            req.insert("cmd".into(), json!("execute"));
            req.insert("sql".into(), json!(sql));
        }
        Cmd::Dump {
            table,
            columns,
            where_clause,
            order_by,
            limit,
        } => {
            req.insert("cmd".into(), json!("dump"));
            req.insert("table".into(), json!(table));
            if let Some(c) = columns {
                req.insert("columns".into(), json!(c));
            }
            if let Some(w) = where_clause {
                req.insert("where".into(), json!(w));
            }
            if let Some(o) = order_by {
                req.insert("order_by".into(), json!(o));
            }
            if let Some(l) = limit {
                req.insert("limit".into(), json!(l));
            }
        }
        Cmd::Tables => {
            req.insert("cmd".into(), json!("tables"));
        }
        Cmd::Databases => {
            req.insert("cmd".into(), json!("databases"));
        }
        Cmd::Schema { table } => {
            req.insert("cmd".into(), json!("schema"));
            req.insert("table".into(), json!(table));
        }
        Cmd::Ping => {
            req.insert("cmd".into(), json!("ping"));
        }
        Cmd::Submit { .. } => bail!("submit not handled here"),
    }

    serde_json::to_string(&req).context("encoding driver request")
}

/* ------------------------------------------------------------------------- */
/* spark-submit plumbing                                                     */
/* ------------------------------------------------------------------------- */

fn locate_spark_submit(cli: &Cli) -> Result<PathBuf> {
    if let Some(p) = &cli.spark_submit {
        if p.is_file() {
            return Ok(p.clone());
        }
        bail!("--spark-submit {} not found", p.display());
    }
    if let Some(home) = &cli.spark_home {
        let p = home.join("bin/spark-submit");
        if p.is_file() {
            return Ok(p);
        }
    }
    // PATH lookup.
    if let Ok(path_var) = std::env::var("PATH") {
        for dir in path_var.split(':') {
            let p = PathBuf::from(dir).join("spark-submit");
            if p.is_file() {
                return Ok(p);
            }
        }
    }
    Err(anyhow!(
        "stryke-spark: `spark-submit` not found on PATH, $SPARK_HOME/bin/, or --spark-submit"
    ))
}

fn apply_common_args(cmd: &mut Command, cli: &Cli) {
    let master = cli.master.as_deref().unwrap_or("local[*]");
    cmd.arg("--master").arg(master);
    cmd.arg("--name").arg(&cli.app_name);
    if let Some(dm) = &cli.deploy_mode {
        cmd.arg("--deploy-mode").arg(dm);
    }
    if let Some(p) = &cli.packages {
        cmd.arg("--packages").arg(p);
    }
    if let Some(j) = &cli.jars {
        cmd.arg("--jars").arg(j);
    }
    for c in &cli.conf {
        cmd.arg("--conf").arg(c);
    }
    // Silence spark-submit's INFO logging so the user only sees driver
    // output. They can override with --conf spark.log.level=INFO.
    if !cli.conf.iter().any(|c| c.starts_with("spark.log.level=")) {
        cmd.arg("--conf").arg("spark.log.level=WARN");
    }
    // Default to the in-memory catalog so `SHOW DATABASES` / `SHOW TABLES`
    // don't try to talk to Hive (which blows up on JDK 17+ with
    // `getSubject is not supported`). Users wiring up a real Hive
    // metastore can override with `--conf spark.sql.catalogImplementation=hive`.
    if !cli
        .conf
        .iter()
        .any(|c| c.starts_with("spark.sql.catalogImplementation="))
    {
        cmd.arg("--conf")
            .arg("spark.sql.catalogImplementation=in-memory");
    }
}

/// Quiet a few warnings the `OsStr` import would generate if unused.
#[allow(dead_code)]
fn _force_osstr(_: &OsStr) {}
