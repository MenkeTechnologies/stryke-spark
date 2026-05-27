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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    fn base_cli(cmd: Cmd) -> Cli {
        Cli {
            master: None,
            spark_home: None,
            spark_submit: None,
            conf: vec![],
            packages: None,
            jars: None,
            app_name: "stryke-spark".into(),
            deploy_mode: None,
            database: None,
            cmd,
        }
    }

    // ─── build_request_json ──────────────────────────────────────────

    fn parse(s: &str) -> Value {
        serde_json::from_str(s).unwrap()
    }

    #[test]
    fn build_request_json_query_basic() {
        let cli = base_cli(Cmd::Query {
            sql: "SELECT 1".into(),
            columnar: false,
            with_meta: false,
            limit: None,
        });
        let v = parse(&build_request_json(&cli).unwrap());
        assert_eq!(v["cmd"], "query");
        assert_eq!(v["sql"], "SELECT 1");
        assert_eq!(v["columnar"], false);
        assert_eq!(v["with_meta"], false);
        assert!(v.as_object().unwrap().get("limit").is_none());
    }

    #[test]
    fn build_request_json_query_with_limit_and_flags() {
        let cli = base_cli(Cmd::Query {
            sql: "SELECT * FROM t".into(),
            columnar: true,
            with_meta: true,
            limit: Some(100),
        });
        let v = parse(&build_request_json(&cli).unwrap());
        assert_eq!(v["columnar"], true);
        assert_eq!(v["with_meta"], true);
        assert_eq!(v["limit"], 100);
    }

    #[test]
    fn build_request_json_execute() {
        let cli = base_cli(Cmd::Execute {
            sql: "CREATE TABLE t (x INT)".into(),
        });
        let v = parse(&build_request_json(&cli).unwrap());
        assert_eq!(v["cmd"], "execute");
        assert_eq!(v["sql"], "CREATE TABLE t (x INT)");
    }

    #[test]
    fn build_request_json_dump_all_optional_fields() {
        let cli = base_cli(Cmd::Dump {
            table: "events".into(),
            columns: Some("a,b,c".into()),
            where_clause: Some("a > 0".into()),
            order_by: Some("a DESC".into()),
            limit: Some(50),
        });
        let v = parse(&build_request_json(&cli).unwrap());
        assert_eq!(v["cmd"], "dump");
        assert_eq!(v["table"], "events");
        assert_eq!(v["columns"], "a,b,c");
        // SQL keyword `where` is reserved; serialized field name is "where".
        assert_eq!(v["where"], "a > 0");
        assert_eq!(v["order_by"], "a DESC");
        assert_eq!(v["limit"], 50);
    }

    #[test]
    fn build_request_json_dump_omits_unset_optionals() {
        let cli = base_cli(Cmd::Dump {
            table: "t".into(),
            columns: None,
            where_clause: None,
            order_by: None,
            limit: None,
        });
        let v = parse(&build_request_json(&cli).unwrap());
        let obj = v.as_object().unwrap();
        assert!(!obj.contains_key("columns"));
        assert!(!obj.contains_key("where"));
        assert!(!obj.contains_key("order_by"));
        assert!(!obj.contains_key("limit"));
    }

    #[test]
    fn build_request_json_tables_minimal() {
        let v = parse(&build_request_json(&base_cli(Cmd::Tables)).unwrap());
        assert_eq!(v["cmd"], "tables");
    }

    #[test]
    fn build_request_json_databases_minimal() {
        let v = parse(&build_request_json(&base_cli(Cmd::Databases)).unwrap());
        assert_eq!(v["cmd"], "databases");
    }

    #[test]
    fn build_request_json_schema_carries_table() {
        let cli = base_cli(Cmd::Schema {
            table: "users".into(),
        });
        let v = parse(&build_request_json(&cli).unwrap());
        assert_eq!(v["cmd"], "schema");
        assert_eq!(v["table"], "users");
    }

    #[test]
    fn build_request_json_ping_minimal() {
        let v = parse(&build_request_json(&base_cli(Cmd::Ping)).unwrap());
        assert_eq!(v["cmd"], "ping");
    }

    #[test]
    fn build_request_json_database_propagates_to_envelope() {
        let mut cli = base_cli(Cmd::Ping);
        cli.database = Some("warehouse".into());
        let v = parse(&build_request_json(&cli).unwrap());
        assert_eq!(v["database"], "warehouse");
        assert_eq!(v["cmd"], "ping");
    }

    #[test]
    fn build_request_json_submit_errors() {
        let cli = base_cli(Cmd::Submit {
            script: PathBuf::from("/tmp/x.py"),
            args: vec![],
        });
        let err = build_request_json(&cli).unwrap_err();
        assert!(format!("{err}").contains("submit"));
    }

    // ─── locate_spark_submit (only the explicit-not-found and PATH-empty branches) ──

    #[test]
    fn locate_spark_submit_explicit_missing_errors() {
        let mut cli = base_cli(Cmd::Ping);
        cli.spark_submit = Some(PathBuf::from("/definitely/not/here/spark-submit"));
        let err = locate_spark_submit(&cli).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("--spark-submit"));
        assert!(msg.contains("not found"));
    }

    #[test]
    fn locate_spark_submit_no_hints_no_path_errors() {
        // Save / clear PATH so the function can't find spark-submit anywhere.
        // SAFETY: env::set_var/remove_var requires unsafe in Rust 2024 edition;
        // this crate is 2021, so still safe.
        let saved = std::env::var("PATH").ok();
        std::env::remove_var("PATH");
        let cli = base_cli(Cmd::Ping);
        let result = locate_spark_submit(&cli);
        if let Some(p) = saved {
            std::env::set_var("PATH", p);
        }
        let err = result.unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("spark-submit"));
        assert!(msg.contains("not found"));
    }

    // ─── apply_common_args ───────────────────────────────────────────

    fn collect_args(cli: &Cli) -> Vec<String> {
        let mut cmd = Command::new("dummy");
        apply_common_args(&mut cmd, cli);
        cmd.get_args()
            .map(|a| a.to_string_lossy().into_owned())
            .collect()
    }

    #[test]
    fn apply_common_args_defaults() {
        let cli = base_cli(Cmd::Ping);
        let args = collect_args(&cli);
        // Master defaults to local[*].
        let pos = args.iter().position(|a| a == "--master").unwrap();
        assert_eq!(args[pos + 1], "local[*]");
        // App name.
        let pos = args.iter().position(|a| a == "--name").unwrap();
        assert_eq!(args[pos + 1], "stryke-spark");
        // Default --conf spark.log.level=WARN.
        assert!(args.windows(2).any(|w| w[0] == "--conf" && w[1] == "spark.log.level=WARN"));
        // Default --conf spark.sql.catalogImplementation=in-memory.
        assert!(args
            .windows(2)
            .any(|w| w[0] == "--conf" && w[1] == "spark.sql.catalogImplementation=in-memory"));
    }

    #[test]
    fn apply_common_args_master_override() {
        let mut cli = base_cli(Cmd::Ping);
        cli.master = Some("yarn".into());
        let args = collect_args(&cli);
        let pos = args.iter().position(|a| a == "--master").unwrap();
        assert_eq!(args[pos + 1], "yarn");
    }

    #[test]
    fn apply_common_args_user_log_level_suppresses_default() {
        // User-supplied spark.log.level= must not be overridden by the
        // default WARN. Pin this so an accidental flip would surface.
        let mut cli = base_cli(Cmd::Ping);
        cli.conf.push("spark.log.level=DEBUG".into());
        let args = collect_args(&cli);
        let warn_count = args
            .windows(2)
            .filter(|w| w[0] == "--conf" && w[1].starts_with("spark.log.level="))
            .count();
        assert_eq!(warn_count, 1, "expected exactly 1 log.level conf, got {warn_count}: {args:?}");
        assert!(args
            .windows(2)
            .any(|w| w[0] == "--conf" && w[1] == "spark.log.level=DEBUG"));
    }

    #[test]
    fn apply_common_args_user_catalog_impl_suppresses_default() {
        let mut cli = base_cli(Cmd::Ping);
        cli.conf.push("spark.sql.catalogImplementation=hive".into());
        let args = collect_args(&cli);
        let count = args
            .windows(2)
            .filter(|w| w[0] == "--conf" && w[1].starts_with("spark.sql.catalogImplementation="))
            .count();
        assert_eq!(count, 1);
    }

    #[test]
    fn apply_common_args_deploy_mode_packages_jars_propagate() {
        let mut cli = base_cli(Cmd::Ping);
        cli.deploy_mode = Some("cluster".into());
        cli.packages = Some("org.x:y:1.0".into());
        cli.jars = Some("/tmp/a.jar,/tmp/b.jar".into());
        let args = collect_args(&cli);
        let pos = args.iter().position(|a| a == "--deploy-mode").unwrap();
        assert_eq!(args[pos + 1], "cluster");
        let pos = args.iter().position(|a| a == "--packages").unwrap();
        assert_eq!(args[pos + 1], "org.x:y:1.0");
        let pos = args.iter().position(|a| a == "--jars").unwrap();
        assert_eq!(args[pos + 1], "/tmp/a.jar,/tmp/b.jar");
    }

    #[test]
    fn apply_common_args_extra_confs_passed_through() {
        let mut cli = base_cli(Cmd::Ping);
        cli.conf.push("spark.executor.memory=4g".into());
        cli.conf.push("spark.executor.cores=2".into());
        let args = collect_args(&cli);
        assert!(args
            .windows(2)
            .any(|w| w[0] == "--conf" && w[1] == "spark.executor.memory=4g"));
        assert!(args
            .windows(2)
            .any(|w| w[0] == "--conf" && w[1] == "spark.executor.cores=2"));
    }

    #[test]
    fn build_request_json_tables_with_database() {
        let mut cli = base_cli(Cmd::Tables);
        cli.database = Some("analytics".into());
        let v = parse(&build_request_json(&cli).unwrap());
        assert_eq!(v["cmd"], "tables");
        assert_eq!(v["database"], "analytics");
    }

    #[test]
    fn build_request_json_execute_minimal() {
        let v = parse(&build_request_json(&base_cli(Cmd::Execute {
            sql: "DROP TABLE t".into(),
        }))
        .unwrap());
        assert_eq!(v["cmd"], "execute");
        assert_eq!(v["sql"], "DROP TABLE t");
    }

    #[test]
    fn apply_common_args_custom_app_name() {
        let mut cli = base_cli(Cmd::Ping);
        cli.app_name = "etl-job".into();
        let args = collect_args(&cli);
        let pos = args.iter().position(|a| a == "--name").unwrap();
        assert_eq!(args[pos + 1], "etl-job");
    }

    #[test]
    fn build_request_json_dump_limit_only() {
        let v = parse(&build_request_json(&base_cli(Cmd::Dump {
            table: "t".into(),
            columns: None,
            where_clause: None,
            order_by: None,
            limit: Some(1),
        }))
        .unwrap());
        assert_eq!(v["limit"], 1);
        assert!(!v.as_object().unwrap().contains_key("columns"));
    }

    #[test]
    fn locate_spark_submit_from_spark_home_when_set() {
        let dir = std::env::temp_dir().join(format!("stryke-spark-test-{}", std::process::id()));
        let bin_dir = dir.join("bin");
        std::fs::create_dir_all(&bin_dir).unwrap();
        let submit = bin_dir.join("spark-submit");
        std::fs::write(&submit, b"#!/bin/sh\n").unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&submit, std::fs::Permissions::from_mode(0o755)).unwrap();
        }
        let mut cli = base_cli(Cmd::Ping);
        cli.spark_home = Some(dir.clone());
        cli.spark_submit = None;
        let got = locate_spark_submit(&cli).unwrap();
        let _ = std::fs::remove_dir_all(&dir);
        assert_eq!(got, submit);
    }

    #[test]
    fn build_request_json_without_database_omits_key() {
        let v = parse(&build_request_json(&base_cli(Cmd::Ping)).unwrap());
        assert!(!v.as_object().unwrap().contains_key("database"));
    }

    #[test]
    fn build_request_json_query_sql_unicode() {
        let cli = base_cli(Cmd::Query {
            sql: "SELECT '日本語'".into(),
            columnar: false,
            with_meta: false,
            limit: None,
        });
        let v = parse(&build_request_json(&cli).unwrap());
        assert_eq!(v["sql"], "SELECT '日本語'");
    }

    #[test]
    fn build_request_json_dump_where_only() {
        let v = parse(&build_request_json(&base_cli(Cmd::Dump {
            table: "t".into(),
            columns: None,
            where_clause: Some("id > 0".into()),
            order_by: None,
            limit: None,
        }))
        .unwrap());
        assert_eq!(v["where"], "id > 0");
        assert!(!v.as_object().unwrap().contains_key("order_by"));
    }

    #[test]
    fn apply_common_args_both_defaults_suppressed_when_set() {
        let mut cli = base_cli(Cmd::Ping);
        cli.conf.push("spark.log.level=ERROR".into());
        cli.conf
            .push("spark.sql.catalogImplementation=hive".into());
        let args = collect_args(&cli);
        assert!(args
            .windows(2)
            .any(|w| w[0] == "--conf" && w[1] == "spark.log.level=ERROR"));
        assert!(args.windows(2).any(|w| {
            w[0] == "--conf" && w[1] == "spark.sql.catalogImplementation=hive"
        }));
        assert!(!args
            .windows(2)
            .any(|w| w[0] == "--conf" && w[1] == "spark.log.level=WARN"));
    }

    #[test]
    fn build_request_json_ping_has_no_sql_key() {
        let v = parse(&build_request_json(&base_cli(Cmd::Ping)).unwrap());
        assert!(!v.as_object().unwrap().contains_key("sql"));
    }

    #[test]
    fn apply_common_args_master_yarn() {
        let mut cli = base_cli(Cmd::Ping);
        cli.master = Some("yarn".into());
        let args = collect_args(&cli);
        let pos = args.iter().position(|a| a == "--master").unwrap();
        assert_eq!(args[pos + 1], "yarn");
    }

    #[test]
    fn build_request_json_schema_table_name() {
        let v = parse(&build_request_json(&base_cli(Cmd::Schema {
            table: "db.tbl".into(),
        }))
        .unwrap());
        assert_eq!(v["table"], "db.tbl");
    }

    #[test]
    fn collect_args_includes_jars_when_set() {
        let mut cli = base_cli(Cmd::Ping);
        cli.jars = Some("/a.jar,/b.jar".into());
        let args = collect_args(&cli);
        assert!(args.iter().any(|a| a == "--jars"));
    }

    #[test]
    fn build_request_json_execute_no_extra_keys() {
        let v = parse(&build_request_json(&base_cli(Cmd::Execute {
            sql: "VACUUM".into(),
        }))
        .unwrap());
        assert_eq!(v["cmd"], "execute");
        assert!(!v.as_object().unwrap().contains_key("columnar"));
    }

    #[test]
    fn apply_common_args_packages_when_set() {
        let mut cli = base_cli(Cmd::Ping);
        cli.packages = Some("org.apache.spark:pkg:1.0".into());
        let args = collect_args(&cli);
        assert!(args.windows(2).any(|w| w[0] == "--packages" && w[1].contains("spark")));
    }

    #[test]
    fn build_request_json_query_limit_zero() {
        let v = parse(&build_request_json(&base_cli(Cmd::Query {
            sql: "SELECT 1".into(),
            columnar: false,
            with_meta: false,
            limit: Some(0),
        }))
        .unwrap());
        assert_eq!(v["limit"], 0);
    }

    #[test]
    fn build_request_json_databases_cmd() {
        let v = parse(&build_request_json(&base_cli(Cmd::Databases)).unwrap());
        assert_eq!(v["cmd"], "databases");
    }

    #[test]
    fn apply_common_args_only_user_conf_no_defaults_when_both_set() {
        let mut cli = base_cli(Cmd::Ping);
        cli.conf = vec![
            "spark.log.level=INFO".into(),
            "spark.sql.catalogImplementation=hive".into(),
        ];
        let args = collect_args(&cli);
        let log_levels: Vec<_> = args
            .windows(2)
            .filter(|w| w[0] == "--conf" && w[1].starts_with("spark.log.level="))
            .map(|w| w[1].as_str())
            .collect();
        assert_eq!(log_levels, vec!["spark.log.level=INFO"]);
    }

    #[test]
    fn build_request_json_dump_columns_only() {
        let v = parse(&build_request_json(&base_cli(Cmd::Dump {
            table: "t".into(),
            columns: Some("a,b".into()),
            where_clause: None,
            order_by: None,
            limit: None,
        }))
        .unwrap());
        assert_eq!(v["columns"], "a,b");
    }

    #[test]
    fn collect_args_deploy_mode_client() {
        let mut cli = base_cli(Cmd::Ping);
        cli.deploy_mode = Some("client".into());
        let args = collect_args(&cli);
        let pos = args.iter().position(|a| a == "--deploy-mode").unwrap();
        assert_eq!(args[pos + 1], "client");
    }

    #[test]
    fn build_request_json_valid_json_string() {
        let s = build_request_json(&base_cli(Cmd::Ping)).unwrap();
        assert!(serde_json::from_str::<serde_json::Value>(&s).is_ok());
    }

    #[test]
    fn build_request_json_query_with_meta_flag() {
        let v = parse(&build_request_json(&base_cli(Cmd::Query {
            sql: "SELECT 1".into(),
            columnar: false,
            with_meta: true,
            limit: None,
        }))
        .unwrap());
        assert_eq!(v["with_meta"], true);
    }

    #[test]
    fn build_request_json_query_columnar_flag() {
        let v = parse(&build_request_json(&base_cli(Cmd::Query {
            sql: "SELECT 1".into(),
            columnar: true,
            with_meta: false,
            limit: None,
        }))
        .unwrap());
        assert_eq!(v["columnar"], true);
    }

    #[test]
    fn collect_args_includes_master_flag() {
        let args = collect_args(&base_cli(Cmd::Ping));
        assert!(args.iter().any(|a| a == "--master"));
    }

    #[test]
    fn build_request_json_dump_order_by_only() {
        let v = parse(&build_request_json(&base_cli(Cmd::Dump {
            table: "t".into(),
            columns: None,
            where_clause: None,
            order_by: Some("id DESC".into()),
            limit: None,
        }))
        .unwrap());
        assert_eq!(v["order_by"], "id DESC");
    }

    #[test]
    fn apply_common_args_empty_conf_still_has_defaults() {
        let cli = base_cli(Cmd::Ping);
        let args = collect_args(&cli);
        assert!(args.windows(2).any(|w| w[0] == "--conf" && w[1] == "spark.log.level=WARN"));
    }

    #[test]
    fn build_request_json_ping_cmd_only() {
        let v = parse(&build_request_json(&base_cli(Cmd::Ping)).unwrap());
        assert_eq!(v.as_object().unwrap().len(), 1);
        assert_eq!(v["cmd"], "ping");
    }

    #[test]
    fn build_request_json_execute_sql_unicode() {
        let v = parse(&build_request_json(&base_cli(Cmd::Execute {
            sql: "SELECT '日本語'".into(),
        }))
        .unwrap());
        assert_eq!(v["sql"], "SELECT '日本語'");
    }

    #[test]
    fn collect_args_master_local_star_default() {
        let args = collect_args(&base_cli(Cmd::Ping));
        let pos = args.iter().position(|a| a == "--master").unwrap();
        assert_eq!(args[pos + 1], "local[*]");
    }

    #[test]
    fn build_request_json_schema_cmd() {
        let v = parse(&build_request_json(&base_cli(Cmd::Schema {
            table: "t".into(),
        }))
        .unwrap());
        assert_eq!(v["cmd"], "schema");
        assert_eq!(v["table"], "t");
    }

    #[test]
    fn build_request_json_tables_cmd() {
        let v = parse(&build_request_json(&base_cli(Cmd::Tables)).unwrap());
        assert_eq!(v["cmd"], "tables");
    }

    #[test]
    fn apply_common_args_name_flag() {
        let args = collect_args(&base_cli(Cmd::Ping));
        assert!(args.windows(2).any(|w| w[0] == "--name" && w[1] == "stryke-spark"));
    }

    #[test]
    fn build_request_json_query_limit_some() {
        let v = parse(&build_request_json(&base_cli(Cmd::Query {
            sql: "SELECT 1".into(),
            columnar: false,
            with_meta: false,
            limit: Some(100),
        }))
        .unwrap());
        assert_eq!(v["limit"], 100);
    }

    #[test]
    fn collect_args_has_conf_flags() {
        let args = collect_args(&base_cli(Cmd::Ping));
        assert!(args.iter().filter(|a| *a == "--conf").count() >= 2);
    }

    #[test]
    fn build_request_json_execute_no_limit_key() {
        let v = parse(&build_request_json(&base_cli(Cmd::Execute {
            sql: "DELETE FROM t".into(),
        }))
        .unwrap());
        assert!(!v.as_object().unwrap().contains_key("limit"));
    }

    #[test]
    fn apply_common_args_deploy_mode_cluster() {
        let mut cli = base_cli(Cmd::Ping);
        cli.deploy_mode = Some("cluster".into());
        let args = collect_args(&cli);
        let pos = args.iter().position(|a| a == "--deploy-mode").unwrap();
        assert_eq!(args[pos + 1], "cluster");
    }

    #[test]
    fn build_request_json_query_columnar_true() {
        let v = parse(&build_request_json(&base_cli(Cmd::Query {
            sql: "SELECT 1".into(),
            columnar: true,
            with_meta: false,
            limit: None,
        }))
        .unwrap());
        assert_eq!(v["columnar"], true);
    }

    #[test]
    fn build_request_json_ping_no_table_key() {
        let v = parse(&build_request_json(&base_cli(Cmd::Ping)).unwrap());
        assert!(!v.as_object().unwrap().contains_key("table"));
    }

    #[test]
    fn apply_common_args_jars_flag() {
        let mut cli = base_cli(Cmd::Ping);
        cli.jars = Some("/x.jar".into());
        assert!(collect_args(&cli).windows(2).any(|w| w[0] == "--jars"));
    }

    #[test]
    fn build_request_json_dump_table_only() {
        let v = parse(&build_request_json(&base_cli(Cmd::Dump {
            table: "events".into(),
            columns: None,
            where_clause: None,
            order_by: None,
            limit: None,
        }))
        .unwrap());
        assert_eq!(v["table"], "events");
    }

    #[test]
    fn build_request_json_valid_roundtrip() {
        let s = build_request_json(&base_cli(Cmd::Tables)).unwrap();
        assert_eq!(serde_json::from_str::<serde_json::Value>(&s).unwrap()["cmd"], "tables");
    }

    #[test]
    fn collect_args_packages_when_set() {
        let mut cli = base_cli(Cmd::Ping);
        cli.packages = Some("g:a:1".into());
        assert!(collect_args(&cli).iter().any(|a| a == "--packages"));
    }

    #[test]
    fn build_request_json_execute_sql_only_keys() {
        let v = parse(&build_request_json(&base_cli(Cmd::Execute {
            sql: "TRUNCATE t".into(),
        }))
        .unwrap());
        let keys: Vec<_> = v.as_object().unwrap().keys().collect();
        assert_eq!(keys, vec!["cmd", "sql"]);
    }

    #[test]
    fn apply_common_args_custom_app_name_propagates() {
        let mut cli = base_cli(Cmd::Ping);
        cli.app_name = "job".into();
        let args = collect_args(&cli);
        let pos = args.iter().position(|a| a == "--name").unwrap();
        assert_eq!(args[pos + 1], "job");
    }
}
