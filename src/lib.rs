//! stryke-spark — Apache Spark cdylib loaded in-process by stryke via dlopen.
//!
//! Each `#[no_mangle] extern "C" fn spark__*` is a JSON-string-in /
//! JSON-string-out wrapper that spawns `spark-submit` with the embedded
//! PySpark driver. stryke's FFI bridge (`rust_ffi.rs::load_cdylib`)
//! resolves these symbols at first `use Spark`.
//!
//! Honest scope note: each call STILL pays SparkSession init cost
//! (seconds, dominated by the JVM warmup). A long-running JVM driver
//! daemon that persists `SparkSession` across calls is deferred to a
//! future revision — it needs a sidecar process design that's larger
//! than the v0.2.0 helper-binary → cdylib refactor. The cdylib model
//! does eliminate the helper-binary fork+exec overhead on top of
//! spark-submit, but the SparkSession init cost is unchanged.
//!
//! v0.2.0 ops: query, execute, dump, schema, tables, databases, ping,
//! submit.

use std::ffi::{CStr, CString};
use std::io::{Read, Write};
use std::os::raw::c_char;
use std::panic::AssertUnwindSafe;
use std::path::PathBuf;
use std::process::{Command, Stdio};

use anyhow::{anyhow, Context, Result};
use serde_json::{json, Value};

const DRIVER_PY: &str = include_str!("driver.py");

// ── connection / spark-submit args ──────────────────────────────────────────

struct SparkOpts {
    master: Option<String>,
    app_name: Option<String>,
    deploy_mode: Option<String>,
    packages: Option<String>,
    jars: Option<String>,
    confs: Vec<String>,
    database: Option<String>,
    spark_submit: Option<String>,
    spark_home: Option<String>,
}

impl SparkOpts {
    fn from_value(opts: &Value) -> Self {
        SparkOpts {
            master: opts
                .get("master")
                .and_then(|v| v.as_str())
                .map(String::from),
            app_name: opts
                .get("app_name")
                .and_then(|v| v.as_str())
                .map(String::from),
            deploy_mode: opts
                .get("deploy_mode")
                .and_then(|v| v.as_str())
                .map(String::from),
            packages: opts
                .get("packages")
                .and_then(|v| v.as_str())
                .map(String::from),
            jars: opts.get("jars").and_then(|v| v.as_str()).map(String::from),
            confs: opts
                .get("confs")
                .and_then(|v| v.as_array())
                .map(|a| {
                    a.iter()
                        .filter_map(|x| x.as_str().map(String::from))
                        .collect()
                })
                .unwrap_or_default(),
            database: opts
                .get("database")
                .and_then(|v| v.as_str())
                .map(String::from),
            spark_submit: opts
                .get("spark_submit")
                .and_then(|v| v.as_str())
                .map(String::from)
                .or_else(|| std::env::var("STRYKE_SPARK_SUBMIT").ok()),
            spark_home: opts
                .get("spark_home")
                .and_then(|v| v.as_str())
                .map(String::from)
                .or_else(|| std::env::var("SPARK_HOME").ok()),
        }
    }

    fn locate_spark_submit(&self) -> Result<PathBuf> {
        if let Some(p) = &self.spark_submit {
            let path = PathBuf::from(p);
            if path.is_file() {
                return Ok(path);
            }
            return Err(anyhow!("spark_submit `{}` is not a file", path.display()));
        }
        if let Some(home) = &self.spark_home {
            let path = PathBuf::from(home).join("bin").join("spark-submit");
            if path.is_file() {
                return Ok(path);
            }
        }
        // Fall back to $PATH.
        if let Ok(path_env) = std::env::var("PATH") {
            for dir in path_env.split(':') {
                let candidate = PathBuf::from(dir).join("spark-submit");
                if candidate.is_file() {
                    return Ok(candidate);
                }
            }
        }
        Err(anyhow!(
            "spark-submit not found — set `spark_home` opt, or pass `spark_submit`, or put it on $PATH"
        ))
    }

    fn apply_to(&self, cmd: &mut Command) {
        let mut user_log_level = false;
        let mut user_catalog_impl = false;
        for c in &self.confs {
            cmd.arg("--conf").arg(c);
            if c.starts_with("spark.log.level=") {
                user_log_level = true;
            }
            if c.starts_with("spark.sql.catalogImplementation=") {
                user_catalog_impl = true;
            }
        }
        if !user_log_level {
            cmd.arg("--conf").arg("spark.log.level=ERROR");
        }
        if !user_catalog_impl {
            cmd.arg("--conf")
                .arg("spark.sql.catalogImplementation=in-memory");
        }
        if let Some(m) = &self.master {
            cmd.arg("--master").arg(m);
        } else {
            cmd.arg("--master").arg("local[*]");
        }
        cmd.arg("--name")
            .arg(self.app_name.as_deref().unwrap_or("stryke-spark"));
        if let Some(d) = &self.deploy_mode {
            cmd.arg("--deploy-mode").arg(d);
        }
        if let Some(p) = &self.packages {
            cmd.arg("--packages").arg(p);
        }
        if let Some(j) = &self.jars {
            cmd.arg("--jars").arg(j);
        }
    }
}

// ── core call ───────────────────────────────────────────────────────────────

/// Spawn spark-submit with the embedded driver + a JSON request envelope.
/// Captures stdout (NDJSON) and returns it as a `String`.
fn run_driver(opts: &SparkOpts, request: Value) -> Result<String> {
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

    let submit = opts.locate_spark_submit()?;
    let mut cmd = Command::new(&submit);
    opts.apply_to(&mut cmd);
    cmd.arg(&driver_path);
    cmd.arg(serde_json::to_string(&request)?);
    cmd.stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit());
    let mut child = cmd
        .spawn()
        .with_context(|| format!("spawning {}", submit.display()))?;
    let mut out = String::new();
    if let Some(mut stdout) = child.stdout.take() {
        stdout.read_to_string(&mut out)?;
    }
    let status = child.wait()?;
    if !status.success() {
        return Err(anyhow!(
            "spark-submit exited with {} — see stderr for driver output",
            status
        ));
    }
    drop(driver_file);
    Ok(out)
}

fn ndjson_to_rows(buf: &str) -> Result<Vec<Value>> {
    buf.lines()
        .filter(|l| !l.is_empty())
        .map(|l| serde_json::from_str::<Value>(l).map_err(Into::into))
        .collect()
}

// ── ops ─────────────────────────────────────────────────────────────────────

fn build_request(opts: &SparkOpts, body: Value) -> Value {
    let mut env = serde_json::Map::new();
    if let Value::Object(map) = body {
        for (k, v) in map {
            env.insert(k, v);
        }
    }
    if let Some(db) = &opts.database {
        env.insert("database".into(), Value::String(db.clone()));
    }
    Value::Object(env)
}

fn op_ping(opts: Value) -> Result<Value> {
    let so = SparkOpts::from_value(&opts);
    let req = build_request(&so, json!({"cmd": "ping"}));
    let out = run_driver(&so, req)?;
    let rows = ndjson_to_rows(&out)?;
    Ok(json!({"ok": true, "rows": rows}))
}

fn op_query(opts: Value) -> Result<Value> {
    let so = SparkOpts::from_value(&opts);
    let sql = opts["sql"]
        .as_str()
        .ok_or_else(|| anyhow!("missing sql"))?
        .to_string();
    let mut body = serde_json::Map::new();
    body.insert("cmd".into(), Value::String("query".into()));
    body.insert("sql".into(), Value::String(sql));
    if let Some(n) = opts["limit"].as_u64() {
        body.insert("limit".into(), Value::from(n));
    }
    let req = build_request(&so, Value::Object(body));
    let out = run_driver(&so, req)?;
    let rows = ndjson_to_rows(&out)?;
    Ok(json!({"rows": rows}))
}

fn op_execute(opts: Value) -> Result<Value> {
    let so = SparkOpts::from_value(&opts);
    let sql = opts["sql"]
        .as_str()
        .ok_or_else(|| anyhow!("missing sql"))?
        .to_string();
    let req = build_request(&so, json!({"cmd": "execute", "sql": sql}));
    let out = run_driver(&so, req)?;
    let rows = ndjson_to_rows(&out)?;
    Ok(json!({"rows": rows}))
}

fn op_dump(opts: Value) -> Result<Value> {
    let so = SparkOpts::from_value(&opts);
    let table = opts["table"]
        .as_str()
        .ok_or_else(|| anyhow!("missing table"))?
        .to_string();
    let mut body = serde_json::Map::new();
    body.insert("cmd".into(), Value::String("dump".into()));
    body.insert("table".into(), Value::String(table));
    if let Some(c) = opts["columns"].as_str() {
        body.insert("columns".into(), Value::String(c.into()));
    }
    if let Some(w) = opts["where"].as_str() {
        body.insert("where".into(), Value::String(w.into()));
    }
    if let Some(o) = opts["order_by"].as_str() {
        body.insert("order_by".into(), Value::String(o.into()));
    }
    if let Some(n) = opts["limit"].as_u64() {
        body.insert("limit".into(), Value::from(n));
    }
    let req = build_request(&so, Value::Object(body));
    let out = run_driver(&so, req)?;
    let rows = ndjson_to_rows(&out)?;
    Ok(json!({"rows": rows}))
}

fn op_tables(opts: Value) -> Result<Value> {
    let so = SparkOpts::from_value(&opts);
    let req = build_request(&so, json!({"cmd": "tables"}));
    let out = run_driver(&so, req)?;
    let rows = ndjson_to_rows(&out)?;
    Ok(json!({"rows": rows}))
}

fn op_databases(opts: Value) -> Result<Value> {
    let so = SparkOpts::from_value(&opts);
    let req = build_request(&so, json!({"cmd": "databases"}));
    let out = run_driver(&so, req)?;
    let rows = ndjson_to_rows(&out)?;
    Ok(json!({"rows": rows}))
}

fn op_schema(opts: Value) -> Result<Value> {
    let so = SparkOpts::from_value(&opts);
    let table = opts["table"]
        .as_str()
        .ok_or_else(|| anyhow!("missing table"))?
        .to_string();
    let req = build_request(&so, json!({"cmd": "schema", "table": table}));
    let out = run_driver(&so, req)?;
    let rows = ndjson_to_rows(&out)?;
    Ok(json!({"rows": rows}))
}

fn op_submit(opts: Value) -> Result<Value> {
    let so = SparkOpts::from_value(&opts);
    let script = opts["script"]
        .as_str()
        .ok_or_else(|| anyhow!("missing script"))?
        .to_string();
    let args: Vec<String> = opts["args"]
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default();
    let submit = so.locate_spark_submit()?;
    let mut cmd = Command::new(&submit);
    so.apply_to(&mut cmd);
    cmd.arg(&script);
    cmd.args(&args);
    cmd.stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit());
    let mut child = cmd
        .spawn()
        .with_context(|| format!("spawning {}", submit.display()))?;
    let mut out = String::new();
    if let Some(mut stdout) = child.stdout.take() {
        stdout.read_to_string(&mut out)?;
    }
    let status = child.wait()?;
    Ok(json!({
        "script": script,
        "exit_code": status.code(),
        "output": out,
    }))
}

// ── FFI plumbing ────────────────────────────────────────────────────────────

fn ffi_call<F>(args: *const c_char, handler: F) -> *const c_char
where
    F: FnOnce(Value) -> Result<Value>,
{
    let input = if args.is_null() {
        Value::Null
    } else {
        let cs = unsafe { CStr::from_ptr(args) };
        serde_json::from_slice::<Value>(cs.to_bytes()).unwrap_or(Value::Null)
    };
    let result = std::panic::catch_unwind(AssertUnwindSafe(|| handler(input)));
    let out = match result {
        Ok(Ok(v)) => v,
        Ok(Err(e)) => json!({ "error": e.to_string() }),
        Err(_) => json!({ "error": "stryke-spark handler panicked" }),
    };
    let s =
        serde_json::to_string(&out).unwrap_or_else(|_| String::from(r#"{"error":"serialize"}"#));
    match CString::new(s) {
        Ok(c) => c.into_raw() as *const c_char,
        Err(_) => std::ptr::null(),
    }
}

/// Free a C string allocated by any export from this cdylib.
///
/// # Safety
///
/// `p` must be a pointer previously returned by an export from this cdylib,
/// or null.
#[no_mangle]
pub unsafe extern "C" fn stryke_free_cstring(p: *mut c_char) {
    if p.is_null() {
        return;
    }
    drop(CString::from_raw(p));
}

// ── exports ─────────────────────────────────────────────────────────────────

#[no_mangle]
pub extern "C" fn spark__pkg_version(args: *const c_char) -> *const c_char {
    ffi_call(args, |_| Ok(json!({"version": env!("CARGO_PKG_VERSION")})))
}

#[no_mangle]
pub extern "C" fn spark__ping(args: *const c_char) -> *const c_char {
    ffi_call(args, op_ping)
}

#[no_mangle]
pub extern "C" fn spark__query(args: *const c_char) -> *const c_char {
    ffi_call(args, op_query)
}

#[no_mangle]
pub extern "C" fn spark__execute(args: *const c_char) -> *const c_char {
    ffi_call(args, op_execute)
}

#[no_mangle]
pub extern "C" fn spark__dump(args: *const c_char) -> *const c_char {
    ffi_call(args, op_dump)
}

#[no_mangle]
pub extern "C" fn spark__tables(args: *const c_char) -> *const c_char {
    ffi_call(args, op_tables)
}

#[no_mangle]
pub extern "C" fn spark__databases(args: *const c_char) -> *const c_char {
    ffi_call(args, op_databases)
}

#[no_mangle]
pub extern "C" fn spark__schema(args: *const c_char) -> *const c_char {
    ffi_call(args, op_schema)
}

#[no_mangle]
pub extern "C" fn spark__submit(args: *const c_char) -> *const c_char {
    ffi_call(args, op_submit)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    fn with_env<F: FnOnce()>(f: F) {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let saved = [
            (
                "STRYKE_SPARK_SUBMIT",
                std::env::var("STRYKE_SPARK_SUBMIT").ok(),
            ),
            ("SPARK_HOME", std::env::var("SPARK_HOME").ok()),
        ];
        std::env::remove_var("STRYKE_SPARK_SUBMIT");
        std::env::remove_var("SPARK_HOME");
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));
        for (k, v) in &saved {
            match v {
                Some(s) => std::env::set_var(k, s),
                None => std::env::remove_var(k),
            }
        }
        if let Err(p) = result {
            std::panic::resume_unwind(p);
        }
    }

    // ── SparkOpts::from_value ──

    #[test]
    fn opts_defaults_all_none() {
        with_env(|| {
            let o = SparkOpts::from_value(&json!({}));
            assert_eq!(o.master, None);
            assert_eq!(o.app_name, None);
            assert!(o.confs.is_empty());
            assert_eq!(o.spark_submit, None);
            assert_eq!(o.spark_home, None);
        });
    }

    #[test]
    fn opts_full_overrides_round_trip() {
        with_env(|| {
            let o = SparkOpts::from_value(&json!({
                "master": "yarn",
                "app_name": "demo",
                "deploy_mode": "client",
                "packages": "org.apache.iceberg:iceberg-spark:1.4.0",
                "jars": "/lib/foo.jar",
                "confs": ["spark.executor.memory=4g", "spark.cores.max=8"],
                "database": "default",
            }));
            assert_eq!(o.master.as_deref(), Some("yarn"));
            assert_eq!(o.app_name.as_deref(), Some("demo"));
            assert_eq!(o.deploy_mode.as_deref(), Some("client"));
            assert_eq!(
                o.packages.as_deref(),
                Some("org.apache.iceberg:iceberg-spark:1.4.0")
            );
            assert_eq!(o.jars.as_deref(), Some("/lib/foo.jar"));
            assert_eq!(o.confs.len(), 2);
            assert_eq!(o.database.as_deref(), Some("default"));
        });
    }

    #[test]
    fn opts_spark_submit_from_env_when_not_in_opts() {
        with_env(|| {
            std::env::set_var("STRYKE_SPARK_SUBMIT", "/from/env/spark-submit");
            let o = SparkOpts::from_value(&json!({}));
            assert_eq!(o.spark_submit.as_deref(), Some("/from/env/spark-submit"));
        });
    }

    #[test]
    fn opts_spark_submit_opts_wins_over_env() {
        with_env(|| {
            std::env::set_var("STRYKE_SPARK_SUBMIT", "/from/env");
            let o = SparkOpts::from_value(&json!({"spark_submit": "/from/opts"}));
            assert_eq!(o.spark_submit.as_deref(), Some("/from/opts"));
        });
    }

    #[test]
    fn opts_spark_home_from_env() {
        with_env(|| {
            std::env::set_var("SPARK_HOME", "/opt/spark");
            let o = SparkOpts::from_value(&json!({}));
            assert_eq!(o.spark_home.as_deref(), Some("/opt/spark"));
        });
    }

    // ── locate_spark_submit ──

    #[test]
    fn locate_explicit_nonfile_errors() {
        with_env(|| {
            let o = SparkOpts::from_value(&json!({"spark_submit": "/does/not/exist"}));
            let err = o.locate_spark_submit().unwrap_err().to_string();
            assert!(err.contains("not a file"), "{err}");
        });
    }

    #[test]
    fn locate_no_env_no_opts_no_path_errors() {
        with_env(|| {
            std::env::set_var("PATH", "/no/spark/here");
            let o = SparkOpts::from_value(&json!({}));
            let err = o.locate_spark_submit().unwrap_err().to_string();
            assert!(err.contains("spark-submit not found"), "{err}");
        });
    }

    #[test]
    fn locate_picks_explicit_when_file_exists() {
        with_env(|| {
            // Create a real temp file we know is_file() == true.
            let f = tempfile::NamedTempFile::new().unwrap();
            let p = f.path().to_path_buf();
            let o = SparkOpts::from_value(&json!({"spark_submit": p.to_str().unwrap()}));
            assert_eq!(o.locate_spark_submit().unwrap(), p);
        });
    }

    // ── build_request ──

    #[test]
    fn build_request_no_database_passes_body_through() {
        with_env(|| {
            let o = SparkOpts::from_value(&json!({}));
            let req = build_request(&o, json!({"cmd": "ping"}));
            assert_eq!(req["cmd"], json!("ping"));
            assert!(req.get("database").is_none());
        });
    }

    #[test]
    fn build_request_database_overrides_body() {
        with_env(|| {
            // Opts.database overrides whatever was in the body's database
            // field — the option wins as the canonical setter.
            let o = SparkOpts::from_value(&json!({"database": "shop"}));
            let req = build_request(&o, json!({"cmd": "query", "database": "other"}));
            assert_eq!(req["database"], json!("shop"));
            assert_eq!(req["cmd"], json!("query"));
        });
    }

    #[test]
    fn build_request_non_object_body_yields_only_database() {
        with_env(|| {
            let o = SparkOpts::from_value(&json!({"database": "d"}));
            let req = build_request(&o, json!(null));
            assert_eq!(req["database"], json!("d"));
        });
    }

    // ── ndjson_to_rows ──

    #[test]
    fn ndjson_multi_line() {
        let buf = "{\"a\":1}\n{\"a\":2}\n";
        let r = ndjson_to_rows(buf).unwrap();
        assert_eq!(r.len(), 2);
        assert_eq!(r[1]["a"], json!(2));
    }

    #[test]
    fn ndjson_skips_blank() {
        let r = ndjson_to_rows("{\"a\":1}\n\n{\"a\":2}\n\n").unwrap();
        assert_eq!(r.len(), 2);
    }

    #[test]
    fn ndjson_empty_yields_empty() {
        assert!(ndjson_to_rows("").unwrap().is_empty());
        assert!(ndjson_to_rows("\n\n").unwrap().is_empty());
    }

    #[test]
    fn ndjson_invalid_line_errors() {
        assert!(ndjson_to_rows("{\"a\":1}\ngarbage\n").is_err());
    }
}
