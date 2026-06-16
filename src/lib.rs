//! stryke-spark — Apache Spark cdylib loaded in-process by stryke via dlopen.
//!
//! Each `#[no_mangle] extern "C" fn spark__*` is a JSON-string-in /
//! JSON-string-out wrapper that spawns `spark-submit` with the embedded
//! PySpark driver. stryke's FFI bridge (`rust_ffi.rs::load_cdylib`)
//! resolves these symbols at first `use Spark`.
//!
//! Scope: each call STILL pays SparkSession init cost (seconds, dominated
//! by JVM warmup). A long-running JVM driver daemon that persists
//! `SparkSession` across calls is deferred — it needs a sidecar process
//! design larger than the helper-binary → cdylib refactor. The cdylib model
//! does eliminate the helper-binary fork+exec overhead on top of
//! spark-submit, but the SparkSession init cost is unchanged.
//!
//! Ops: query/dump/execute/explain SQL, read/write external sources
//! (parquet/csv/json/orc), catalog metadata (tables/databases/schema/columns/
//! functions), cache/uncache, runtime config get/set, ping, and a raw
//! spark-submit pass-through.

use std::ffi::{CStr, CString};
use std::io::{Read, Write};
use std::os::raw::c_char;
use std::panic::AssertUnwindSafe;
use std::path::PathBuf;
use std::process::{Command, Stdio};

use anyhow::{anyhow, bail, Context, Result};
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

fn op_views(opts: Value) -> Result<Value> {
    let so = SparkOpts::from_value(&opts);
    let req = build_request(&so, json!({"cmd": "views"}));
    let out = run_driver(&so, req)?;
    let rows = ndjson_to_rows(&out)?;
    Ok(json!({"rows": rows}))
}

fn op_catalogs(opts: Value) -> Result<Value> {
    let so = SparkOpts::from_value(&opts);
    let req = build_request(&so, json!({"cmd": "catalogs"}));
    let out = run_driver(&so, req)?;
    let rows = ndjson_to_rows(&out)?;
    Ok(json!({"rows": rows}))
}

fn op_current_database(opts: Value) -> Result<Value> {
    let so = SparkOpts::from_value(&opts);
    let req = build_request(&so, json!({"cmd": "current_database"}));
    let out = run_driver(&so, req)?;
    let rows = ndjson_to_rows(&out)?;
    // Single-row { database } response.
    Ok(json!({"database": rows.first().and_then(|r| r.get("database")).cloned()}))
}

/// Drive a single-row command that takes one extra string field (e.g. a
/// temp-view name, database, or table) and returns the driver's ok-row.
fn op_single(opts: Value, cmd: &str, field: &str, extra: &[&str]) -> Result<Value> {
    let so = SparkOpts::from_value(&opts);
    let val = opts[field]
        .as_str()
        .ok_or_else(|| anyhow!("missing {}", field))?
        .to_string();
    let mut body = json!({"cmd": cmd, field: val});
    for e in extra {
        if let Some(s) = opts[*e].as_str() {
            body[*e] = json!(s);
        }
    }
    let out = run_driver(&so, build_request(&so, body))?;
    let rows = ndjson_to_rows(&out)?;
    Ok(rows.into_iter().next().unwrap_or(json!({"ok": true})))
}

fn op_create_temp_view(opts: Value) -> Result<Value> {
    op_single(opts, "create_temp_view", "name", &["sql"])
}

fn op_drop_temp_view(opts: Value) -> Result<Value> {
    op_single(opts, "drop_temp_view", "name", &[])
}

fn op_set_database(opts: Value) -> Result<Value> {
    op_single(opts, "set_database", "database", &[])
}

fn op_refresh_table(opts: Value) -> Result<Value> {
    op_single(opts, "refresh_table", "table", &[])
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

/// Copy an optional string field from `opts` into `body` under the same key.
fn copy_str(opts: &Value, body: &mut serde_json::Map<String, Value>, key: &str) {
    if let Some(s) = opts[key].as_str() {
        body.insert(key.into(), Value::String(s.into()));
    }
}

fn op_explain(opts: Value) -> Result<Value> {
    let so = SparkOpts::from_value(&opts);
    let sql = opts["sql"]
        .as_str()
        .ok_or_else(|| anyhow!("missing sql"))?
        .to_string();
    let mut body = serde_json::Map::new();
    body.insert("cmd".into(), Value::String("explain".into()));
    body.insert("sql".into(), Value::String(sql));
    copy_str(&opts, &mut body, "mode");
    let req = build_request(&so, Value::Object(body));
    let out = run_driver(&so, req)?;
    let rows = ndjson_to_rows(&out)?;
    // explain emits one summary line: { plan: "..." }.
    Ok(rows
        .into_iter()
        .next()
        .unwrap_or_else(|| json!({"plan": ""})))
}

fn op_read(opts: Value) -> Result<Value> {
    let so = SparkOpts::from_value(&opts);
    let path = opts["path"]
        .as_str()
        .ok_or_else(|| anyhow!("missing path"))?
        .to_string();
    let mut body = serde_json::Map::new();
    body.insert("cmd".into(), Value::String("read".into()));
    body.insert("path".into(), Value::String(path));
    copy_str(&opts, &mut body, "format");
    copy_str(&opts, &mut body, "view");
    copy_str(&opts, &mut body, "sql");
    if opts["options"].is_object() {
        body.insert("options".into(), opts["options"].clone());
    }
    if let Some(n) = opts["limit"].as_u64() {
        body.insert("limit".into(), Value::from(n));
    }
    let req = build_request(&so, Value::Object(body));
    let out = run_driver(&so, req)?;
    let rows = ndjson_to_rows(&out)?;
    Ok(json!({"rows": rows}))
}

fn op_write(opts: Value) -> Result<Value> {
    let so = SparkOpts::from_value(&opts);
    let sql = opts["sql"]
        .as_str()
        .ok_or_else(|| anyhow!("missing sql (query producing the data to write)"))?
        .to_string();
    // Exactly one of `path` / `table` is the write target.
    if opts["path"].as_str().is_none() && opts["table"].as_str().is_none() {
        return Err(anyhow!("write needs a `path` or a `table` target"));
    }
    let mut body = serde_json::Map::new();
    body.insert("cmd".into(), Value::String("write".into()));
    body.insert("sql".into(), Value::String(sql));
    copy_str(&opts, &mut body, "path");
    copy_str(&opts, &mut body, "table");
    copy_str(&opts, &mut body, "format");
    copy_str(&opts, &mut body, "mode");
    if opts["options"].is_object() {
        body.insert("options".into(), opts["options"].clone());
    }
    let req = build_request(&so, Value::Object(body));
    let out = run_driver(&so, req)?;
    let rows = ndjson_to_rows(&out)?;
    Ok(rows
        .into_iter()
        .next()
        .unwrap_or_else(|| json!({"ok": true})))
}

fn op_columns(opts: Value) -> Result<Value> {
    let so = SparkOpts::from_value(&opts);
    let table = opts["table"]
        .as_str()
        .ok_or_else(|| anyhow!("missing table"))?
        .to_string();
    let req = build_request(&so, json!({"cmd": "columns", "table": table}));
    let out = run_driver(&so, req)?;
    Ok(json!({"rows": ndjson_to_rows(&out)?}))
}

fn op_functions(opts: Value) -> Result<Value> {
    let so = SparkOpts::from_value(&opts);
    let req = build_request(&so, json!({"cmd": "functions"}));
    let out = run_driver(&so, req)?;
    Ok(json!({"rows": ndjson_to_rows(&out)?}))
}

/// Shared body for cache / uncache — both take a single `table`.
fn op_cache_like(opts: Value, cmd: &str) -> Result<Value> {
    let so = SparkOpts::from_value(&opts);
    let table = opts["table"]
        .as_str()
        .ok_or_else(|| anyhow!("missing table"))?
        .to_string();
    let req = build_request(&so, json!({"cmd": cmd, "table": table}));
    let out = run_driver(&so, req)?;
    let rows = ndjson_to_rows(&out)?;
    Ok(rows
        .into_iter()
        .next()
        .unwrap_or_else(|| json!({"ok": true})))
}

fn op_config(opts: Value) -> Result<Value> {
    let so = SparkOpts::from_value(&opts);
    let key = opts["key"]
        .as_str()
        .ok_or_else(|| anyhow!("missing key"))?
        .to_string();
    let mut body = serde_json::Map::new();
    body.insert("cmd".into(), Value::String("config".into()));
    body.insert("key".into(), Value::String(key));
    // Presence of `value` switches the driver from get to set.
    if !opts["value"].is_null() {
        body.insert("value".into(), opts["value"].clone());
    }
    let req = build_request(&so, Value::Object(body));
    let out = run_driver(&so, req)?;
    let rows = ndjson_to_rows(&out)?;
    Ok(rows
        .into_iter()
        .next()
        .unwrap_or_else(|| json!({"ok": true})))
}

/// Coerce a JSON-array `args` slot into a list of CLI args for
/// `spark-submit`. Pre-fix the caller used
/// `arr.iter().filter_map(|v| v.as_str())` which silently dropped
/// non-strings — `["--num", 42, "--flag"]` became `["--num", "--flag"]`
/// and the user got a confusing failure deeper in spark.
///
/// New contract: non-strings are coerced to their canonical text form
/// (bool/number → `to_string`, array/object → JSON); `null` is a hard
/// error since it can't be meaningfully passed to a CLI.
fn coerce_submit_args(arr: &[Value]) -> Result<Vec<String>> {
    let mut out = Vec::with_capacity(arr.len());
    for (i, v) in arr.iter().enumerate() {
        match v {
            Value::String(s) => out.push(s.clone()),
            Value::Bool(b) => out.push(b.to_string()),
            Value::Number(n) => out.push(n.to_string()),
            Value::Null => bail!("args[{i}] is null — spark-submit args must be scalar"),
            Value::Array(_) | Value::Object(_) => out.push(v.to_string()),
        }
    }
    Ok(out)
}

fn op_submit(opts: Value) -> Result<Value> {
    let so = SparkOpts::from_value(&opts);
    let script = opts["script"]
        .as_str()
        .ok_or_else(|| anyhow!("missing script"))?
        .to_string();
    let args: Vec<String> = match opts["args"].as_array() {
        Some(arr) => coerce_submit_args(arr)?,
        None => Vec::new(),
    };
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

// ── pure helpers (no Spark) ──────────────────────────────────────────────────

/// Parse a Spark master URL into its parts. Handles `local`, `local[N]`,
/// `local[*]`, `yarn`, `spark://host:port[,host:port]` (standalone HA), and
/// scheme-prefixed forms (`k8s://…`, `mesos://…`). Pure.
fn op_parse_master_url(opts: Value) -> Result<Value> {
    let url = opts
        .get("url")
        .or_else(|| opts.get("master"))
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("missing url"))?;
    if url == "local" {
        return Ok(json!({"scheme": "local", "threads": 1}));
    }
    if let Some(inner) = url.strip_prefix("local[").and_then(|s| s.strip_suffix(']')) {
        let threads = if inner == "*" {
            json!("*")
        } else {
            inner
                .parse::<u32>()
                .map(|n| json!(n))
                .map_err(|_| anyhow!("invalid local thread count `{inner}`"))?
        };
        return Ok(json!({"scheme": "local", "threads": threads}));
    }
    if url == "yarn" {
        return Ok(json!({"scheme": "yarn"}));
    }
    let (scheme, rest) = url
        .split_once("://")
        .ok_or_else(|| anyhow!("not a Spark master URL: {url}"))?;
    match scheme {
        "spark" => {
            let hosts: Vec<Value> = rest
                .split(',')
                .filter(|s| !s.is_empty())
                .map(|hp| match hp.rsplit_once(':') {
                    Some((h, p)) => match p.parse::<u32>() {
                        Ok(port) => json!({"host": h, "port": port}),
                        Err(_) => json!({"host": hp, "port": Value::Null}),
                    },
                    None => json!({"host": hp, "port": Value::Null}),
                })
                .collect();
            Ok(json!({"scheme": "spark", "hosts": hosts}))
        }
        other => Ok(json!({"scheme": other, "master": rest})),
    }
}

/// Build a Spark master URL from parts — the inverse of `parse_master_url`.
/// opts: `scheme` (required). `local` honors `threads` (a count, or `"*"`; 1 or
/// absent yields the bare `local`); `spark` joins `hosts` (`[{host, port?}]`)
/// into `spark://h1:p1,h2`; `yarn` is the bare word; any other scheme needs a
/// `master` string and yields `scheme://master`. Returns `{url}`. Pure.
fn op_build_master_url(opts: Value) -> Result<Value> {
    let scheme = opts
        .get("scheme")
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("missing scheme"))?;
    let url = match scheme {
        "local" => match opts.get("threads") {
            Some(Value::String(s)) if s == "*" => "local[*]".to_string(),
            Some(Value::Number(n)) => {
                let t = n.as_u64().ok_or_else(|| anyhow!("invalid thread count"))?;
                if t <= 1 {
                    "local".to_string()
                } else {
                    format!("local[{t}]")
                }
            }
            None => "local".to_string(),
            Some(other) => return Err(anyhow!("invalid threads value: {other}")),
        },
        "yarn" => "yarn".to_string(),
        "spark" => {
            let hosts = opts
                .get("hosts")
                .and_then(Value::as_array)
                .filter(|h| !h.is_empty())
                .ok_or_else(|| anyhow!("spark scheme requires a non-empty hosts list"))?;
            let parts: Result<Vec<String>> = hosts
                .iter()
                .map(|h| {
                    let host = h
                        .get("host")
                        .and_then(Value::as_str)
                        .filter(|s| !s.is_empty())
                        .ok_or_else(|| anyhow!("host entry missing `host`"))?;
                    match h.get("port").and_then(Value::as_u64) {
                        Some(port) => Ok(format!("{host}:{port}")),
                        None => Ok(host.to_string()),
                    }
                })
                .collect();
            format!("spark://{}", parts?.join(","))
        }
        other => {
            let master = opts
                .get("master")
                .and_then(Value::as_str)
                .ok_or_else(|| anyhow!("`{other}` scheme requires a master address"))?;
            format!("{other}://{master}")
        }
    };
    Ok(json!({ "url": url }))
}

/// Split a possibly-backtick-quoted dotted identifier, treating `.` inside
/// backticks as literal and `` `` `` (doubled) as an escaped backtick.
fn split_qualified(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut in_tick = false;
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '`' => {
                if in_tick && chars.peek() == Some(&'`') {
                    cur.push('`');
                    chars.next();
                } else {
                    in_tick = !in_tick;
                }
            }
            '.' if !in_tick => out.push(std::mem::take(&mut cur)),
            _ => cur.push(c),
        }
    }
    out.push(cur);
    out
}

/// Parse a Spark table identifier `[catalog.][database.]table` (Spark 3
/// three-level namespace) into `{catalog, database, table, parts}`. Backtick
/// quoting is honored, so a `.` inside backticks stays in one part. Pure.
fn op_parse_table_name(opts: Value) -> Result<Value> {
    let name = opts
        .get("name")
        .or_else(|| opts.get("table"))
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("missing name"))?;
    let parts = split_qualified(name);
    let (catalog, database, table) = match parts.len() {
        1 => (Value::Null, Value::Null, json!(parts[0])),
        2 => (Value::Null, json!(parts[0]), json!(parts[1])),
        3 => (json!(parts[0]), json!(parts[1]), json!(parts[2])),
        _ => {
            return Err(anyhow!(
                "too many parts (max catalog.database.table): {name}"
            ))
        }
    };
    Ok(json!({"catalog": catalog, "database": database, "table": table, "parts": parts}))
}

/// Backtick-quote a single Spark SQL identifier, doubling embedded backticks.
fn quote_ident_str(name: &str) -> String {
    format!("`{}`", name.replace('`', "``"))
}

/// Build a qualified table name from parts — the inverse of `parse_table_name`.
/// opts: `table` (required), and optional `database` and `catalog` (a catalog
/// requires a database). A segment is backtick-quoted only when it contains a
/// `.` or backtick (the chars that would break the namespace split), so plain
/// names stay clean and the result round-trips through `parse_table_name`. Pure.
fn op_build_table_name(opts: Value) -> Result<Value> {
    let req = |k: &str| {
        opts.get(k)
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
    };
    let table = req("table").ok_or_else(|| anyhow!("missing table"))?;
    let database = req("database");
    let catalog = req("catalog");
    if catalog.is_some() && database.is_none() {
        return Err(anyhow!("catalog requires a database"));
    }
    // Quote a segment only if it would otherwise break the dotted split.
    let seg = |s: &str| {
        if s.contains('.') || s.contains('`') {
            quote_ident_str(s)
        } else {
            s.to_string()
        }
    };
    let mut parts: Vec<String> = Vec::new();
    if let Some(c) = catalog {
        parts.push(seg(c));
    }
    if let Some(d) = database {
        parts.push(seg(d));
    }
    parts.push(seg(table));
    Ok(json!({"name": parts.join(".")}))
}

/// Parse a Spark memory/size config string (`512m`, `2g`, `1024kb`) into bytes.
/// Spark's `JavaUtils.byteStringAsBytes` treats every suffix as BINARY —
/// `1k`/`1kb`/`1ki`/`1kib` all mean 1024 bytes, not 1000 — and is
/// case-insensitive; no suffix means bytes. opts: `memory` (required). Returns
/// `{value, suffix, bytes, mib}`; errors on an unknown suffix or u64 overflow.
/// Pure.
fn op_parse_memory(opts: Value) -> Result<Value> {
    let raw = opts
        .get("memory")
        .or_else(|| opts.get("size"))
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("missing memory"))?;
    let s = raw.trim();
    let split = s.find(|c: char| !c.is_ascii_digit()).unwrap_or(s.len());
    let (num, suffix) = s.split_at(split);
    if num.is_empty() {
        return Err(anyhow!("Spark size `{s}` has no numeric value"));
    }
    let value: u64 = num.parse()?;
    let suffix = suffix.to_ascii_lowercase();
    let mult: u64 = match suffix.as_str() {
        "" | "b" => 1,
        "k" | "kb" | "ki" | "kib" => 1024,
        "m" | "mb" | "mi" | "mib" => 1024u64.pow(2),
        "g" | "gb" | "gi" | "gib" => 1024u64.pow(3),
        "t" | "tb" | "ti" | "tib" => 1024u64.pow(4),
        "p" | "pb" | "pi" | "pib" => 1024u64.pow(5),
        other => return Err(anyhow!("unknown Spark size suffix `{other}`")),
    };
    let bytes = value
        .checked_mul(mult)
        .ok_or_else(|| anyhow!("size overflows u64 bytes: `{s}`"))?;
    Ok(json!({
        "value": value,
        "suffix": suffix,
        "bytes": bytes,
        "mib": bytes as f64 / (1024.0 * 1024.0),
    }))
}

/// Encode a byte count as a Spark size string — the inverse of `parse_memory`.
/// Picks the largest binary unit (P→T→G→M→K→B) that divides the byte count
/// evenly, so `parse_memory` round-trips it exactly; `0` gives `"0b"`. Emits the
/// short lowercase suffix Spark config accepts (e.g. `spark.executor.memory`).
/// opts: `bytes` (required). Returns `{value, suffix, string, bytes}`. Pure.
fn op_build_memory(opts: Value) -> Result<Value> {
    let bytes = opts
        .get("bytes")
        .and_then(Value::as_u64)
        .ok_or_else(|| anyhow!("missing bytes"))?;
    if bytes == 0 {
        return Ok(json!({"value": 0, "suffix": "b", "string": "0b", "bytes": 0}));
    }
    // Largest unit first; the first that divides evenly wins.
    let units: [(u64, &str); 6] = [
        (1024u64.pow(5), "p"),
        (1024u64.pow(4), "t"),
        (1024u64.pow(3), "g"),
        (1024u64.pow(2), "m"),
        (1024, "k"),
        (1, "b"),
    ];
    for (mult, suffix) in units {
        if bytes.is_multiple_of(mult) {
            let value = bytes / mult;
            return Ok(json!({
                "value": value,
                "suffix": suffix,
                "string": format!("{value}{suffix}"),
                "bytes": bytes,
            }));
        }
    }
    unreachable!("the 1-byte unit divides every count")
}

/// Quote a Spark SQL identifier with backticks, doubling any embedded backtick.
fn op_quote_ident(opts: Value) -> Result<Value> {
    let name = opts
        .get("name")
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("missing name"))?;
    Ok(json!({"quoted": quote_ident_str(name)}))
}

/// Decode a backtick-quoted Spark SQL identifier back to its raw name — the
/// inverse of `quote_ident`. The input must be wrapped in matching backticks
/// with every embedded backtick doubled (`` `` `` → `` ` ``); an unpaired
/// backtick is rejected. opts: `quoted` (required). Returns `{name}`. Pure.
fn op_unquote_ident(opts: Value) -> Result<Value> {
    let input = opts
        .get("quoted")
        .or_else(|| opts.get("ident"))
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("missing quoted"))?;
    let inner = input
        .strip_prefix('`')
        .and_then(|s| s.strip_suffix('`'))
        .filter(|_| input.len() >= 2)
        .ok_or_else(|| anyhow!("not a backtick-quoted identifier: {input}"))?;
    // Every embedded backtick must be doubled — an odd count means a stray one.
    if inner.matches('`').count() % 2 != 0 {
        return Err(anyhow!(
            "malformed identifier: unpaired backtick in {input}"
        ));
    }
    Ok(json!({ "name": inner.replace("``", "`") }))
}

/// Quote a qualified Spark table identifier `[catalog.][database.]table`,
/// backtick-quoting each part and rejoining with `.`. Splitting honors backtick
/// quoting (a `.` inside backticks stays in one part), so it round-trips with
/// `parse_table_name`. Caps at three parts (catalog.database.table). Pure.
fn op_quote_qualified_ident(opts: Value) -> Result<Value> {
    let name = opts
        .get("name")
        .or_else(|| opts.get("table"))
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("missing name"))?;
    let parts = split_qualified(name);
    if parts.iter().any(|p| p.is_empty()) {
        return Err(anyhow!("qualified identifier has an empty part: {name}"));
    }
    if parts.len() > 3 {
        return Err(anyhow!(
            "too many parts (max catalog.database.table): {name}"
        ));
    }
    let quoted = parts
        .iter()
        .map(|p| quote_ident_str(p))
        .collect::<Vec<_>>()
        .join(".");
    Ok(json!({"quoted": quoted, "parts": parts}))
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
pub extern "C" fn spark__views(args: *const c_char) -> *const c_char {
    ffi_call(args, op_views)
}

#[no_mangle]
pub extern "C" fn spark__catalogs(args: *const c_char) -> *const c_char {
    ffi_call(args, op_catalogs)
}

#[no_mangle]
pub extern "C" fn spark__current_database(args: *const c_char) -> *const c_char {
    ffi_call(args, op_current_database)
}

#[no_mangle]
pub extern "C" fn spark__create_temp_view(args: *const c_char) -> *const c_char {
    ffi_call(args, op_create_temp_view)
}

#[no_mangle]
pub extern "C" fn spark__drop_temp_view(args: *const c_char) -> *const c_char {
    ffi_call(args, op_drop_temp_view)
}

#[no_mangle]
pub extern "C" fn spark__set_database(args: *const c_char) -> *const c_char {
    ffi_call(args, op_set_database)
}

#[no_mangle]
pub extern "C" fn spark__refresh_table(args: *const c_char) -> *const c_char {
    ffi_call(args, op_refresh_table)
}

#[no_mangle]
pub extern "C" fn spark__schema(args: *const c_char) -> *const c_char {
    ffi_call(args, op_schema)
}

#[no_mangle]
pub extern "C" fn spark__submit(args: *const c_char) -> *const c_char {
    ffi_call(args, op_submit)
}

#[no_mangle]
pub extern "C" fn spark__explain(args: *const c_char) -> *const c_char {
    ffi_call(args, op_explain)
}

#[no_mangle]
pub extern "C" fn spark__read(args: *const c_char) -> *const c_char {
    ffi_call(args, op_read)
}

#[no_mangle]
pub extern "C" fn spark__write(args: *const c_char) -> *const c_char {
    ffi_call(args, op_write)
}

#[no_mangle]
pub extern "C" fn spark__columns(args: *const c_char) -> *const c_char {
    ffi_call(args, op_columns)
}

#[no_mangle]
pub extern "C" fn spark__functions(args: *const c_char) -> *const c_char {
    ffi_call(args, op_functions)
}

#[no_mangle]
pub extern "C" fn spark__cache(args: *const c_char) -> *const c_char {
    ffi_call(args, |o| op_cache_like(o, "cache"))
}

#[no_mangle]
pub extern "C" fn spark__uncache(args: *const c_char) -> *const c_char {
    ffi_call(args, |o| op_cache_like(o, "uncache"))
}

#[no_mangle]
pub extern "C" fn spark__config(args: *const c_char) -> *const c_char {
    ffi_call(args, op_config)
}

#[no_mangle]
pub extern "C" fn spark__parse_master_url(args: *const c_char) -> *const c_char {
    ffi_call(args, op_parse_master_url)
}

#[no_mangle]
pub extern "C" fn spark__build_master_url(args: *const c_char) -> *const c_char {
    ffi_call(args, op_build_master_url)
}

#[no_mangle]
pub extern "C" fn spark__parse_table_name(args: *const c_char) -> *const c_char {
    ffi_call(args, op_parse_table_name)
}

#[no_mangle]
pub extern "C" fn spark__build_table_name(args: *const c_char) -> *const c_char {
    ffi_call(args, op_build_table_name)
}

#[no_mangle]
pub extern "C" fn spark__parse_memory(args: *const c_char) -> *const c_char {
    ffi_call(args, op_parse_memory)
}

#[no_mangle]
pub extern "C" fn spark__build_memory(args: *const c_char) -> *const c_char {
    ffi_call(args, op_build_memory)
}

#[no_mangle]
pub extern "C" fn spark__quote_ident(args: *const c_char) -> *const c_char {
    ffi_call(args, op_quote_ident)
}

#[no_mangle]
pub extern "C" fn spark__unquote_ident(args: *const c_char) -> *const c_char {
    ffi_call(args, op_unquote_ident)
}

#[no_mangle]
pub extern "C" fn spark__quote_qualified_ident(args: *const c_char) -> *const c_char {
    ffi_call(args, op_quote_qualified_ident)
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

    /// Restore PATH around a closure. The crate-wide `with_env` helper
    /// saves/restores STRYKE_SPARK_SUBMIT and SPARK_HOME but NOT PATH, and
    /// the locate-precedence tests below must clobber PATH to control the
    /// `$PATH` fallback scan. Without this, a clobbered PATH would leak into
    /// every later test in the same binary. ENV_LOCK (held by `with_env`)
    /// already serializes these, so saving/restoring inside is race-free.
    fn with_path<F: FnOnce()>(p: &str, f: F) {
        let saved = std::env::var("PATH").ok();
        std::env::set_var("PATH", p);
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));
        match saved {
            Some(s) => std::env::set_var("PATH", s),
            None => std::env::remove_var("PATH"),
        }
        if let Err(panic) = r {
            std::panic::resume_unwind(panic);
        }
    }

    /// Catches: a refactor that turns the explicit-`spark_submit`
    /// not-a-file branch (`return Err(...)` at the end of the
    /// `if let Some(p) = &self.spark_submit` arm) into a fall-through to
    /// the `$PATH` scan. That regression is a binary-substitution bug: a
    /// user who typos an explicit `spark_submit` path would SILENTLY run a
    /// different `spark-submit` found on `$PATH` instead of getting the
    /// "is not a file" error. The contract is "explicit path is
    /// authoritative — if it's wrong, fail loudly; never substitute."
    ///
    /// We prove non-substitution by planting a REAL, valid `spark-submit`
    /// on `$PATH` and confirming locate still errors on the bad explicit
    /// path rather than resolving the PATH one.
    #[test]
    fn locate_explicit_bad_does_not_fall_through_to_valid_path() {
        with_env(|| {
            let dir = tempfile::tempdir().unwrap();
            let on_path = dir.path().join("spark-submit");
            std::fs::write(&on_path, b"#!/bin/sh\n").unwrap();
            assert!(on_path.is_file(), "fixture spark-submit must exist");

            with_path(dir.path().to_str().unwrap(), || {
                let o = SparkOpts::from_value(&json!({
                    "spark_submit": "/definitely/not/here/spark-submit",
                }));
                let err = o.locate_spark_submit().unwrap_err().to_string();
                assert!(
                    err.contains("is not a file"),
                    "explicit bad path must error, not silently use $PATH: {err}",
                );
                // The PATH one must NOT be what got returned.
                assert!(
                    !err.contains(dir.path().to_str().unwrap()),
                    "error must reference the explicit path, not the PATH dir: {err}",
                );
            });
        });
    }

    /// Catches: a regression in the `spark_home` → `bin/spark-submit` join
    /// (line `PathBuf::from(home).join("bin").join("spark-submit")`). If a
    /// refactor drops the `"bin"` segment, joins in the wrong order, or
    /// hardcodes a separator, this resolution silently breaks and every
    /// SPARK_HOME-only user falls back to `$PATH` (or errors). Pins the
    /// exact resolved path AND that spark_home takes precedence over the
    /// `$PATH` scan (we leave PATH empty so only the home branch can win).
    #[test]
    fn locate_spark_home_resolves_bin_spark_submit() {
        with_env(|| {
            let home = tempfile::tempdir().unwrap();
            let bin = home.path().join("bin");
            std::fs::create_dir(&bin).unwrap();
            let expected = bin.join("spark-submit");
            std::fs::write(&expected, b"#!/bin/sh\n").unwrap();

            with_path("/no/spark/here", || {
                let o =
                    SparkOpts::from_value(&json!({"spark_home": home.path().to_str().unwrap()}));
                let got = o
                    .locate_spark_submit()
                    .expect("spark_home bin must resolve");
                assert_eq!(
                    got, expected,
                    "must resolve $SPARK_HOME/bin/spark-submit exactly",
                );
            });
        });
    }

    /// Catches: a refactor that makes a *present-but-incomplete*
    /// `spark_home` (one without `bin/spark-submit`) a hard error instead
    /// of a soft preference. The current contract is: spark_home is tried
    /// first, and if `bin/spark-submit` is absent the code FALLS THROUGH to
    /// the `$PATH` scan (the `if path.is_file()` guard does not early-error
    /// in the home arm — only the explicit `spark_submit` arm does). A
    /// refactor that adds an `else { return Err }` to the home arm would
    /// break users who set SPARK_HOME for other Spark tooling but rely on a
    /// PATH-installed spark-submit. We prove fall-through by planting the
    /// real binary ONLY on PATH and leaving spark_home's bin empty.
    #[test]
    fn locate_spark_home_without_bin_falls_through_to_path() {
        with_env(|| {
            let home = tempfile::tempdir().unwrap(); // no bin/ created
            let path_dir = tempfile::tempdir().unwrap();
            let on_path = path_dir.path().join("spark-submit");
            std::fs::write(&on_path, b"#!/bin/sh\n").unwrap();

            with_path(path_dir.path().to_str().unwrap(), || {
                let o =
                    SparkOpts::from_value(&json!({"spark_home": home.path().to_str().unwrap()}));
                let got = o
                    .locate_spark_submit()
                    .expect("incomplete spark_home must fall through to PATH, not error");
                assert_eq!(
                    got, on_path,
                    "must resolve the PATH spark-submit when spark_home lacks bin/",
                );
            });
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

    // ── apply_to ──
    //
    // These tests pin the *exact* CLI surface assembled into spark-submit.
    // Refactors that move the `cmd.arg("--conf").arg(c)` line inside an
    // if-branch, swap `starts_with` for substring `contains`, or drop the
    // default-injection logic will get caught here.

    fn args_of(cmd: &std::process::Command) -> Vec<String> {
        cmd.get_args()
            .map(|a| a.to_string_lossy().into_owned())
            .collect()
    }

    /// Catches: regressions where a user-supplied `spark.log.level=…` conf
    /// fails to suppress the hard-coded `spark.log.level=ERROR` default,
    /// resulting in two `--conf` entries for the same key. Spark's behavior
    /// for duplicate keys is implementation-defined; the contract here is
    /// "user wins, default is silent." Also pins that user confs are
    /// ALWAYS appended (not gated behind the suppression check) — a
    /// tempting refactor that puts the `cmd.arg` inside the `if` would
    /// silently drop every user conf.
    #[test]
    fn apply_to_user_log_level_suppresses_default_and_keeps_user_conf() {
        with_env(|| {
            let o = SparkOpts::from_value(&json!({
                "confs": ["spark.log.level=DEBUG"],
            }));
            let mut cmd = Command::new("dummy");
            o.apply_to(&mut cmd);
            let args = args_of(&cmd);

            // user conf must be present, exactly once
            let user_pos = args
                .iter()
                .position(|a| a == "spark.log.level=DEBUG")
                .expect("user log.level conf must survive apply_to");
            assert_eq!(args[user_pos - 1], "--conf", "user conf needs --conf flag");

            // default ERROR must NOT be present
            assert!(
                !args.iter().any(|a| a == "spark.log.level=ERROR"),
                "default log.level=ERROR must be suppressed when user supplies one: {:?}",
                args
            );

            // exactly one log.level conf overall
            let n = args
                .iter()
                .filter(|a| a.starts_with("spark.log.level="))
                .count();
            assert_eq!(
                n, 1,
                "exactly one log.level conf expected, got {n}: {:?}",
                args
            );
        });
    }

    /// Catches: regression where the default master fallback (`local[*]`)
    /// is removed or accidentally gated behind a user-conf check. Also
    /// pins that an empty `confs` array does NOT suppress the two
    /// hard-coded defaults (`spark.log.level=ERROR`,
    /// `spark.sql.catalogImplementation=in-memory`) — both must appear
    /// because the suppression check is keyed on prefix-match, not on
    /// "any user confs at all."
    #[test]
    fn apply_to_no_user_confs_emits_both_defaults_and_local_master() {
        with_env(|| {
            let o = SparkOpts::from_value(&json!({}));
            let mut cmd = Command::new("dummy");
            o.apply_to(&mut cmd);
            let args = args_of(&cmd);

            assert!(
                args.iter().any(|a| a == "spark.log.level=ERROR"),
                "default log.level=ERROR missing: {:?}",
                args
            );
            assert!(
                args.iter()
                    .any(|a| a == "spark.sql.catalogImplementation=in-memory"),
                "default catalogImplementation=in-memory missing: {:?}",
                args
            );

            // master must default to local[*] — adjacent --master flag.
            let mpos = args
                .iter()
                .position(|a| a == "--master")
                .expect("--master flag missing");
            assert_eq!(
                args.get(mpos + 1).map(String::as_str),
                Some("local[*]"),
                "default master must be local[*]: {:?}",
                args
            );

            // app name defaults to "stryke-spark" — adjacent --name flag.
            let npos = args
                .iter()
                .position(|a| a == "--name")
                .expect("--name flag missing");
            assert_eq!(
                args.get(npos + 1).map(String::as_str),
                Some("stryke-spark"),
                "default app name must be stryke-spark: {:?}",
                args
            );
        });
    }

    /// Catches: case-sensitivity bug in the catalogImplementation
    /// suppression check. The code uses `starts_with` on the canonical
    /// camelCase `spark.sql.catalogImplementation=` but Spark's own
    /// config keys are case-insensitive on the JVM side. If a user passes
    /// the conf with different casing (lowercased), the suppression
    /// fails and BOTH the user's value AND the hard-coded `in-memory`
    /// default end up on the command line as duplicate `--conf` args.
    /// Spark's behavior is then "last-wins" — so the default silently
    /// overrides the user's intent. Documents the current behavior as
    /// a known limitation; if the impl is fixed to be case-insensitive,
    /// this test will flip and need updating.
    #[test]
    fn apply_to_catalog_impl_suppression_is_case_sensitive() {
        with_env(|| {
            // Lowercased key — does NOT match the starts_with prefix.
            let o = SparkOpts::from_value(&json!({
                "confs": ["spark.sql.catalogimplementation=hive"],
            }));
            let mut cmd = Command::new("dummy");
            o.apply_to(&mut cmd);
            let args = args_of(&cmd);

            // Both end up in the command — this is the case-sensitivity bug.
            assert!(
                args.iter()
                    .any(|a| a == "spark.sql.catalogimplementation=hive"),
                "user (lowercased) conf must still appear: {:?}",
                args
            );
            assert!(
                args.iter()
                    .any(|a| a == "spark.sql.catalogImplementation=in-memory"),
                "default conf survives because suppression check is case-sensitive: {:?}",
                args
            );
        });
    }

    /// Catches: regressions that drop or reorder the optional flag arms
    /// (`if let Some(m) = master`, `if let Some(d) = deploy_mode`,
    /// `if let Some(p) = packages`, `if let Some(j) = jars`). Today
    /// NO existing test fires the Some(_) branches for deploy_mode,
    /// packages, or jars — refactoring any of those arms to a no-op
    /// would compile cleanly and break user behavior silently.
    /// Also pins the contract that each flag immediately precedes
    /// its value (spark-submit positional contract: `--key value`).
    #[test]
    fn apply_to_all_optional_flags_emit_with_adjacent_values() {
        with_env(|| {
            let o = SparkOpts::from_value(&json!({
                "master": "spark://m:7077",
                "app_name": "custom-app",
                "deploy_mode": "cluster",
                "packages": "com.example:lib:1.0",
                "jars": "/a.jar,/b.jar",
            }));
            let mut cmd = Command::new("dummy");
            o.apply_to(&mut cmd);
            let args = args_of(&cmd);

            // Helper: assert flag is present AND its value follows immediately.
            let assert_flag_value = |flag: &str, val: &str| {
                let pos = args
                    .iter()
                    .position(|a| a == flag)
                    .unwrap_or_else(|| panic!("{flag} missing: {args:?}"));
                assert_eq!(
                    args.get(pos + 1).map(String::as_str),
                    Some(val),
                    "{flag} value mismatch: {args:?}",
                );
            };

            assert_flag_value("--master", "spark://m:7077");
            assert_flag_value("--name", "custom-app");
            assert_flag_value("--deploy-mode", "cluster");
            assert_flag_value("--packages", "com.example:lib:1.0");
            assert_flag_value("--jars", "/a.jar,/b.jar");

            // Default master local[*] MUST NOT appear when user provided one.
            assert!(
                !args.iter().any(|a| a == "local[*]"),
                "user master must suppress local[*] default: {args:?}",
            );
        });
    }

    /// Catches: refactors that swap `Vec<String>` for `HashSet<String>`
    /// or otherwise lose stable iteration order on user confs. Spark
    /// honors last-wins for duplicate keys, so the iteration order of
    /// `confs` is a load-bearing part of the user contract — a
    /// HashMap/HashSet refactor would silently make the "last conf wins"
    /// behavior non-deterministic across runs.
    #[test]
    fn apply_to_preserves_user_conf_insertion_order() {
        with_env(|| {
            // Three confs chosen so a hash-based reordering is overwhelmingly
            // likely to scramble them (different prefixes, different lengths).
            let o = SparkOpts::from_value(&json!({
                "confs": [
                    "spark.executor.memory=4g",
                    "spark.cores.max=8",
                    "spark.dynamicAllocation.enabled=false",
                ],
            }));
            let mut cmd = Command::new("dummy");
            o.apply_to(&mut cmd);
            let args = args_of(&cmd);

            let p1 = args
                .iter()
                .position(|a| a == "spark.executor.memory=4g")
                .expect("first user conf missing");
            let p2 = args
                .iter()
                .position(|a| a == "spark.cores.max=8")
                .expect("second user conf missing");
            let p3 = args
                .iter()
                .position(|a| a == "spark.dynamicAllocation.enabled=false")
                .expect("third user conf missing");

            assert!(
                p1 < p2 && p2 < p3,
                "user conf order must be preserved (got positions {p1}, {p2}, {p3}): {args:?}",
            );
        });
    }

    /// `coerce_submit_args` must NOT silently drop non-string entries.
    /// Pre-fix the broken `filter_map(as_str)` turned
    /// `["--num", 42, "--flag"]` into `["--num", "--flag"]` — the value
    /// `42` vanished and spark-submit got `--num --flag` (a confusing
    /// nested-flag error). Now numbers and bools coerce to their string
    /// form so the user's intent survives.
    #[test]
    fn coerce_submit_args_preserves_number_and_bool_values() {
        let arr = vec![json!("--num"), json!(42), json!("--flag"), json!(true)];
        let out = coerce_submit_args(&arr).expect("scalar args must coerce");
        assert_eq!(out, vec!["--num", "42", "--flag", "true"]);
    }

    /// Arrays/objects in args round-trip via JSON encoding (rare but
    /// supported — some spark configs accept JSON-encoded values).
    #[test]
    fn coerce_submit_args_encodes_compound_values_as_json() {
        let arr = vec![json!("--config"), json!({"k": 1})];
        let out = coerce_submit_args(&arr).expect("compound args coerce to JSON");
        assert_eq!(out, vec!["--config", "{\"k\":1}"]);
    }

    /// `null` is a hard error — `spark-submit` can't accept it
    /// meaningfully and pre-fix it would have been silently dropped.
    /// The user gets a clear error naming the offending index.
    #[test]
    fn coerce_submit_args_rejects_null_with_indexed_error() {
        let arr = vec![json!("--x"), Value::Null, json!("--y")];
        let err = coerce_submit_args(&arr).expect_err("null must hard-fail");
        let msg = err.to_string();
        assert!(
            msg.contains("args[1]"),
            "error must name the index, got: {msg}"
        );
        assert!(msg.contains("null"), "error must mention null, got: {msg}");
    }

    // ── FFI contract ──
    //
    // The cdylib is dlopen'd by stryke; every `#[no_mangle] extern "C"`
    // export crosses an unwind-unsafe boundary. The invariants below are
    // load-bearing for stryke process stability — a panic that unwinds
    // past the FFI edge is UB, and an alloc/free mismatch in
    // `stryke_free_cstring` corrupts the heap of whatever loaded the
    // cdylib. These are the highest-blast-radius bugs in the crate.

    /// Round-trip an FFI response pointer through `CStr` for assertions.
    /// SAFETY: `p` must be a non-null pointer returned by an export from
    /// this cdylib (we only call this on values we just allocated).
    unsafe fn ffi_to_json(p: *const c_char) -> Value {
        assert!(!p.is_null(), "FFI export returned null pointer");
        let s = CStr::from_ptr(p)
            .to_str()
            .expect("FFI output must be UTF-8");
        let v: Value = serde_json::from_str(s).expect("FFI output must be valid JSON");
        // Caller is responsible for freeing — round-trip back through
        // stryke_free_cstring on the same pointer to validate the free
        // contract without leaking.
        stryke_free_cstring(p as *mut c_char);
        v
    }

    /// Catches: a panic inside any handler unwinding across the C ABI.
    /// `ffi_call` MUST catch panics via `catch_unwind` and convert them
    /// to a JSON error envelope — otherwise the panic escapes the FFI
    /// boundary into the dlopen'ing process, which is undefined behavior
    /// on stable Rust (the stryke host would abort or corrupt). A future
    /// refactor that drops the `AssertUnwindSafe` wrapper, replaces
    /// `catch_unwind` with a bare call, or "simplifies" the error path
    /// to a `?` would silently introduce process-level UB.
    ///
    /// The handler used here panics with a unique sentinel so we know
    /// the panic actually fired (vs the handler returning Ok early and
    /// faking the test). The returned JSON must contain the documented
    /// "stryke-spark handler panicked" string — not the panic payload,
    /// which would leak internal panic messages to stryke users.
    #[test]
    fn ffi_call_catches_handler_panic_without_unwinding() {
        let input = CString::new(r#"{"x":1}"#).unwrap();
        let p = ffi_call(input.as_ptr(), |_v| -> Result<Value> {
            panic!("INTERNAL_SENTINEL_PANIC_xyz_42");
        });
        // SAFETY: ffi_call always returns a non-null pointer or null on
        // CString::new failure; we never trigger the latter here.
        let v = unsafe { ffi_to_json(p) };
        assert_eq!(
            v["error"], json!("stryke-spark handler panicked"),
            "panic must be reported as the documented error string, not the panic payload (got {v})"
        );
        // The raw panic payload must NOT leak — stryke users would see
        // internal panic messages and that's a contract violation.
        let s = v.to_string();
        assert!(
            !s.contains("INTERNAL_SENTINEL_PANIC_xyz_42"),
            "panic payload leaked to FFI output: {s}",
        );
    }

    /// Catches: a refactor that hardcodes a version literal in
    /// `spark__pkg_version` instead of using `env!("CARGO_PKG_VERSION")`.
    /// stryke's loader checks the cdylib version against its expected
    /// version at first `use Spark` to detect ABI skew — a hardcoded
    /// version literal would silently report the wrong version after a
    /// `cargo bp` bumps `Cargo.toml`, and stryke would happily call into
    /// an ABI-incompatible cdylib. Pins the env! → output invariant.
    #[test]
    fn ffi_pkg_version_matches_cargo_pkg_version() {
        let p = spark__pkg_version(std::ptr::null());
        let v = unsafe { ffi_to_json(p) };
        assert_eq!(
            v["version"],
            json!(env!("CARGO_PKG_VERSION")),
            "spark__pkg_version must echo CARGO_PKG_VERSION (got {v})",
        );
        // Defensive: env! at compile time should be a non-empty version
        // string. If the build system ever produced an empty version,
        // stryke's loader check would compare "" == "" and pass — a
        // false-negative on ABI skew detection.
        assert!(
            !env!("CARGO_PKG_VERSION").is_empty(),
            "CARGO_PKG_VERSION must not be empty at build time",
        );
    }

    /// Catches: a refactor of `stryke_free_cstring` that uses
    /// `Box::from_raw` instead of `CString::from_raw`, or that frees
    /// pointers it didn't allocate (the C ABI symmetry that backs every
    /// dlopen'd cdylib). The pairing under test:
    ///   - `ffi_call` allocates via `CString::new(...).into_raw()`
    ///   - `stryke_free_cstring` deallocates via `CString::from_raw(p)`
    ///
    /// These MUST use the same allocator and the same layout — a refactor
    /// that swaps one to `Box`/`Vec` would corrupt the host heap in
    /// release builds where the mismatch goes silently undetected.
    ///
    /// We exercise the full round-trip: call an FFI export, hand the
    /// returned pointer back to `stryke_free_cstring`, repeat many times.
    /// If the alloc/free pair drifts apart this test will SEGFAULT or
    /// abort under ASAN/Miri rather than pass silently.
    #[test]
    fn ffi_free_cstring_round_trips_with_into_raw() {
        // Repeated alloc+free under realistic-shaped payloads. If the
        // allocator pairing is broken, heap corruption surfaces within
        // a few iterations — pre-fix Box/CString swap would crash here.
        for _ in 0..256 {
            let p = spark__pkg_version(std::ptr::null());
            assert!(!p.is_null(), "spark__pkg_version must not return null");
            // SAFETY: pointer was just returned by spark__pkg_version,
            // which allocates via CString::into_raw — the documented
            // pairing with stryke_free_cstring.
            unsafe { stryke_free_cstring(p as *mut c_char) };
        }
        // Null-free is documented as a no-op; verify it doesn't crash.
        // A future refactor that drops the early-return null check would
        // dereference null in CString::from_raw and abort the process.
        unsafe { stryke_free_cstring(std::ptr::null_mut()) };
    }

    /// Catches: a refactor that propagates JSON-parse errors out of
    /// `ffi_call` instead of silently coercing malformed input to
    /// `Value::Null`. Today, garbage input bytes → `Value::Null` →
    /// handler runs against Null and returns whatever it returns for
    /// missing fields. This is the documented contract that stryke's
    /// dlopen bridge relies on — it never has to pre-validate JSON
    /// before passing the C string in. A refactor to `?` the parse
    /// error would crash the test that follows because stryke's bridge
    /// passes through arbitrary bytes from user code.
    ///
    /// Counter-fact: this DOES mean genuine JSON errors are invisible
    /// to the user. If the bug-bias is the other direction, the test
    /// flips and we surface the parse error to the JSON envelope. For
    /// now we pin the current contract because that's what the cdylib
    /// FFI bridge in stryke is built against.
    #[test]
    fn ffi_call_malformed_json_input_silently_becomes_null() {
        // Garbage that is not valid JSON.
        let garbage = CString::new("not json {{{").unwrap();
        let mut captured: Option<Value> = None;
        let p = ffi_call(garbage.as_ptr(), |v| {
            captured = Some(v.clone());
            Ok(json!({"ok": true}))
        });
        let _ = unsafe { ffi_to_json(p) };
        assert_eq!(
            captured,
            Some(Value::Null),
            "malformed JSON input must silently become Value::Null (current contract)",
        );
    }

    /// Every new export must validate its required arg BEFORE locating /
    /// spawning spark-submit — a typo should surface instantly, not after a
    /// multi-second JVM warmup. Each call passes args missing the required
    /// key; the documented error must come back fast and must never mention
    /// spark-submit (which would mean it tried to spawn).
    #[test]
    fn new_ops_validate_before_spark_submit() {
        with_env(|| {
            let cases: &[(extern "C" fn(*const c_char) -> *const c_char, &str, &str)] = &[
                (spark__explain, r#"{}"#, "missing sql"),
                (spark__read, r#"{}"#, "missing path"),
                (spark__write, r#"{}"#, "missing sql"),
                (
                    spark__write,
                    r#"{"sql":"select 1"}"#,
                    "needs a `path` or a `table`",
                ),
                (spark__columns, r#"{}"#, "missing table"),
                (spark__cache, r#"{}"#, "missing table"),
                (spark__uncache, r#"{}"#, "missing table"),
                (spark__config, r#"{}"#, "missing key"),
            ];
            for (f, arg, want) in cases {
                let cs = CString::new(*arg).unwrap();
                let start = std::time::Instant::now();
                let v = unsafe { ffi_to_json(f(cs.as_ptr())) };
                assert!(
                    start.elapsed() < std::time::Duration::from_secs(2),
                    "validation for {want} took too long — spawned spark-submit?"
                );
                let err = v.get("error").and_then(|e| e.as_str()).unwrap_or("");
                assert!(
                    err.contains(want),
                    "expected `{want}` for {arg}; got: {err}"
                );
                assert!(
                    !err.contains("spark-submit"),
                    "validation must fire before spark-submit lookup; got: {err}"
                );
            }
        });
    }

    // ── pure helpers (no Spark) ──────────────────────────────────────────────

    #[test]
    fn parse_master_url_local_forms() {
        assert_eq!(
            op_parse_master_url(json!({"url": "local"})).unwrap(),
            json!({"scheme": "local", "threads": 1})
        );
        assert_eq!(
            op_parse_master_url(json!({"url": "local[8]"})).unwrap()["threads"],
            json!(8)
        );
        assert_eq!(
            op_parse_master_url(json!({"url": "local[*]"})).unwrap()["threads"],
            json!("*")
        );
        assert!(op_parse_master_url(json!({"url": "local[x]"})).is_err());
    }

    #[test]
    fn parse_master_url_standalone_and_scheme_forms() {
        let ha = op_parse_master_url(json!({"url": "spark://m1:7077,m2:7077"})).unwrap();
        let hosts = ha["hosts"].as_array().unwrap();
        assert_eq!(hosts.len(), 2, "standalone HA host list");
        assert_eq!(hosts[0]["host"], json!("m1"));
        assert_eq!(hosts[0]["port"], json!(7077));
        assert_eq!(
            op_parse_master_url(json!({"url": "yarn"})).unwrap()["scheme"],
            json!("yarn")
        );
        let k8s = op_parse_master_url(json!({"url": "k8s://https://api:6443"})).unwrap();
        assert_eq!(k8s["scheme"], json!("k8s"));
        assert_eq!(
            k8s["master"],
            json!("https://api:6443"),
            "k8s keeps the full inner URL"
        );
    }

    #[test]
    fn build_master_url_inverts_parse_master_url() {
        // local thread forms: 1/absent → bare, N>1 → local[N], "*" → local[*].
        assert_eq!(
            op_build_master_url(json!({"scheme": "local"})).unwrap()["url"],
            json!("local")
        );
        assert_eq!(
            op_build_master_url(json!({"scheme": "local", "threads": 1})).unwrap()["url"],
            json!("local")
        );
        assert_eq!(
            op_build_master_url(json!({"scheme": "local", "threads": 8})).unwrap()["url"],
            json!("local[8]")
        );
        assert_eq!(
            op_build_master_url(json!({"scheme": "local", "threads": "*"})).unwrap()["url"],
            json!("local[*]")
        );
        // Round-trips the canonical forms through parse_master_url.
        for url in [
            "local",
            "local[8]",
            "local[*]",
            "yarn",
            "spark://m1:7077",
            "spark://m1:7077,m2:7077",
            "mesos://host:5050",
            "k8s://https://api:6443",
        ] {
            let parsed = op_parse_master_url(json!({ "url": url })).unwrap();
            let rebuilt = op_build_master_url(parsed).unwrap()["url"].clone();
            assert_eq!(rebuilt, json!(url), "round-trip for {url}");
        }
        // Error cases: empty spark hosts, missing master, missing scheme.
        assert!(op_build_master_url(json!({"scheme": "spark", "hosts": []})).is_err());
        assert!(op_build_master_url(json!({"scheme": "mesos"})).is_err());
        assert!(op_build_master_url(json!({})).is_err());
    }

    #[test]
    fn parse_memory_uses_binary_suffixes() {
        // 512m = 512 * 1024^2 bytes; the suffix is binary, not decimal.
        let m = op_parse_memory(json!({"memory": "512m"})).unwrap();
        assert_eq!(m["value"], json!(512));
        assert_eq!(m["bytes"], json!(536_870_912u64));
        assert_eq!(m["mib"], json!(512.0));
        // Spark treats `kb` as KiB (1024), NOT 1000.
        assert_eq!(
            op_parse_memory(json!({"memory": "1kb"})).unwrap()["bytes"],
            json!(1024),
            "kb is 1024 bytes in Spark"
        );
        // Case-insensitive, and `gib`/`g`/`gb` all agree.
        for s in ["2g", "2G", "2gb", "2GiB"] {
            assert_eq!(
                op_parse_memory(json!({ "memory": s })).unwrap()["bytes"],
                json!(2_147_483_648u64),
                "{s} = 2 GiB"
            );
        }
        // No suffix is bytes.
        assert_eq!(
            op_parse_memory(json!({"memory": "4096"})).unwrap()["bytes"],
            json!(4096)
        );
        // Every magnitude.
        assert_eq!(
            op_parse_memory(json!({"memory": "1t"})).unwrap()["bytes"],
            json!(1_099_511_627_776u64)
        );
        // No numeric value and an unknown suffix reject.
        assert!(op_parse_memory(json!({"memory": "g"})).is_err());
        assert!(op_parse_memory(json!({"memory": "10x"})).is_err());
        assert!(op_parse_memory(json!({})).is_err());
    }

    #[test]
    fn build_memory_picks_largest_dividing_unit_and_round_trips() {
        // 0 → "0b".
        assert_eq!(
            op_build_memory(json!({"bytes": 0})).unwrap()["string"],
            json!("0b")
        );
        // Even GiB collapses to the gigabyte unit.
        let g = op_build_memory(json!({"bytes": 2_147_483_648u64})).unwrap();
        assert_eq!(g["string"], json!("2g"));
        assert_eq!(g["suffix"], json!("g"));
        // 1.5 GiB isn't a whole number of GiB, so it drops to MiB (1536m).
        assert_eq!(
            op_build_memory(json!({"bytes": 1_610_612_736u64})).unwrap()["string"],
            json!("1536m")
        );
        // A non-power-of-1024 count falls through to plain bytes.
        assert_eq!(
            op_build_memory(json!({"bytes": 1234})).unwrap()["string"],
            json!("1234b")
        );
        // Largest unit wins: exactly 1 TiB is "1t", not "1024g".
        assert_eq!(
            op_build_memory(json!({"bytes": 1_099_511_627_776u64})).unwrap()["string"],
            json!("1t")
        );
        // Round-trips through parse_memory for every produced string.
        for bytes in [1234u64, 1024, 536_870_912, 2_147_483_648, 1_610_612_736] {
            let s = op_build_memory(json!({ "bytes": bytes })).unwrap()["string"]
                .as_str()
                .unwrap()
                .to_string();
            assert_eq!(
                op_parse_memory(json!({ "memory": s })).unwrap()["bytes"],
                json!(bytes),
                "round-trip {bytes}"
            );
        }
        assert!(op_build_memory(json!({})).is_err());
    }

    #[test]
    fn parse_table_name_one_two_three_parts() {
        let t = op_parse_table_name(json!({"name": "events"})).unwrap();
        assert_eq!(t["table"], json!("events"));
        assert_eq!(t["database"], Value::Null);
        let dt = op_parse_table_name(json!({"name": "analytics.events"})).unwrap();
        assert_eq!(dt["database"], json!("analytics"));
        assert_eq!(dt["table"], json!("events"));
        let cdt = op_parse_table_name(json!({"name": "iceberg.analytics.events"})).unwrap();
        assert_eq!(cdt["catalog"], json!("iceberg"));
        assert_eq!(cdt["database"], json!("analytics"));
        assert_eq!(cdt["table"], json!("events"));
        assert!(op_parse_table_name(json!({"name": "a.b.c.d"})).is_err());
    }

    #[test]
    fn parse_table_name_honors_backtick_quoting() {
        // A `.` inside backticks is part of the name, not a separator.
        let t = op_parse_table_name(json!({"name": "db.`weird.table`"})).unwrap();
        assert_eq!(t["database"], json!("db"));
        assert_eq!(
            t["table"],
            json!("weird.table"),
            "dotted quoted name stays together"
        );
        // Doubled backtick is a literal backtick.
        let t2 = op_parse_table_name(json!({"name": "`a``b`"})).unwrap();
        assert_eq!(t2["table"], json!("a`b"));
    }

    #[test]
    fn build_table_name_inverts_parse_table_name() {
        // Plain three-level name stays clean (no needless quoting).
        assert_eq!(
            op_build_table_name(
                json!({"catalog": "iceberg", "database": "analytics", "table": "events"})
            )
            .unwrap()["name"],
            json!("iceberg.analytics.events")
        );
        // table-only and db.table forms.
        assert_eq!(
            op_build_table_name(json!({"table": "events"})).unwrap()["name"],
            json!("events")
        );
        assert_eq!(
            op_build_table_name(json!({"database": "db", "table": "t"})).unwrap()["name"],
            json!("db.t")
        );
        // A segment with a dot is backtick-quoted so it round-trips through parse.
        let built = op_build_table_name(json!({"database": "db", "table": "weird.table"})).unwrap()
            ["name"]
            .clone();
        assert_eq!(built, json!("db.`weird.table`"));
        let back = op_parse_table_name(json!({"name": built})).unwrap();
        assert_eq!(back["database"], json!("db"));
        assert_eq!(back["table"], json!("weird.table"));
        // catalog without database, and missing table, are rejected.
        assert!(op_build_table_name(json!({"catalog": "c", "table": "t"})).is_err());
        assert!(op_build_table_name(json!({"database": "db"})).is_err());
    }

    #[test]
    fn quote_ident_doubles_backticks() {
        assert_eq!(
            op_quote_ident(json!({"name": "weird`col"})).unwrap()["quoted"],
            json!("`weird``col`")
        );
    }

    #[test]
    fn unquote_ident_inverts_quote_ident() {
        // Doubled backtick decodes to one.
        assert_eq!(
            op_unquote_ident(json!({"quoted": "`weird``col`"})).unwrap()["name"],
            json!("weird`col")
        );
        // Plain and empty quoted names.
        assert_eq!(
            op_unquote_ident(json!({"quoted": "`plain`"})).unwrap()["name"],
            json!("plain")
        );
        assert_eq!(
            op_unquote_ident(json!({"quoted": "``"})).unwrap()["name"],
            json!("")
        );
        // Round-trips quote_ident for any input.
        for raw in ["table", "weird`col", "has space", "a``b`c"] {
            let q = op_quote_ident(json!({ "name": raw })).unwrap()["quoted"].clone();
            assert_eq!(
                op_unquote_ident(json!({ "quoted": q })).unwrap()["name"],
                json!(raw),
                "round-trip {raw:?}"
            );
        }
        // Not quoted / unpaired backtick reject.
        assert!(op_unquote_ident(json!({"quoted": "plain"})).is_err());
        assert!(op_unquote_ident(json!({"quoted": "`a`b`"})).is_err());
        assert!(op_unquote_ident(json!({})).is_err());
    }

    #[test]
    fn quote_qualified_ident_backticks_each_part() {
        // Three-level namespace, each part backtick-quoted.
        let v = op_quote_qualified_ident(json!({"name": "cat.db.my table"})).unwrap();
        assert_eq!(v["quoted"], json!("`cat`.`db`.`my table`"));
        assert_eq!(v["parts"], json!(["cat", "db", "my table"]));
        // Already-quoted part with an inner dot round-trips through parse_table_name.
        let q = op_quote_qualified_ident(json!({"name": "db.`weird.table`"})).unwrap();
        assert_eq!(q["quoted"], json!("`db`.`weird.table`"));
        let back = op_parse_table_name(json!({"name": q["quoted"].as_str().unwrap()})).unwrap();
        assert_eq!(back["database"], json!("db"));
        assert_eq!(back["table"], json!("weird.table"));
        // Bare name still backticked; over-deep names rejected.
        assert_eq!(
            op_quote_qualified_ident(json!({"name": "t"})).unwrap()["quoted"],
            json!("`t`")
        );
        assert!(op_quote_qualified_ident(json!({"name": "a.b.c.d"})).is_err());
    }
}
