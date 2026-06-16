```
 ███████╗████████╗██████╗ ██╗   ██╗██╗  ██╗███████╗
 ██╔════╝╚══██╔══╝██╔══██╗╚██╗ ██╔╝██║ ██╔╝██╔════╝
 ███████╗   ██║   ██████╔╝ ╚████╔╝ █████╔╝ █████╗
 ╚════██║   ██║   ██╔══██╗  ╚██╔╝  ██╔═██╗ ██╔══╝
 ███████║   ██║   ██║  ██║   ██║   ██║  ██╗███████╗
 ╚══════╝   ╚═╝   ╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝╚══════╝
                   [ s p a r k ]
```

[![CI](https://github.com/MenkeTechnologies/stryke-spark/actions/workflows/ci.yml/badge.svg)](https://github.com/MenkeTechnologies/stryke-spark/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![stryke](https://img.shields.io/badge/stryke-package-cyan.svg)](https://github.com/MenkeTechnologies/strykelang)

### `[APACHE SPARK CLIENT FOR STRYKE // OPT-IN PACKAGE]`

> *"Distributed compute from a stryke one-liner."*

Apache Spark client for stryke. Opt-in package, kept out of the stryke core
binary so the daily-driver install stays slim.

### [`strykelang`](https://github.com/MenkeTechnologies/strykelang) &middot; [`MenkeTechnologiesMeta`](https://github.com/MenkeTechnologies/MenkeTechnologiesMeta) · [`stryke-arrow`](https://github.com/MenkeTechnologies/stryke-arrow) · [`stryke-parquet`](https://github.com/MenkeTechnologies/stryke-parquet) · [`stryke-kafka`](https://github.com/MenkeTechnologies/stryke-kafka) · [`stryke-demo`](https://github.com/MenkeTechnologies/stryke-demo)

---

## Table of Contents

- [\[0x00\] Why this is a package, not a builtin](#0x00-why-this-is-a-package-not-a-builtin)
- [\[0x01\] Install](#0x01-install)
- [\[0x02\] Quick start](#0x02-quick-start)
- [\[0x04\] API reference](#0x04-api-reference)
- [\[0x06\] Type encoding](#0x06-type-encoding)
- [\[0x07\] Bind parameters](#0x07-bind-parameters)
- [\[0x08\] Performance notes](#0x08-performance-notes)
- [\[0x09\] Tests](#0x09-tests)
- [\[0x0A\] Dev workflow](#0x0a-dev-workflow)
- [\[0x0B\] Layout](#0x0b-layout)
- [\[0xFF\] License](#0xff-license)

---

## [0x00] Why this is a package, not a builtin

Same rationale as the other `stryke-*` data packages: Spark integration
requires the JVM, `spark-submit`, and PySpark on the host. Most stryke
one-liners never touch Spark; for the ones that do, opt in with this
package.

`stryke-spark` ships as a thin stryke library plus a Rust cdylib
(`libstryke_spark.{dylib,so}`). The cdylib shells out to `spark-submit`
with an **embedded PySpark driver** (`src/driver.py`, compiled in via
`include_str!`) that reads a JSON request envelope and writes JSON rows
to stdout. Universal across Spark 3.x and 4.x — anywhere `spark-submit`
runs, this works.

## [0x01] Install

From a release (no rustc on the consumer machine):

```sh
s pkg install -g github.com/MenkeTechnologies/stryke-spark
```

From a local checkout:

```sh
cd ~/projects/stryke-spark
cargo build --release          # produces target/release/libstryke_spark.{dylib,so}
s pkg install -g .             # cdylib lands in ~/.stryke/store/spark@<version>/
```

Or:

```sh
make install
```

The cdylib is dlopened in-process on first `use Spark`. **Honest scope
note:** each call still pays SparkSession init cost (seconds, dominated
by JVM warmup). A long-running JVM driver daemon that persists
`SparkSession` across calls is deferred — it needs a sidecar process
design that's larger than the v0.2.1 helper-binary → cdylib refactor.
What the cdylib model does eliminate is the helper-binary fork+exec
overhead on top of spark-submit.

You also need `spark-submit` reachable: install Spark via `brew install
apache-spark`, your distro's package, or unpack a tarball and set
`$SPARK_HOME`.

### JDK compatibility

Spark 4.x officially supports JDK 17 — JDK 21+ trips a
`getSubject is not supported` error in the Hive catalog code path even
under `local[*]`. The cdylib defaults to
`--conf spark.sql.catalogImplementation=in-memory` to dodge Hive, but a
JDK 17 environment is still the smoothest. Set `JAVA_HOME` before running:

```sh
export JAVA_HOME=/path/to/jdk-17     # e.g. corretto-17, temurin-17
```

## [0x02] Quick start

```stryke
use Spark

# Plain query — defaults to --master local[*].
my @rows = Spark::query "
    SELECT id, id * 2 AS doubled
    FROM range(5)
"
@rows |> ep

# Against a remote cluster.
my @rows = Spark::query "SELECT * FROM events WHERE day = '2026-01-01'",
                        master => "spark://cluster:7077",
                        confs  => { "spark.executor.memory" => "8g",
                                    "spark.executor.cores"  => "4" }

# Scalar shortcut.
p Spark::query_scalar "SELECT COUNT(*) FROM range(1000000)"

# DDL (returns { ok: true }).
Spark::execute "CREATE TABLE IF NOT EXISTS logs (ts TIMESTAMP, msg STRING)"

# Schema + table listings.
p to_json Spark::schema "logs"
p Spark::tables |> ep
p Spark::databases |> ep

# Pass-through to spark-submit for jobs outside the SQL surface.
Spark::submit "jobs/etl_pipeline.py",
              args  => ["--date", "2026-01-01"],
              confs => { "spark.driver.memory" => "4g" }
```

Each Spark call spins up a fresh JVM (~5–10s warmup). For multi-statement
work, prefer one SQL with CTEs / subqueries over many separate calls.

## [0x04] API reference

### Read paths

```stryke
Spark::query        $sql, %opts → @rows
Spark::query_stream $sql, %opts → $count               # callback per row
Spark::query_one    $sql, %opts → \%row | undef
Spark::query_col    $sql, %opts → @values
Spark::query_scalar $sql, %opts → $value | undef
Spark::dump         $table, %opts → @rows
Spark::count        $table, $where?, %opts → $row_count   # SELECT count(*) [WHERE $where]
```

`%opts` keys: `master`, `spark_home`, `spark_submit`, `app_name`,
`deploy_mode`, `packages`, `jars`, `database`, `confs` (hashref),
`limit`, `callback` (stream only).

### DDL / DML

```stryke
Spark::execute    $sql, %opts → { ok: true }
Spark::explain    $sql, %opts → $plan_text     # opts: mode (simple|extended|codegen|cost|formatted)
```

DDL covers `CREATE TABLE`, `INSERT INTO`, `DROP`, `MERGE`, etc. Spark's
own SQL parser handles the dispatch; the driver just runs `spark.sql(...)`
and emits a single `{ok}` ack on success. `explain` returns the query plan.

### External read / write

```stryke
Spark::read   $path, %opts → @rows         # opts: format, options, view, sql, limit
Spark::write  $sql,  %opts → { ok, ... }   # opts: path|table, format, mode, options
```

`read` loads a parquet/csv/json/orc source; pass `view => "v", sql => "SELECT
… FROM v"` to query it in the same call (each call is a fresh session).
`write` runs `$sql` and saves the result to a `path` or `table`, with `mode`
∈ `overwrite|append|ignore|errorifexists`.

### Metadata

```stryke
Spark::ping       %opts → 1 | 0
Spark::tables     %opts → @rows            # catalog rows (in-memory or hive)
Spark::databases  %opts → @rows
Spark::views      %opts → @rows            # views (catalog tableType VIEW / temp)
Spark::catalogs   %opts → @rows            # { name, description }
Spark::current_database %opts → $name      # current database
Spark::create_temp_view $name, $sql, %opts → \%resp  # register $sql as a temp view
Spark::drop_temp_view   $name, %opts → \%resp
Spark::set_database     $database, %opts → \%resp    # catalog.setCurrentDatabase
Spark::refresh_table    $table, %opts → \%resp       # catalog.refreshTable
Spark::schema     $table, %opts → @rows    # DESCRIBE TABLE column rows
Spark::columns    $table, %opts → @rows    # catalog columns (name/type/nullable/partition/bucket)
Spark::functions  %opts → @rows            # catalog functions
```

### Caching + runtime config

```stryke
Spark::cache    $table, %opts → { ok, cached }
Spark::uncache  $table, %opts → { ok, uncached }
Spark::config   $key, %opts → $value | { ok }   # set with value => ...
```

### Pure helpers (no Spark)

```stryke
Spark::parse_master_url($url)   → { scheme, threads?, hosts?, master? }   # local[N], spark://… HA, k8s://…, yarn
Spark::build_master_url(%opts)  → $url   # { scheme, threads|hosts|master } → master URL; inverse of parse_master_url
Spark::parse_table_name($name)  → { catalog, database, table, parts }     # backtick-aware catalog.db.table
Spark::build_table_name(%opts)  → $name                                   # catalog/database/table → dotted name; inverse of parse_table_name
Spark::parse_memory($memory)    → { value, suffix, bytes, mib }           # Spark size config 512m/2g/1kb → bytes (binary suffixes: 1kb=1024)
Spark::build_memory($bytes)     → { value, suffix, string, bytes }        # bytes → Spark size string (largest binary unit that divides evenly); inverse of parse_memory
Spark::quote_ident($name)       → $quoted                                 # `weird``col`
Spark::unquote_ident($quoted)   → $name                                   # inverse of quote_ident: strip backticks, un-double
Spark::quote_qualified_ident($name) → $quoted                            # cat.db.my table → `cat`.`db`.`my table`
```

### Submit pass-through

```stryke
Spark::submit $script_path, args => [...], %opts → { exit_code, output }
```

Runs the script through `spark-submit`. Use for `.py` / `.jar` workloads.

### Versions

```stryke
Spark::version()  → package version string
```

The embedded PySpark driver lives in `src/driver.py` (compiled into the
cdylib via `include_str!`). It is written to a temp file at run time so
`spark-submit` can pick it up.

## [0x06] Type encoding

Spark `df.toJSON()` does the heavy lifting; types map to JSON as Spark's
JSON serializer dictates:

| Spark | JSON |
|---|---|
| `boolean` | bool |
| `byte`, `short`, `int`, `long` | number |
| `float`, `double` | number |
| `decimal(p,s)` | number (precision permitting) |
| `string`, `varchar`, `char` | string |
| `binary` | base64 string |
| `date` | `"yyyy-MM-dd"` |
| `timestamp` | `"yyyy-MM-dd HH:mm:ss"` |
| `array<T>` | JSON array |
| `struct<…>` | JSON object |
| `map<K,V>` | JSON object |
| `NULL` | null |

The columnar path also coerces Python `date`/`datetime`/`Decimal` to
strings if Spark's serializer leaves them as native Python objects.

## [0x07] Bind parameters

Spark SQL doesn't accept positional binds the way Postgres / MySQL do (the
3.5+ `args=` keyword on `SparkSession.sql` is gated on Connect for some
deployments). For v1, inline values into the SQL string. Use literal
quoting at the Spark SQL level (`'string'`, numeric, date literals
`DATE '2026-01-01'`, etc.).

Bind support via the cdylib's request JSON can be added once a clean
cross-version path exists.

## [0x08] Performance notes

* Each call boots a fresh JVM via `spark-submit`. Plan for ~5–10s startup per call.
* Batch work into one `query` with CTEs / subqueries / temp views when
  possible — that's a single submit, one JVM.
* Local Spark warehouse files land under `./spark-warehouse/` and a
  `metastore_db/` directory in the cwd. Both are in the `.gitignore`.
* For interactive work against a remote cluster, point `--master` at a
  long-running standalone / YARN / k8s Spark cluster — the submit time is
  the same but the actual compute runs on warm executors.

## [0x09] Tests

```sh
cargo test                                       # Rust unit tests, no live JVM
JAVA_HOME=/path/to/jdk-17 s test t/              # end-to-end against local[*]
```

The end-to-end suite skips cleanly when `spark-submit` isn't on PATH or
the JVM can't start.

## [0x0A] Dev workflow

```sh
make             # release build
make test
make install     # release + pkg install -g .
make clean
```

## [0x0B] Layout

```
stryke-spark/
  stryke.toml                    # stryke package manifest ([ffi] table)
  Cargo.toml                     # cdylib crate manifest
  Makefile
  src/
    lib.rs                       # cdylib — spark__* extern "C" exports
    driver.py                    # embedded PySpark driver (include_str!)
  lib/
    Spark.stk                    # `use Spark`
  t/
    test_spark.stk               # live end-to-end suite (skips without spark-submit)
    test_stryke_spark_surface.stk
  tests/
    contract_cli_round4.rs       # Rust contract tests (+ repo lint gates *.sh)
  examples/
    discover.stk
    quick_query.stk
    range_stats.stk
    sql_explain.stk
    parquet_pipeline.stk         # pairs with stryke-arrow
  .github/workflows/
    ci.yml                       # cargo + install Spark + local[*] smoke
    release.yml                  # cross-compile + GH release on tag push
```

## [0xFF] License

MIT.
