# stryke-spark embedded PySpark driver.
#
# Invoked by `stryke-spark-helper` via `spark-submit`. Reads a single JSON
# arg from argv[1] describing what to run; emits NDJSON rows or a JSON
# summary on stdout. Status / errors go to stderr.

import json
import sys
import traceback


def _err(msg):
    sys.stderr.write("stryke-spark driver: " + msg + "\n")


def _out(obj):
    sys.stdout.write(json.dumps(obj) + "\n")


def _row_columns(df):
    return [f.name for f in df.schema.fields]


def cmd_query(spark, req):
    sql = req["sql"]
    df = spark.sql(sql)
    if req.get("limit"):
        df = df.limit(int(req["limit"]))
    cols = _row_columns(df)

    if req.get("columnar"):
        rows = [list(r) for r in df.collect()]
        # Make rows JSON-serializable (Python date/datetime/Decimal → str).
        rows = [[_safe(v) for v in r] for r in rows]
        _out({"columns": cols, "num_rows": len(rows), "rows": rows})
        return

    if req.get("with_meta"):
        _out({"meta": {"columns": cols}})

    # Streaming NDJSON via df.toJSON(). Each element is a JSON string of one row.
    for json_row in df.toJSON().collect():
        sys.stdout.write(json_row)
        sys.stdout.write("\n")
    sys.stdout.flush()


def cmd_execute(spark, req):
    spark.sql(req["sql"])
    _out({"ok": True})


def cmd_dump(spark, req):
    tbl = _qident(req["table"])
    cols = req.get("columns") or "*"
    sql = "SELECT " + cols + " FROM " + tbl
    if req.get("where"):
        sql += " WHERE " + req["where"]
    if req.get("order_by"):
        sql += " ORDER BY " + req["order_by"]
    if req.get("limit"):
        sql += " LIMIT " + str(int(req["limit"]))
    df = spark.sql(sql)
    for json_row in df.toJSON().collect():
        sys.stdout.write(json_row)
        sys.stdout.write("\n")
    sys.stdout.flush()


def cmd_tables(spark, req):
    # Use the Catalog API rather than `SHOW TABLES` so we don't depend on
    # the Hive metastore being reachable (it isn't, on JDK 17+ with no
    # security manager). `spark.catalog.listTables()` works against the
    # default in-memory catalog.
    try:
        for t in spark.catalog.listTables():
            _out({
                "name": t.name,
                "database": getattr(t, "database", None) or getattr(t, "namespace", None),
                "is_temp": getattr(t, "isTemporary", None),
                "type": getattr(t, "tableType", None),
            })
    except Exception:
        # Fall back to SHOW TABLES for clusters where catalog API fails.
        df = spark.sql("SHOW TABLES")
        for row in df.collect():
            d = row.asDict()
            _out({
                "name": d.get("tableName") or d.get("table_name") or (row[1] if len(row) > 1 else row[0]),
                "database": d.get("namespace") or d.get("database") or (row[0] if len(row) > 1 else None),
                "is_temp": d.get("isTemporary"),
            })


def cmd_databases(spark, req):
    # Catalog API first; falls back to SHOW DATABASES if it isn't supported.
    try:
        for d in spark.catalog.listDatabases():
            _out({
                "name": d.name,
                "description": getattr(d, "description", None),
                "location": getattr(d, "locationUri", None),
            })
    except Exception:
        df = spark.sql("SHOW DATABASES")
        for row in df.collect():
            d = row.asDict()
            name = d.get("namespace") or d.get("databaseName") or d.get("name") or row[0]
            _out({"name": name})


def cmd_schema(spark, req):
    tbl = _qident(req["table"])
    df = spark.sql("DESCRIBE TABLE EXTENDED " + tbl)
    columns = []
    properties = {}
    section = "columns"
    for row in df.collect():
        col = (row[0] or "").strip() if row[0] is not None else ""
        ty = (row[1] or "").strip() if row[1] is not None else ""
        comment = (row[2] or "").strip() if (len(row) > 2 and row[2] is not None) else ""
        # The "Detailed Table Information" sentinel divides the columns from
        # the property block in `DESCRIBE EXTENDED`.
        if col == "" and ty == "":
            section = "props"
            continue
        if col.startswith("#"):
            # Section header like "# Partition Information".
            if "Partition" in col:
                section = "partitions"
            elif "Detailed" in col:
                section = "props"
            continue
        if section == "columns":
            columns.append({"name": col, "type": ty, "comment": comment or None})
        else:
            if col:
                properties[col] = ty
    _out({
        "table": req["table"],
        "columns": columns,
        "properties": properties,
    })


def cmd_explain(spark, req):
    # EXPLAIN [EXTENDED|CODEGEN|COST|FORMATTED] <statement>. SIMPLE is the
    # default (no keyword). Collect the single plan column into one text blob.
    mode = (req.get("mode") or "formatted").upper()
    keyword = "" if mode in ("", "SIMPLE") else mode + " "
    plan_df = spark.sql("EXPLAIN " + keyword + req["sql"])
    text = "\n".join((r[0] or "") for r in plan_df.collect())
    _out({"plan": text})


def _apply_options(builder, options):
    for k, v in (options or {}).items():
        builder = builder.option(k, str(v))
    return builder


def cmd_read(spark, req):
    # Load an external source (parquet/csv/json/orc/...) and either preview it
    # or run a follow-up SQL against it via a registered temp view. Each call
    # is a fresh SparkSession, so the read + query happen together here.
    fmt = req.get("format") or "parquet"
    reader = _apply_options(spark.read.format(fmt), req.get("options"))
    df = reader.load(req["path"])
    view = req.get("view")
    if view:
        df.createOrReplaceTempView(view)
    if req.get("sql"):
        df = spark.sql(req["sql"])
    if req.get("limit"):
        df = df.limit(int(req["limit"]))
    for json_row in df.toJSON().collect():
        sys.stdout.write(json_row)
        sys.stdout.write("\n")
    sys.stdout.flush()


def cmd_write(spark, req):
    # Run `sql` to produce a DataFrame, then write it to a path or table.
    df = spark.sql(req["sql"])
    fmt = req.get("format") or "parquet"
    mode = req.get("mode") or "errorifexists"
    writer = _apply_options(df.write.format(fmt).mode(mode), req.get("options"))
    if req.get("table"):
        writer.saveAsTable(req["table"])
        _out({"ok": True, "format": fmt, "mode": mode, "table": req["table"]})
    else:
        writer.save(req["path"])
        _out({"ok": True, "format": fmt, "mode": mode, "path": req["path"]})


def cmd_functions(spark, req):
    for f in spark.catalog.listFunctions():
        _out({
            "name": f.name,
            "description": getattr(f, "description", None),
            "class_name": getattr(f, "className", None),
            "is_temporary": getattr(f, "isTemporary", None),
        })


def cmd_views(spark, req):
    # Views surface through the Catalog API as tables whose tableType is a
    # VIEW (or that are temporary). Fall back to SHOW VIEWS on clusters where
    # the catalog API isn't available.
    try:
        for t in spark.catalog.listTables():
            tt = str(getattr(t, "tableType", "") or "")
            is_temp = getattr(t, "isTemporary", None)
            if "VIEW" in tt.upper() or is_temp:
                _out({
                    "name": t.name,
                    "database": getattr(t, "database", None) or getattr(t, "namespace", None),
                    "is_temp": is_temp,
                    "type": getattr(t, "tableType", None),
                })
    except Exception:
        df = spark.sql("SHOW VIEWS")
        for row in df.collect():
            d = row.asDict()
            _out({
                "name": d.get("viewName") or d.get("view_name"),
                "database": d.get("namespace") or d.get("database"),
                "is_temp": d.get("isTemporary"),
            })


def cmd_catalogs(spark, req):
    try:
        for c in spark.catalog.listCatalogs():
            _out({
                "name": getattr(c, "name", None),
                "description": getattr(c, "description", None),
            })
    except Exception:
        df = spark.sql("SHOW CATALOGS")
        for row in df.collect():
            _out({"name": row[0]})


def cmd_current_database(spark, req):
    _out({"database": spark.catalog.currentDatabase()})


def cmd_create_temp_view(spark, req):
    spark.sql(req["sql"]).createOrReplaceTempView(req["name"])
    _out({"ok": True, "view": req["name"]})


def cmd_drop_temp_view(spark, req):
    dropped = spark.catalog.dropTempView(req["name"])
    _out({"ok": True, "dropped": bool(dropped)})


def cmd_set_database(spark, req):
    spark.catalog.setCurrentDatabase(req["database"])
    _out({"ok": True, "database": req["database"]})


def cmd_refresh_table(spark, req):
    spark.catalog.refreshTable(req["table"])
    _out({"ok": True, "refreshed": req["table"]})


def cmd_columns(spark, req):
    for c in spark.catalog.listColumns(req["table"]):
        _out({
            "name": c.name,
            "type": getattr(c, "dataType", None),
            "nullable": getattr(c, "nullable", None),
            "is_partition": getattr(c, "isPartition", None),
            "is_bucket": getattr(c, "isBucket", None),
            "description": getattr(c, "description", None),
        })


def cmd_cache(spark, req):
    spark.catalog.cacheTable(req["table"])
    _out({"ok": True, "cached": req["table"]})


def cmd_uncache(spark, req):
    spark.catalog.uncacheTable(req["table"])
    _out({"ok": True, "uncached": req["table"]})


def cmd_config(spark, req):
    key = req["key"]
    if "value" in req and req["value"] is not None:
        spark.conf.set(key, str(req["value"]))
        _out({"ok": True, "key": key})
    else:
        _out({"key": key, "value": spark.conf.get(key, None)})


def cmd_ping(spark, req):
    val = spark.sql("SELECT 1 AS one").collect()[0][0]
    if val == 1:
        sys.stdout.write("ok\n")
        sys.stdout.flush()
    else:
        raise RuntimeError("SELECT 1 returned " + repr(val))


def _qident(s):
    # Spark SQL identifier quoting via backticks; double any embedded.
    return "`" + s.replace("`", "``") + "`"


def _safe(v):
    # Make non-JSON-native types serializable for the columnar path.
    try:
        json.dumps(v)
        return v
    except (TypeError, ValueError):
        return str(v)


DISPATCH = {
    "query": cmd_query,
    "execute": cmd_execute,
    "dump": cmd_dump,
    "explain": cmd_explain,
    "read": cmd_read,
    "write": cmd_write,
    "tables": cmd_tables,
    "databases": cmd_databases,
    "views": cmd_views,
    "catalogs": cmd_catalogs,
    "current_database": cmd_current_database,
    "create_temp_view": cmd_create_temp_view,
    "drop_temp_view": cmd_drop_temp_view,
    "set_database": cmd_set_database,
    "refresh_table": cmd_refresh_table,
    "schema": cmd_schema,
    "columns": cmd_columns,
    "functions": cmd_functions,
    "cache": cmd_cache,
    "uncache": cmd_uncache,
    "config": cmd_config,
    "ping": cmd_ping,
}


def main():
    if len(sys.argv) < 2:
        _err("missing request JSON arg")
        sys.exit(2)

    try:
        req = json.loads(sys.argv[1])
    except Exception as e:
        _err("parse request: " + str(e))
        sys.exit(2)

    cmd = req.get("cmd")
    if cmd not in DISPATCH:
        _err("unknown cmd: " + repr(cmd))
        sys.exit(2)

    # Import here so spark-submit's classpath is fully built before we touch
    # pyspark internals.
    from pyspark.sql import SparkSession

    spark = SparkSession.builder.getOrCreate()
    try:
        if req.get("database"):
            spark.sql("USE " + _qident(req["database"]))
        DISPATCH[cmd](spark, req)
    except Exception as e:
        _err("driver error: " + str(e))
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
