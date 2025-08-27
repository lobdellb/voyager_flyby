import marimo

__generated_with = "0.14.17"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import sqlite3, time, random

    return mo, sqlite3, time


@app.cell
def _(mo, sqlite3, time):

    with sqlite3.connect("file:////home/lobdellb/repos/voyager_flyby/cache/pipeline.db?mode=ro") as conn:
        #conn.execute("PRAGMA journal_mode=WAL;")
        #conn.execute("PRAGMA synchronous=NORMAL;")
        #conn.execute("PRAGMA temp_store=MEMORY;")
        # if 64-bit system and modern SQLite:
        #conn.execute("PRAGMA mmap_size=268435456;")  # 256MB (optional)
        #conn.execute("PRAGMA cache_size=-1048576;")  # ~1GB cache in pages (negative = KB)
        #conn.execute("PRAGMA optimize;")

        # make sure stats exist
        #conn.execute("ANALYZE;")

        cur = conn.cursor()

        # get all product_ids or a lot

        r_ids = cur.execute("select product_id from voyager_images").fetchall()
        all_ids = [ r[0] for r in r_ids ]

        # Use a prepared statement and avoid SELECT *
        # sql = "SELECT * FROM voyager_images WHERE PRODUCT_ID=?;"

        sql = """SELECT voyager_images."PDS_VERSION_ID", voyager_images."RECORD_TYPE", voyager_images."RECORD_BYTES", voyager_images."FILE_RECORDS", voyager_images."DATA_SET_ID", voyager_images."PRODUCT_ID", voyager_images."PRODUCT_CREATION_TIME", voyager_images."PRODUCT_TYPE", voyager_images."INSTRUMENT_HOST_NAME", voyager_images."INSTRUMENT_HOST_ID", voyager_images."INSTRUMENT_NAME", voyager_images."INSTRUMENT_ID", voyager_images."MISSION_PHASE_NAME", voyager_images."TARGET_NAME", voyager_images."IMAGE_ID", voyager_images."IMAGE_NUMBER", voyager_images."IMAGE_TIME", voyager_images."EARTH_RECEIVED_TIME", voyager_images."SCAN_MODE_ID", voyager_images."SHUTTER_MODE_ID", voyager_images."GAIN_MODE_ID", voyager_images."EDIT_MODE_ID", voyager_images."FILTER_NAME", voyager_images."FILTER_NUMBER", voyager_images."EXPOSURE_DURATION_value", voyager_images."EXPOSURE_DURATION_units", voyager_images."START_TIME", voyager_images."STOP_TIME", voyager_images."SPACECRAFT_CLOCK_START_COUNT", voyager_images."SPACECRAFT_CLOCK_STOP_COUNT", voyager_images."NOTE", voyager_images."VICAR_HEADER_HEADER_TYPE", voyager_images."VICAR_HEADER_BYTES", voyager_images."VICAR_HEADER_RECORDS", voyager_images."VICAR_HEADER_INTERCHANGE_FORMAT", voyager_images."IMAGE_LINES", voyager_images."IMAGE_LINE_SAMPLES", voyager_images."IMAGE_SAMPLE_TYPE", voyager_images."IMAGE_SAMPLE_BITS", voyager_images."IMAGE_SAMPLE_DISPLAY_DIRECTION", voyager_images."IMAGE_LINE_DISPLAY_DIRECTION", voyager_images."IMAGE_HORIZONTAL_PIXEL_FOV_value", voyager_images."IMAGE_HORIZONTAL_PIXEL_FOV_units", voyager_images."IMAGE_VERTICAL_PIXEL_FOV_value", voyager_images."IMAGE_VERTICAL_PIXEL_FOV_units", voyager_images."IMAGE_HORIZONTAL_FOV_value", voyager_images."IMAGE_HORIZONTAL_FOV_units", voyager_images."IMAGE_VERTICAL_FOV_value", voyager_images."IMAGE_VERTICAL_FOV_units", voyager_images."IMAGE_REFLECTANCE_SCALING_FACTOR" 
    FROM voyager_images WHERE voyager_images."PRODUCT_ID" = ?"""


        # ids = [f"C3517{i:03d}_GEOMED.IMG" for i in range(1000)]  # replace with real keys present

        # # warm up
        # for k in ids[:100]: cur.execute(sql, (k,)).fetchone()

        # t0 = time.perf_counter()
        # for k in ids:
        #     cur.execute(sql, (k,)).fetchone()
        # t1 = time.perf_counter()
        # print(f"{len(ids)/(t1-t0):.0f} lookups/sec")

        start_time = time.time()
        for id in mo.status.progress_bar( all_ids ):
            r = cur.execute(sql,( id ,)).fetchone()
            if len(r) != 50:
                raise Exception("what?")

        end_time = time.time()

        # r = cur.execute(sql,(k,)).fetchone()

        print(f"took {(end_time-start_time)/len(all_ids):.6f}ms per item")
    return (r,)


@app.cell
def _(r):
    r
    return


@app.cell
def _(sqlite3):
    with sqlite3.connect("file:////home/lobdellb/repos/voyager_flyby/cache/pipeline.db?mode=ro") as conn:
        #conn.execute("PRAGMA journal_mode=WAL;")
        #conn.execute("PRAGMA synchronous=NORMAL;")
        #conn.execute("PRAGMA temp_store=MEMORY;")
        # if 64-bit system and modern SQLite:
        #conn.execute("PRAGMA mmap_size=268435456;")  # 256MB (optional)
        #conn.execute("PRAGMA cache_size=-1048576;")  # ~1GB cache in pages (negative = KB)
        #conn.execute("PRAGMA optimize;")

        # make sure stats exist
        #conn.execute("ANALYZE;")

        cur = conn.cursor()

        # get all product_ids or a lot

        r = cur.execute("select * from voyager_images where product_id like 'C2783018%' ").fetchall()

    r
    return (r,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
