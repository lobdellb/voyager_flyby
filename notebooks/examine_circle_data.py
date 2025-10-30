import marimo

__generated_with = "0.14.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import sys
    import os
    import cProfile
    import io
    import pstats
    import marimo as mo
    import pickle

    # print( os.environ["MARIMO_OUTPUT_MAX_BYTES"] )

    if os.getcwd().endswith("src"):
        os.chdir( os.getcwd() + "/.." )

    sys.path[0] = sys.path[0].replace("notebooks","src")


    print( sys.path )
    print( os.getcwd() )

    import marimo as mo
    import sqlite3, time, random
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import datetime
    import vicar
    # import analysis
    import cv2
    from functools import wraps

    # import repository.image as image
    # import database as db
    return datetime, pd, plt, sqlite3


@app.cell
def _(datetime, pd, sqlite3):
    fn = "/home/lobdellb/repos/voyager_flyby/cache/pipeline.db"

    with sqlite3.connect( fn ) as conn:
        # Step 2: Define your SQL query.
        # This can be as simple or complex as needed.
        sql_query = "SELECT * FROM voyager_images"

        # Step 3: Load the query results into a pandas DataFrame.
        df = pd.read_sql_query(sql_query, conn)

        # Step 4: Display the DataFrame to verify the results.

    print( df.shape )

    df["one"] = 1
    df["START_TIME"] = df.START_TIME.apply( lambda s : None if s is None else datetime.datetime.fromisoformat(s) )
    df
    return


@app.cell
def _(plt):
    plt.plot( )
    return


if __name__ == "__main__":
    app.run()
