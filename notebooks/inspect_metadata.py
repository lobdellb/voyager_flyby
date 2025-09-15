import marimo

__generated_with = "0.14.17"
app = marimo.App(width="medium")


@app.cell
def _():
    import sys
    import os

    sys.path.append("./")

    import marimo as mo
    import sqlite3, time, random
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import datetime
    import vicar
    import analysis

    pd.set_option('display.max_rows', 100)
    return analysis, datetime, np, pd, plt, sqlite3, vicar


@app.cell
def _(pd, sqlite3):
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
    df
    return (df,)


@app.cell
def _(df):
    print( df.columns)
    return


@app.cell
def _(df):
    # Let's see the proportion of wide vs narrow

    df.groupby("INSTRUMENT_NAME").count()["one"].reset_index()
    return


@app.cell
def _(df):
    # Let's look at counts of distinct values

    high_card = [
        "PRODUCT_ID",
        "IMAGE_ID",
        "IMAGE_NUMBER",
        "IMAGE_TIME",
        "EARTH_RECEIVED_TIME",
        "START_TIME",  
        "STOP_TIME",
        "SPACECRAFT_CLOCK_START_COUNT",
        "SPACECRAFT_CLOCK_STOP_COUNT",
        "NOTE"
    ]

    low_card = []

    for c in df.columns:

        n = df[c].nunique()

        if n > 1 and c not in high_card:
            print( f"{c:40} --> {n}")
            low_card.append( c )

    relevant = high_card + low_card
    print( low_card )
    print( relevant )
    return (low_card,)


@app.cell
def _(df, low_card):
    # let's look at unique values on the interesting low cards

    for c2 in low_card:

        summary_df = df.groupby(c2).count()["one"].reset_index()

        print("*"*100)
        print( c2 )
        print( summary_df )

    # summary_df
    return


@app.cell
def _(df, np):
    # Let's look at the photo timing

    start_time = np.array( list(filter( lambda s : s is not None, df["START_TIME"] ) ) )



    # plt.bar( start_time, n )
    return (start_time,)


@app.cell
def _(datetime, np, start_time):
    def generate_windowed_avg( events ):

        window = datetime.timedelta(hours=6)
        time_increment = datetime.timedelta(hours=1)

        events_dt = np.array([ datetime.datetime.fromisoformat(s) for s in events ])

        earliest = ( min( events_dt ) )
        latest = ( max( events_dt ) )

        print( earliest )
        print( latest )

        try:
            current_dt = earliest - window
        except Exception as e:
            print( earliest )
            print( window )
            print ( type( earliest ) )
            print( type( window ) )
            raise e

        times = []
        data = []

        while current_dt < latest + window:

            # print( current_dt )

            this_val = sum( ( events_dt > current_dt ) & ( events_dt < current_dt + window ) )
            data.append( this_val )
            times.append( current_dt )

            current_dt += time_increment 

        return np.array(times),np.array(data) # [ np.datetime64(s) for s in data ]

    times,data = generate_windowed_avg( start_time )
    return data, times


@app.cell
def _(data, datetime, plt, times):

    if "data" in locals():

        plt.figure(figsize=(14,4))
        # n3 = np.ones( len(data) )

        p = plt.plot( times, data/6)
        plt.xlim( [datetime.date(1980,8,18), datetime.date(1980,12,20) ])
        plt.xticks(rotation=45)
        plt.ylabel("photographs per hour")
        plt.xlabel("date")
        plt.title("Voyager 1 photographs per hour during the Saturn encounter")

    p
    return


@app.cell
def _(analysis, df, plt, vicar):
    # Next let's look at some of the photos

        # We need to do the scaled part here

        # scaled = ( scale_image( v_im.array.squeeze() ) * 255 ).astype( np.uint8 )

        # x,y,radius = find_circle_center( scaled ) 

        # new_image = center_object_in_larger_image( scaled.astype( np.float64 ), x, y ).astype( np.uint8 )

        # dim1, dim2, dim3 = new_image.shape

        # new_image = np.stack([new_image]*3, axis=-1).reshape(dim1,dim2,3)

    # rows, cols
    fig, axes = plt.subplots(8, 5, figsize=(10, 15))

    # Flatten the axes array for easy iteration
    axes = axes.ravel()

    # for i in range(16):
    #     axes[i].imshow(images[i], cmap="gray")
    #     axes[i].axis("off")  # Hide axes ticks




    for n3,(i,r) in enumerate( df[ df.TARGET_NAME == "TETHYS"].iterrows() ):

        if n3 > ( len(axes)-1 ):
            break

        v_im = vicar.VicarImage( r.LOCAL_FILENAME )

        im = analysis.scale_image( v_im.array.squeeze() )

        # print( n3 )
        # plt.imshow( im , cmap="grey")

        axes[n3].imshow( im, cmap="gray")
        axes[n3].axis("off")  # Hide axes ticks


    plt.tight_layout()
    plt.show()
    return (im,)


@app.cell
def _(im, plt):
    plt.imshow( im , cmap="grey")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
