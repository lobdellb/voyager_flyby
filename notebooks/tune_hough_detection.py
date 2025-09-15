import marimo

__generated_with = "0.14.17"
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

    # sys.path.append("./")

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
    import analysis
    import cv2

    import repository.image as image
    import database as db

    pd.set_option('display.max_rows', 100)


    return (
        analysis,
        cProfile,
        cv2,
        datetime,
        io,
        np,
        pd,
        pickle,
        plt,
        pstats,
        sqlite3,
    )


@app.cell
def _():
    # with db.SessionLocal() as session:
    #     image_record_obj = image.get_voyager_image_by_product_id(session , "C3592952_GEOMED.IMG")


    # # print( image_record_obj.LOCAL_IMAGE_PICKLE_FN )
    # print( image_record_obj.DATA_SET_ID )

    # mo.stop(True)
    return


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
    return (df,)


@app.cell
def _(df):
    # Let's see the proportion of wide vs narrow

    df.groupby("INSTRUMENT_NAME").count()["one"].reset_index()
    return


@app.cell
def _(df):
    df.columns
    return


@app.cell
def _(analysis, cProfile, cv2, datetime, df, io, np, pickle, plt, pstats):
    pr = cProfile.Profile()

    with pr:

        # Next let's look at some of the photos

            # We need to do the scaled part here

            # scaled = ( scale_image( v_im.array.squeeze() ) * 255 ).astype( np.uint8 )

            # x,y,radius = find_circle_center( scaled ) 

            # new_image = center_object_in_larger_image( scaled.astype( np.float64 ), x, y ).astype( np.uint8 )

            # dim1, dim2, dim3 = new_image.shape

            # new_image = np.stack([new_image]*3, axis=-1).reshape(dim1,dim2,3)



        # for i in range(16):
        #     axes[i].imshow(images[i], cmap="gray")
        #     axes[i].axis("off")  # Hide axes ticks

        # Let's get some in several date ranges to get a sample of the various sizes in play
        # date_ranges = [
        #     datetime.datetime(1980,9,1),
        #     datetime.datetime(1980,9,15),
        #     datetime.datetime(1980,10,1),
        #     datetime.datetime(1980,10,15),
        #     datetime.datetime(1980,11,1),
        #     datetime.datetime(1980,11,16),
        #     datetime.datetime(1980,12,2),
        # ]

        # all_dfs = []

        # for date_range in date_ranges:
        #     this_df = df[ 
        #         ( df.TARGET_NAME == "SATURN")
        #         & ( df.INSTRUMENT_NAME == "IMAGING SCIENCE SUBSYSTEM - NARROW ANGLE" )
        #         & ( df.START_TIME > date_range  )
        #         & ( df.START_TIME <= ( date_range + datetime.timedelta(hours=6) )  )
        #     ]

        #     print( this_df.shape )

        #     all_dfs.append( this_df )

        # these_df = pd.concat( all_dfs )
        # # these_df = df[ ( df.TARGET_NAME == "SATURN") & ( df.START_TIME > datetime.datetime(1980,10,15)  ) ].sort_values("START_TIME")

        these_df = df[ 
            ( df.TARGET_NAME == "SATURN")
            & ( df.INSTRUMENT_NAME == "IMAGING SCIENCE SUBSYSTEM - NARROW ANGLE" )
            & ( df.START_TIME > datetime.datetime(1980,8,20)  )
        ].sort_values("START_TIME").iloc[::50]



        print( these_df.shape )

        # rows, cols
        how_many_rows = int( np.ceil( these_df.shape[0] / 5 ) )
        how_tall = 2* how_many_rows
        fig, axes = plt.subplots( how_many_rows , 5, figsize=(10, how_tall))

        # Flatten the axes array for easy iteration
        axes = axes.ravel()


        colors_rgb = [
            (228, 26, 28),    # red
            (55, 126, 184),   # blue
            (77, 175, 74),    # green
            (152, 78, 163),   # purple
            (255, 127, 0),    # orange
            (255, 255, 51),   # yellow
        ]


        for n3,(i,r) in enumerate( these_df.iterrows() ):

            if n3 > ( len(axes)-1 ):
                break

            with open( r.LOCAL_IMAGE_PICKLE_FN, "rb" ) as fp2:
                # v_im = vicar.VicarImage( r.LOCAL_FILENAME )

                v_im = pickle.load( fp2 )

            im = analysis.scale_image( v_im.array.squeeze() )

            im_prepped_for_cv2 = ( im * 255 ).astype( np.uint8 )

            # circles = analysis.find_circle_center( im_prepped_for_cv2 ) 
            circles = analysis.find_circle_center_parametrized(
                    im_prepped_for_cv2,
                    blur_width=5,
                    method=cv2.HOUGH_GRADIENT_ALT,
                    dp=1,
                    minDist=50,
                    param1=300,
                    param2=0.9,
                    minRadius=25,
                    maxRadius=0 
                )


            dim1, dim2, dim3 = im.shape
            new_image = np.stack([im*255]*3, axis=-1).reshape(dim1,dim2,3).astype( np.uint8 )

            if circles is None:
                # print("none")
                pass
            else:

                # print( type( circles[0][0][0] ) )

                # circles = np.uint16(np.around(circles))

                circles = circles[0] # get the list of circles

                # print( circles )

            #     x, y, r = circles[0][0]

            #     return int(x), int(y), int(r)

                # print( len( circles ) ) 

                for color_index,(x,y,radius) in enumerate( circles ):

                # x,y,radius = circles[0]
                    color_index = min( len(colors_rgb)-1, color_index )
                    im_annotated = cv2.circle(new_image, (int(x),int(y)), int(radius) , colors_rgb[ color_index ], 4 )
                    pass

            # print( x, y, radius )

            # print( n3 )
            # plt.imshow( im , cmap="grey")



            # print( dim1, dim2, dim3 )



            axes[n3].imshow( new_image, cmap="gray")

        for a in axes:
            a.axis("off")  # Hide axes ticks


        plt.tight_layout()
        plt.show()

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(10)   # top 10 results
    print(s.getvalue())
    return (v_im,)


@app.cell
def _():


    # How to proceed?

    # - Write down the nobs and levers we have.
    # DONE - See if chatgpt can do this and how much it cost.  <-- it sucks at it
    # DONE - Diplay the items which did not get a circle
    #   - see whether there is sensitivity lever available
    # DONE - Look all the circles drawn on the images.
    # DONE - Space the images out reasonable, 1/100 shown


    # Parameters
    # - The medianBlur size - odd numbers only, 3 sucks, 7 is about the same as 5, 9 starts getting doubles and less accuracy
    # - maxRadius - Made 2000 rather than image width (1000, by default) so that it could get bigger circles.


    # Learnings
    # - It looks like Hough does great so long as I always pick the smallest circle.
    # 



    return


@app.cell
def _():
    print("done")
    return


@app.cell
def _():
    # I want to do a hyperparameter search for the params of the Hough circle transform.
    # The parameters are:
    # int	method,   --> HOUGH_GRADIENT , HOUGH_GRADIENT_ALT , HOUGH_STANDARD
    # double	dp,   --> ratio of accumulator resolution to image resolution 0.5, 1, 2
    # double	minDist --> 0 , 100, 200 , 400, 800 
    # double	param1 = 100,
    # double	param2 = 100,
    # int	minRadius = 0,
    # int	maxRadius = 0 )

    # we're going to measure the correctness by similarity to neighbors location and similarity to the modeled radius


    return


@app.cell
def _():
    return


@app.cell
def _(pickle, v_im):

    # v_im

    with open("/home/lobdellb/garbage/my_data.pkl", "wb") as fp:
        pickle.dump(v_im, fp)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
