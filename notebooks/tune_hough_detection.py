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
    from functools import wraps

    import repository.image as image
    import database as db

    pd.set_option('display.max_rows', 100)

    import time
    from functools import wraps

    # Global accumulator: dict of name -> accumulated time
    execution_time_accumulator: dict[str, float] = {}

    def accumulate_time(name: str):
        """
        Decorator factory that returns a decorator which accumulates execution time
        into a global dict under the given `name`.
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start = time.perf_counter()
                try:
                    return func(*args, **kwargs)
                finally:
                    duration = time.perf_counter() - start
                    execution_time_accumulator[name] = (
                        execution_time_accumulator.get(name, 0.0) + duration
                    )
            return wrapper
        return decorator
    return (
        accumulate_time,
        analysis,
        cv2,
        datetime,
        execution_time_accumulator,
        np,
        pd,
        pickle,
        plt,
        sqlite3,
        time,
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
def _(accumulate_time, analysis, pickle):


    @accumulate_time("scale_image")
    def scale_image( v_im_arr_squeezed ):
        return analysis.scale_image( v_im_arr_squeezed )


    @accumulate_time("load_pickle")
    def load_pickle( fn ):
        with open( fn, "rb" ) as fp2:
        # v_im = vicar.VicarImage( r.LOCAL_FILENAME )

            v_im = pickle.load( fp2 )

        return v_im
    return load_pickle, scale_image


@app.cell
def _(
    analysis,
    cv2,
    datetime,
    df,
    execution_time_accumulator: dict[str, float],
    load_pickle,
    np,
    plt,
    scale_image,
    time,
):

    start_time = time.time()
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

    # The "I can see all of Saturn" regime
    # idx = 20  # --> 1980-11-01 06:34:08.560000
    # idx = 27  # --> 1980-11-25 20:07:45.080000

    these_df = df[ 
        ( df.TARGET_NAME == "SATURN")
        & ( df.INSTRUMENT_NAME == "IMAGING SCIENCE SUBSYSTEM - NARROW ANGLE" )
        & ( df.START_TIME > datetime.datetime(1980,8,20)  )
        & ~ ( ( df.START_TIME < datetime.datetime(1980,11,25)  ) & ( df.START_TIME > datetime.datetime(1980,11,1)  ) )
    ].sort_values("START_TIME").iloc[::250]

    print( f"these_df.shape is {these_df.shape}" )

    # tried 0.5 to 0.9 and 50 to 500, all sucked:
    # ( cv2.HOUGH_GRADIENT_ALT, 50, 0.6 ),

    blur_widths = [3,5,7]
    # p1 = 100
    # methods = [
    #         ( cv2.HOUGH_GRADIENT, p1, 40 ),
    #         ( cv2.HOUGH_GRADIENT, p1, 42 ),
    #         ( cv2.HOUGH_GRADIENT, p1, 44 ),  # <--- this was the best performer
    #         ( cv2.HOUGH_GRADIENT, p1, 46 ),
    #         ( cv2.HOUGH_GRADIENT, p1, 48 ),
    #     ( cv2.HOUGH_GRADIENT, p1, 50 ),
    #     ]


    p2 = 44
    methods = [
            ( cv2.HOUGH_GRADIENT, 50, p2 ),  # <--- the best performer
            ( cv2.HOUGH_GRADIENT, 75, p2 ),
            ( cv2.HOUGH_GRADIENT, 100, p2 ), 
            ( cv2.HOUGH_GRADIENT, 150, p2 ),
            ( cv2.HOUGH_GRADIENT, 200, p2 ),
        ( cv2.HOUGH_GRADIENT, 400, p2 ),
        ]





    # Sucked, didn't classify most images.
    param1s = [ 50, 100, 200, 400 ]
    #     accumulate_time("hough_circles")( lambda im_prepped_for_cv2 : analysis.find_circle_center_parametrized(
    #             im_prepped_for_cv2,
    #             blur_width=5,
    #             method=cv2.HOUGH_GRADIENT_ALT,
    #             dp=1,
    #             minDist=50,
    #             param1=param1,
    #             param2=0.9,
    #             minRadius=25,
    #             maxRadius=0 
    #         ) ) ) for param1 in param1s 


    # circles_funcs = [
    #     ({"method": str(method)},
    #     accumulate_time("hough_circles")( lambda im_prepped_for_cv2 : analysis.find_circle_center_parametrized(
    #             im_prepped_for_cv2,
    #             blur_width=5,
    #             method=method[0],
    #             dp=1,
    #             minDist=50,
    #             param1=method[1],
    #             param2=method[2],
    #             minRadius=25,
    #             maxRadius=0,
    #             message=f"poop {method}"
    #         ) ) ) for method in methods 
    # ]

    circles_funcs = []

    for method in methods:

        def this_func( im_prepped_for_cv2, method=method ):

            return analysis.find_circle_center_parametrized(
                im_prepped_for_cv2,
                blur_width=5,
                method=method[0],
                dp=1,
                minDist=50,
                param1=method[1],
                param2=method[2],
                minRadius=25,
                maxRadius=0,
                message=f"poop {method}"
            )

        circles_funcs.append( ( {"method": str(method) } , this_func ) )







    # rows, cols
    # how_many_rows = int( np.ceil( these_df.shape[0] / 5 ) )
    # how_tall = 2* how_many_rows
    # fig, axes = plt.subplots( how_many_rows , 5, figsize=(10, how_tall))

    cols = len( circles_funcs )

    how_many_rows = int( np.ceil( these_df.shape[0] ) )
    how_tall = 2 * ( 5 / cols ) * how_many_rows
    fig, axes = plt.subplots( how_many_rows , cols, figsize=(10, how_tall))

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

    frame = 0

    result_dicts = []

    for n3,(i,r) in enumerate( these_df.iterrows() ):

        if n3 > ( len(axes)-1 ):
            break

        v_im = load_pickle( r.LOCAL_IMAGE_PICKLE_FN )

        im = scale_image( v_im.array.squeeze() )

        # im_prepped_for_cv2 = ( im * 255 ).astype( np.uint8 )

        # circles = analysis.find_circle_center( im_prepped_for_cv2 ) 

        dim1, dim2, dim3 = im.shape
        # new_image = np.stack([im*255]*3, axis=-1).reshape(dim1,dim2,3).astype( np.uint8 )

        # for condition,circle_func in circles_funcs:
        for method in methods:

            new_image = np.stack([im*255]*3, axis=-1).reshape(dim1,dim2,3).astype( np.uint8 )
            im_prepped_for_cv2 = ( im * 255 ).astype( np.uint8 )

            circles = analysis.find_circle_center_parametrized(
                im_prepped_for_cv2,
                blur_width=5,
                method=method[0],
                dp=1,
                minDist=50,
                param1=method[1],
                param2=method[2],
                minRadius=25,
                maxRadius=0,
                message=f"the method is {method}"
            )


            # print( condition, circle_func )

            # circles = circle_func( im_prepped_for_cv2 )


            if circles is None:
                # print("none")
                circles = []


            else:
                circles = circles[0]

            result_dicts.append( { "circles":circles , "product_id": r.PRODUCT_ID, "image_seq_num": n3, "method":method } )

            if True:

                # print( type( circles[0][0][0] ) )

                # circles = np.uint16(np.around(circles))

            # print( circles )

        #     x, y, r = circles[0][0]

        #     return int(x), int(y), int(r)

            # print( len( circles ) ) 

                im_annotated = new_image

                for color_index,(x,y,radius) in enumerate( circles ):

                # x,y,radius = circles[0]
                    color_index = min( len(colors_rgb)-1, color_index )
                    im_annotated = cv2.circle(im_annotated, (int(x),int(y)), int(radius) , colors_rgb[ color_index ], 8 )
                    break

                axes[frame].imshow( im_annotated, cmap="gray")
                frame += 1

    for a in axes:
        a.axis("off")  # Hide axes ticks


    plt.tight_layout()
    plt.show()

    end_time = time.time()

    total_time = end_time - start_time 

    print( f"total time is {total_time}")

    print( execution_time_accumulator ) 

    # last sizeable photo is #20
    # next sizable photo is 29

    return circles_funcs, r, result_dicts, these_df


@app.cell
def _(circles_funcs):
    circles_funcs
    return


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
    # result_df = pd.DataFrame( result_dicts )

    # result_df.apply( lambda r : len(r.circles),axis=1 )
    # result_df
    return


@app.cell
def _(result_dicts):
    result_dicts
    return


@app.cell
def _(r):
    r
    return


@app.cell
def _(these_df):
    these_df.iloc[20]
    return


@app.cell
def _(idx, load_pickle, plt, scale_image, these_df):
    # idx = 20  # --> 1980-11-01 06:34:08.560000
    # idx = 27  # --> 1980-11-25 20:07:45.080000

    v_imx= load_pickle( these_df.iloc[ idx ].LOCAL_IMAGE_PICKLE_FN )

    imx = scale_image( v_imx.array.squeeze() )

    print( these_df.iloc[ idx ].START_TIME )

    plt.imshow( imx )


    return


@app.cell
def _():
    s = """<class 'numpy.ndarray'> - 1 - [889.5 229.5 738.7] - (1000, 1000, 1) - 0.016168832778930664
    <class 'numpy.ndarray'> - 1 - [431.5 442.5 163.1] - (1000, 1000, 1) - 0.0185549259185791
    <class 'numpy.ndarray'> - 1 - [888.5 240.5 603.6] - (1000, 1000, 1) - 0.020563840866088867
    <class 'numpy.ndarray'> - 1 - [531.5 519.5 163.1] - (1000, 1000, 1) - 0.0192108154296875
    <class 'numpy.ndarray'> - 1 - [433.5 516.5 144.2] - (1000, 1000, 1) - 0.03411746025085449
    <class 'numpy.ndarray'> - 1 - [562.5 409.5 408.7] - (1000, 1000, 1) - 0.017226457595825195
    <class 'numpy.ndarray'> - 1 - [566.5 407.5 166.7] - (1000, 1000, 1) - 0.016490459442138672
    <class 'numpy.ndarray'> - 1 - [549.5 211.5 424.7] - (1000, 1000, 1) - 0.015997648239135742
    <class 'numpy.ndarray'> - 1 - [429.5 415.5 138.4] - (1000, 1000, 1) - 0.018253326416015625
    <class 'numpy.ndarray'> - 1 - [353.5 754.5 327.5] - (1000, 1000, 1) - 0.018214941024780273
    <class 'numpy.ndarray'> - 1 - [562.5 490.5 168.9] - (1000, 1000, 1) - 0.019804954528808594
    <class 'numpy.ndarray'> - 1 - [465.5 496.5 166.1] - (1000, 1000, 1) - 0.035909414291381836
    <class 'numpy.ndarray'> - 1 - [486.5 381.5 305.2] - (1000, 1000, 1) - 56.7055823802948
    <class 'numpy.ndarray'> - 1 - [404.5 636.5 311.5] - (1000, 1000, 1) - 56.6219367980957
    <class 'numpy.ndarray'> - 1 - [597.5 427.5 373.6] - (1000, 1000, 1) - 57.48758244514465
    <class 'numpy.ndarray'> - 1 - [622.5 508.5 341.1] - (1000, 1000, 1) - 57.838589906692505
    <class 'numpy.ndarray'> - 1 - [443.5 491.5 307.3] - (1000, 1000, 1) - 57.760791063308716
    <class 'numpy.ndarray'> - 1 - [521.5 352.5 286. ] - (1000, 1000, 1) - 56.55821442604065
    <class 'numpy.ndarray'> - 1 - [447.5 525.5 324.9] - (1000, 1000, 1) - 57.34959149360657
    None
    <class 'numpy.ndarray'> - 1 - [216.5 613.5 587.9] - (1000, 1000, 1) - 0.02425217628479004
    <class 'numpy.ndarray'> - 1 - [564.5 442.5 171.5] - (1000, 1000, 1) - 0.0304720401763916
    <class 'numpy.ndarray'> - 1 - [430.5 554.5 463.4] - (1000, 1000, 1) - 0.03158712387084961
    <class 'numpy.ndarray'> - 1 - [532.5 456.5 166.5] - (1000, 1000, 1) - 0.025400400161743164
    <class 'numpy.ndarray'> - 1 - [507.5 547.5 465.6] - (1000, 1000, 1) - 0.019522905349731445
    <class 'numpy.ndarray'> - 1 - [381.5 414.5 202.9] - (1000, 1000, 1) - 0.019581079483032227
    <class 'numpy.ndarray'> - 1 - [506.5 447.5 169.8] - (1000, 1000, 1) - 0.020772457122802734
    <class 'numpy.ndarray'> - 1 - [783.5 490.5 476.7] - (1000, 1000, 1) - 0.019985437393188477
    <class 'numpy.ndarray'> - 1 - [431.5 438.5 168.8] - (1000, 1000, 1) - 0.026442289352416992
    <class 'numpy.ndarray'> - 1 - [470.5 151.5 450.5] - (1000, 1000, 1) - 0.02309584617614746
    <class 'numpy.ndarray'> - 1 - [552.5 497.5 170.6] - (1000, 1000, 1) - 0.031771183013916016
    <class 'numpy.ndarray'> - 1 - [439.5 499.5 139.1] - (1000, 1000, 1) - 0.05819892883300781
    <class 'numpy.ndarray'> - 1 - [272.5 728.5 701.4] - (1000, 1000, 1) - 0.01917099952697754
    <class 'numpy.ndarray'> - 1 - [530.5 420.5 140.5] - (1000, 1000, 1) - 0.023336410522460938
    <class 'numpy.ndarray'> - 1 - [443.5 421.5 141.1] - (1000, 1000, 1) - 0.023305416107177734
    <class 'numpy.ndarray'> - 1 - [972.5 146.5 812.8] - (1000, 1000, 1) - 0.023595809936523438
    <class 'numpy.ndarray'> - 1 - [530.5 532.5 166.4] - (1000, 1000, 1) - 0.02501201629638672
    <class 'numpy.ndarray'> - 1 - [459.5 522.5 148.2] - (1000, 1000, 1) - 0.06373977661132812
    <class 'numpy.ndarray'> - 1 - [723.5 339.5  58.4] - (1000, 1000, 1) - 0.010227203369140625
    <class 'numpy.ndarray'> - 1 - [642.5 454.5  56.8] - (1000, 1000, 1) - 0.010894775390625
    <class 'numpy.ndarray'> - 1 - [669.5 401.5  57.4] - (1000, 1000, 1) - 0.019199609756469727
    <class 'numpy.ndarray'> - 1 - [603.5 487.5  58.5] - (1000, 1000, 1) - 0.015010356903076172
    <class 'numpy.ndarray'> - 1 - [503.5 495.5 447.8] - (1000, 1000, 1) - 33.98907685279846
    <class 'numpy.ndarray'> - 1 - [602.5 379.5  57.9] - (1000, 1000, 1) - 0.013779640197753906
    <class 'numpy.ndarray'> - 1 - [411.5 456.5 384.4] - (1000, 1000, 1) - 57.06262183189392
    <class 'numpy.ndarray'> - 1 - [656.5 621.5 313.7] - (1000, 1000, 1) - 55.56410837173462
    <class 'numpy.ndarray'> - 1 - [487.5 512.5 451.5] - (1000, 1000, 1) - 23.3725323677063
    <class 'numpy.ndarray'> - 1 - [439.5 428.5 377.5] - (1000, 1000, 1) - 0.0195462703704834
    <class 'numpy.ndarray'> - 1 - [513.5 433.5 169.2] - (1000, 1000, 1) - 0.02216196060180664
    <class 'numpy.ndarray'> - 1 - [573.5 191.5 554.3] - (1000, 1000, 1) - 0.02216482162475586
    <class 'numpy.ndarray'> - 1 - [448.5 460.5 169.5] - (1000, 1000, 1) - 0.027209997177124023
    <class 'numpy.ndarray'> - 1 - [712.5 180.5 584.6] - (1000, 1000, 1) - 0.02303314208984375
    <class 'numpy.ndarray'> - 1 - [530.5 531.5 170.8] - (1000, 1000, 1) - 0.026809215545654297
    <class 'numpy.ndarray'> - 1 - [128.5 896.5  94.3] - (1000, 1000, 1) - 0.019939422607421875
    <class 'numpy.ndarray'> - 1 - [454.5 507.5 150.4] - (1000, 1000, 1) - 0.07731246948242188
    <class 'numpy.ndarray'> - 1 - [284.5 730.5 702.9] - (1000, 1000, 1) - 0.024047136306762695
    <class 'numpy.ndarray'> - 1 - [532.5 445.5 163.6] - (1000, 1000, 1) - 0.024147748947143555
    <class 'numpy.ndarray'> - 1 - [647.5 266.5 395.6] - (1000, 1000, 1) - 0.023926973342895508
    <class 'numpy.ndarray'> - 1 - [460.5 450.5 175.7] - (1000, 1000, 1) - 0.027922391891479492
    <class 'numpy.ndarray'> - 1 - [954.5 225.5 649.2] - (1000, 1000, 1) - 0.035135746002197266
    <class 'numpy.ndarray'> - 1 - [554.5 512.5 150.9] - (1000, 1000, 1) - 0.02677130699157715
    <class 'numpy.ndarray'> - 1 - [474.5 495.5 174.8] - (1000, 1000, 1) - 0.07140707969665527
    <class 'numpy.ndarray'> - 1 - [522.5 586.5 381.7] - (1000, 1000, 1) - 22.97806406021118
    <class 'numpy.ndarray'> - 1 - [543.5 467.5 387.2] - (1000, 1000, 1) - 52.4951286315918
    <class 'numpy.ndarray'> - 1 - [483.5 555.5 340.5] - (1000, 1000, 1) - 50.104509115219116
    <class 'numpy.ndarray'> - 1 - [575.5 552.5 306.4] - (1000, 1000, 1) - 52.8363139629364
    <class 'numpy.ndarray'> - 1 - [385.5 521.5 323.6] - (1000, 1000, 1) - 56.16871881484985
    <class 'numpy.ndarray'> - 1 - [533.5 502.5 345.5] - (1000, 1000, 1) - 50.59341382980347
    <class 'numpy.ndarray'> - 1 - [513.5 603.5 357.5] - (1000, 1000, 1) - 56.06876730918884
    <class 'numpy.ndarray'> - 1 - [467.5 437.5 296. ] - (1000, 1000, 1) - 55.20439910888672
    <class 'numpy.ndarray'> - 1 - [412.5 507.5 339.1] - (1000, 1000, 1) - 55.57266139984131
    <class 'numpy.ndarray'> - 1 - [328.5 659.5 265.6] - (1000, 1000, 1) - 58.41656470298767
    <class 'numpy.ndarray'> - 1 - [175.5 159.5 156.3] - (1000, 1000, 1) - 0.06219148635864258
    <class 'numpy.ndarray'> - 1 - [565.5 736.5 407.5] - (1000, 1000, 1) - 0.026118040084838867
    <class 'numpy.ndarray'> - 1 - [515.5 428.5 373.6] - (1000, 1000, 1) - 56.66858696937561
    <class 'numpy.ndarray'> - 1 - [766.5 127.5 839.9] - (1000, 1000, 1) - 0.024492263793945312
    <class 'numpy.ndarray'> - 1 - [421.5 449.5 171.1] - (1000, 1000, 1) - 0.0421299934387207
    <class 'numpy.ndarray'> - 1 - [963.5 233.5 761.9] - (1000, 1000, 1) - 0.034564971923828125
    <class 'numpy.ndarray'> - 1 - [522.5 524.5 171.4] - (1000, 1000, 1) - 0.03975105285644531
    <class 'numpy.ndarray'> - 1 - [469.5 489.5 177.6] - (1000, 1000, 1) - 0.07846283912658691
    <class 'numpy.ndarray'> - 1 - [462.5 559.5 480.4] - (1000, 1000, 1) - 0.02989053726196289
    <class 'numpy.ndarray'> - 1 - [519.5 436.5 174. ] - (1000, 1000, 1) - 0.026688575744628906
    <class 'numpy.ndarray'> - 1 - [845.5 320.5 580.7] - (1000, 1000, 1) - 0.02797698974609375
    <class 'numpy.ndarray'> - 1 - [438.5 432.5 173.1] - (1000, 1000, 1) - 0.035347938537597656
    <class 'numpy.ndarray'> - 1 - [873.5 267.5 701.8] - (1000, 1000, 1) - 0.031006813049316406
    <class 'numpy.ndarray'> - 1 - [514.5 527.5 174.8] - (1000, 1000, 1) - 0.03584456443786621
    <class 'numpy.ndarray'> - 1 - [482.5 528.5 464.9] - (1000, 1000, 1) - 0.02292656898498535
    <class 'numpy.ndarray'> - 1 - [443.5 517.5 171.5] - (1000, 1000, 1) - 0.07067275047302246
    <class 'numpy.ndarray'> - 1 - [682.5 364.5 288.2] - (1000, 1000, 1) - 56.91052794456482
    <class 'numpy.ndarray'> - 1 - [810.5 712.5  59.8] - (1000, 1000, 1) - 0.013175725936889648
    <class 'numpy.ndarray'> - 1 - [479.5 542.5 310. ] - (1000, 1000, 1) - 54.76618480682373
    <class 'numpy.ndarray'> - 1 - [484.5 485.5 464.4] - (1000, 1000, 1) - 0.31212663650512695
    <class 'numpy.ndarray'> - 1 - [450.5 548.5 175.2] - (1000, 1000, 1) - 0.02243947982788086
    <class 'numpy.ndarray'> - 1 - [442.5 557.5 175.6] - (1000, 1000, 1) - 0.02804112434387207
    <class 'numpy.ndarray'> - 1 - [290.5 367.5 377.1] - (1000, 1000, 1) - 0.02470564842224121
    <class 'numpy.ndarray'> - 1 - [219.5 516.5 490.4] - (1000, 1000, 1) - 0.025030136108398438
    <class 'numpy.ndarray'> - 1 - [595.5 481.5 355.3] - (1000, 1000, 1) - 56.51109075546265
    <class 'numpy.ndarray'> - 1 - [816.5 186.5 781.5] - (1000, 1000, 1) - 0.022969722747802734
    <class 'numpy.ndarray'> - 1 - [473.5 435.5 179.2] - (1000, 1000, 1) - 0.026589393615722656
    <class 'numpy.ndarray'> - 1 - [986.5  12.5 936.7] - (1000, 1000, 1) - 0.047829389572143555
    <class 'numpy.ndarray'> - 1 - [529.5 534.5 177.8] - (1000, 1000, 1) - 0.02710700035095215
    <class 'numpy.ndarray'> - 1 - [440.5 515.5 157.2] - (1000, 1000, 1) - 0.06528663635253906
    <class 'numpy.ndarray'> - 1 - [493.5 512.5 490.4] - (1000, 1000, 1) - 0.04027056694030762
    <class 'numpy.ndarray'> - 1 - [565.5 401.5 183.6] - (1000, 1000, 1) - 0.0217437744140625
    <class 'numpy.ndarray'> - 1 - [854.5 111.5 856.6] - (1000, 1000, 1) - 0.14703083038330078
    <class 'numpy.ndarray'> - 1 - [436.5 461.5 180.8] - (1000, 1000, 1) - 0.026646852493286133
    <class 'numpy.ndarray'> - 1 - [985.5  60.5 852.5] - (1000, 1000, 1) - 0.15927910804748535
    <class 'numpy.ndarray'> - 1 - [536.5 530.5 177.2] - (1000, 1000, 1) - 0.029002666473388672
    <class 'numpy.ndarray'> - 1 - [440.5 518.5 156.8] - (1000, 1000, 1) - 0.06671309471130371
    <class 'numpy.ndarray'> - 1 - [551.5 355.5 301.8] - (1000, 1000, 1) - 59.99875330924988
    <class 'numpy.ndarray'> - 1 - [469.5 494.5 311.5] - (1000, 1000, 1) - 57.4265661239624
    <class 'numpy.ndarray'> - 1 - [355.5 446.5 311.5] - (1000, 1000, 1) - 58.307326793670654
    <class 'numpy.ndarray'> - 1 - [456.5 456.5 366. ] - (1000, 1000, 1) - 56.21872401237488
    <class 'numpy.ndarray'> - 1 - [617.5 426.5 310.1] - (1000, 1000, 1) - 51.47094655036926
    <class 'numpy.ndarray'> - 1 - [604.5 590.5 348.6] - (1000, 1000, 1) - 51.83330845832825
    <class 'numpy.ndarray'> - 1 - [395.5 474.5 290.9] - (1000, 1000, 1) - 52.22780704498291
    <class 'numpy.ndarray'> - 1 - [583.5 602.5 306.5] - (1000, 1000, 1) - 54.983378887176514
    <class 'numpy.ndarray'> - 1 - [620.5 397.5 311.5] - (1000, 1000, 1) - 52.96324014663696
    <class 'numpy.ndarray'> - 1 - [579.5 361.5 320.5] - (1000, 1000, 1) - 54.29756999015808
    <class 'numpy.ndarray'> - 1 - [682.5 529.5 288.2] - (1000, 1000, 1) - 51.90610694885254
    <class 'numpy.ndarray'> - 1 - [520.5 467.5 305.2] - (1000, 1000, 1) - 52.36101794242859
    <class 'numpy.ndarray'> - 1 - [575.5 454.5 288.2] - (1000, 1000, 1) - 52.34847068786621
    <class 'numpy.ndarray'> - 1 - [462.5 579.5 324.9] - (1000, 1000, 1) - 51.71824049949646
    <class 'numpy.ndarray'> - 1 - [608.5 395.5 325. ] - (1000, 1000, 1) - 47.6452898979187
    <class 'numpy.ndarray'> - 1 - [155.5 786.5 752.1] - (1000, 1000, 1) - 0.022109270095825195
    <class 'numpy.ndarray'> - 1 - [515.5 503.5 426.8] - (1000, 1000, 1) - 52.177523612976074
    <class 'numpy.ndarray'> - 1 - [473.5 321.5 450. ] - (1000, 1000, 1) - 0.02317333221435547
    <class 'numpy.ndarray'> - 1 - [441.5 433.5 151.8] - (1000, 1000, 1) - 0.03624129295349121
    <class 'numpy.ndarray'> - 1 - [696.5 118.5 677.7] - (1000, 1000, 1) - 0.030704736709594727
    <class 'numpy.ndarray'> - 1 - [539.5 542.5 178.3] - (1000, 1000, 1) - 0.03076934814453125
    <class 'numpy.ndarray'> - 1 - [451.5 534.5 159.2] - (1000, 1000, 1) - 0.0583798885345459
    <class 'numpy.ndarray'> - 1 - [495.5 521.5 492.6] - (1000, 1000, 1) - 0.02848649024963379
    <class 'numpy.ndarray'> - 1 - [577.5 414.5 184.3] - (1000, 1000, 1) - 0.023358583450317383
    <class 'numpy.ndarray'> - 1 - [628.5 215.5 615.4] - (1000, 1000, 1) - 0.02928948402404785
    <class 'numpy.ndarray'> - 1 - [461.5 416.5 185.6] - (1000, 1000, 1) - 0.14775657653808594
    <class 'numpy.ndarray'> - 1 - [897.5 458.5 442.7] - (1000, 1000, 1) - 0.045281171798706055
    <class 'numpy.ndarray'> - 1 - [565.5 479.5 187. ] - (1000, 1000, 1) - 0.031316518783569336
    <class 'numpy.ndarray'> - 1 - [217.5 783.5 194.2] - (1000, 1000, 1) - 0.018024206161499023
    <class 'numpy.ndarray'> - 1 - [450.5 509.5 159.4] - (1000, 1000, 1) - 0.06600475311279297
    <class 'numpy.ndarray'> - 1 - [505.5 530.5 311.5] - (1000, 1000, 1) - 57.52811884880066
    <class 'numpy.ndarray'> - 1 - [727.5 352.5  63.2] - (1000, 1000, 1) - 0.014778614044189453
    <class 'numpy.ndarray'> - 1 - [601.5 537.5  62.6] - (1000, 1000, 1) - 0.01979994773864746
    <class 'numpy.ndarray'> - 1 - [678.5 386.5  61.3] - (1000, 1000, 1) - 0.02402520179748535
    <class 'numpy.ndarray'> - 1 - [569.5 491.5  63.7] - (1000, 1000, 1) - 0.015285968780517578
    <class 'numpy.ndarray'> - 1 - [514.5 483.5 455.2] - (1000, 1000, 1) - 21.770065784454346
    <class 'numpy.ndarray'> - 1 - [544.5 389.5  63.3] - (1000, 1000, 1) - 0.02803802490234375
    <class 'numpy.ndarray'> - 1 - [470.5 435.5 305.2] - (1000, 1000, 1) - 57.98974871635437
    <class 'numpy.ndarray'> - 1 - [618.5 488.5 265.4] - (1000, 1000, 1) - 56.84864139556885
    <class 'numpy.ndarray'> - 1 - [499.5 512.5 452.3] - (1000, 1000, 1) - 26.609132051467896
    <class 'numpy.ndarray'> - 1 - [210.5 817.5 790.6] - (1000, 1000, 1) - 0.02272772789001465
    <class 'numpy.ndarray'> - 1 - [578.5 432.5 190.5] - (1000, 1000, 1) - 0.02502727508544922
    <class 'numpy.ndarray'> - 1 - [812.5 182.5 664.1] - (1000, 1000, 1) - 0.022728681564331055
    <class 'numpy.ndarray'> - 1 - [455.5 454.5 183.9] - (1000, 1000, 1) - 0.03545951843261719
    <class 'numpy.ndarray'> - 1 - [800.5  76.5 779.8] - (1000, 1000, 1) - 0.026543140411376953
    <class 'numpy.ndarray'> - 1 - [529.5 534.5 183.6] - (1000, 1000, 1) - 0.03752589225769043
    <class 'numpy.ndarray'> - 1 - [475.5 505.5 162.7] - (1000, 1000, 1) - 0.0863335132598877
    <class 'numpy.ndarray'> - 1 - [506.5 514.5 482.8] - (1000, 1000, 1) - 0.03046560287475586
    <class 'numpy.ndarray'> - 1 - [556.5 453.5 185.8] - (1000, 1000, 1) - 0.024005889892578125
    <class 'numpy.ndarray'> - 1 - [995.5  30.5 940. ] - (1000, 1000, 1) - 0.08904457092285156
    <class 'numpy.ndarray'> - 1 - [454.5 460.5 184.2] - (1000, 1000, 1) - 0.027634143829345703
    <class 'numpy.ndarray'> - 1 - [455.5 893.5 436.7] - (1000, 1000, 1) - 0.0726008415222168
    <class 'numpy.ndarray'> - 1 - [555.5 524.5 184.2] - (1000, 1000, 1) - 0.03193092346191406
    <class 'numpy.ndarray'> - 1 - [857.5  21.5 684.1] - (1000, 1000, 1) - 0.08397531509399414
    <class 'numpy.ndarray'> - 1 - [763.5 345.5  65.9] - (1000, 1000, 1) - 0.011400699615478516
    <class 'numpy.ndarray'> - 1 - [710.5 399.5  65.9] - (1000, 1000, 1) - 0.011486291885375977
    <class 'numpy.ndarray'> - 1 - [644.5 447.5  66.5] - (1000, 1000, 1) - 0.012824296951293945
    <class 'numpy.ndarray'> - 1 - [567.5 499.5  65.6] - (1000, 1000, 1) - 0.012915849685668945
    <class 'numpy.ndarray'> - 1 - [489.5 494.5 465.7] - (1000, 1000, 1) - 18.57476830482483
    <class 'numpy.ndarray'> - 1 - [465.5 419.5  65.2] - (1000, 1000, 1) - 0.023070096969604492
    <class 'numpy.ndarray'> - 1 - [496.5 433.5 271.2] - (1000, 1000, 1) - 59.383594036102295
    <class 'numpy.ndarray'> - 1 - [621.5 395.5 343.9] - (1000, 1000, 1) - 47.94399046897888
    <class 'numpy.ndarray'> - 1 - [440.5 409.5 385.8] - (1000, 1000, 1) - 49.98736023902893
    <class 'numpy.ndarray'> - 1 - [434.5 569.5 189.1] - (1000, 1000, 1) - 0.040372371673583984
    <class 'numpy.ndarray'> - 1 - [550.5 465.5 352.4] - (1000, 1000, 1) - 47.92099189758301
    <class 'numpy.ndarray'> - 1 - [810.5 234.5  66.5] - (1000, 1000, 1) - 0.012993574142456055
    <class 'numpy.ndarray'> - 1 - [439.5 580.5 191.2] - (1000, 1000, 1) - 0.06789350509643555
    <class 'numpy.ndarray'> - 1 - [435.5 577.5 189.5] - (1000, 1000, 1) - 0.03639650344848633
    <class 'numpy.ndarray'> - 1 - [422.5 537.5 154. ] - (1000, 1000, 1) - 0.07575368881225586
    <class 'numpy.ndarray'> - 1 - [432.5 560.5 189.8] - (1000, 1000, 1) - 0.042021751403808594
    <class 'numpy.ndarray'> - 1 - [317.5 603.5 292.6] - (1000, 1000, 1) - 0.0377955436706543
    <class 'numpy.ndarray'> - 1 - [422.5 550.5 188.6] - (1000, 1000, 1) - 0.7324666976928711
    <class 'numpy.ndarray'> - 1 - [466.5 521.5 195.6] - (1000, 1000, 1) - 0.03136801719665527
    <class 'numpy.ndarray'> - 1 - [432.5 572.5 190.3] - (1000, 1000, 1) - 0.11814069747924805
    <class 'numpy.ndarray'> - 1 - [479.5 510.5 200.6] - (1000, 1000, 1) - 0.03920793533325195
    <class 'numpy.ndarray'> - 1 - [448.5 486.5 323.6] - (1000, 1000, 1) - 44.545201778411865
    <class 'numpy.ndarray'> - 1 - [620.5 305.5  68.8] - (1000, 1000, 1) - 0.01354360580444336
    <class 'numpy.ndarray'> - 1 - [507.5 653.5 312.9] - (1000, 1000, 1) - 59.200931549072266
    <class 'numpy.ndarray'> - 1 - [606.5 675.5 292.3] - (1000, 1000, 1) - 58.88908934593201
    <class 'numpy.ndarray'> - 1 - [156.5 182.5 135.9] - (1000, 1000, 1) - 0.015485286712646484
    <class 'numpy.ndarray'> - 1 - [452.5 521.5 381.6] - (1000, 1000, 1) - 55.81273055076599
    <class 'numpy.ndarray'> - 1 - [377.5 449.5 297.2] - (1000, 1000, 1) - 61.98069500923157
    <class 'numpy.ndarray'> - 1 - [611.5 449.5 287.4] - (1000, 1000, 1) - 59.30723237991333
    <class 'numpy.ndarray'> - 1 - [623.5 469.5 311.5] - (1000, 1000, 1) - 56.47691226005554
    <class 'numpy.ndarray'> - 1 - [454.5 505.5 332. ] - (1000, 1000, 1) - 57.40801763534546
    <class 'numpy.ndarray'> - 1 - [495.5 428.5 286. ] - (1000, 1000, 1) - 56.64824891090393
    <class 'numpy.ndarray'> - 1 - [613.5 813.5  58.4] - (1000, 1000, 1) - 0.0156857967376709
    <class 'numpy.ndarray'> - 1 - [827.5 137.5  50.9] - (1000, 1000, 1) - 0.014856576919555664
    <class 'numpy.ndarray'> - 1 - [205.5 274.5  57.5] - (1000, 1000, 1) - 0.012290477752685547
    <class 'numpy.ndarray'> - 1 - [401.5 536.5 304.5] - (1000, 1000, 1) - 55.06957697868347
    <class 'numpy.ndarray'> - 1 - [435.5 645.5 320.9] - (1000, 1000, 1) - 56.64704251289368
    <class 'numpy.ndarray'> - 1 - [313.5 508.5 288.2] - (1000, 1000, 1) - 52.41711068153381
    <class 'numpy.ndarray'> - 1 - [519.5 559.5 358.9] - (1000, 1000, 1) - 50.712486267089844
    <class 'numpy.ndarray'> - 1 - [465.5 539.5 309.7] - (1000, 1000, 1) - 52.26203274726868
    <class 'numpy.ndarray'> - 1 - [536.5 552.5 311.5] - (1000, 1000, 1) - 53.55046224594116
    <class 'numpy.ndarray'> - 1 - [336.5 556.5 311.5] - (1000, 1000, 1) - 52.397141218185425
    <class 'numpy.ndarray'> - 1 - [459.5 516.5 322.8] - (1000, 1000, 1) - 51.965397357940674
    <class 'numpy.ndarray'> - 1 - [563.5 566.5 295.8] - (1000, 1000, 1) - 57.14696478843689
    <class 'numpy.ndarray'> - 1 - [459.5 498.5 324.9] - (1000, 1000, 1) - 54.77811121940613
    <class 'numpy.ndarray'> - 1 - [563.5 435.5 408.3] - (1000, 1000, 1) - 0.030836820602416992
    <class 'numpy.ndarray'> - 1 - [604.5 527.5 206.3] - (1000, 1000, 1) - 0.04432272911071777
    <class 'numpy.ndarray'> - 1 - [432.5 451.5 421.9] - (1000, 1000, 1) - 0.02403998374938965
    <class 'numpy.ndarray'> - 1 - [487.5 522.5 166.4] - (1000, 1000, 1) - 0.03341507911682129
    <class 'numpy.ndarray'> - 1 - [528.5 485.5 482.8] - (1000, 1000, 1) - 0.04673337936401367
    <class 'numpy.ndarray'> - 1 - [623.5 592.5 202.9] - (1000, 1000, 1) - 0.037108659744262695
    <class 'numpy.ndarray'> - 1 - [524.5 603.5 202.4] - (1000, 1000, 1) - 0.10330080986022949
    <class 'numpy.ndarray'> - 1 - [899.5   5.5 888. ] - (1000, 1000, 1) - 1.2277226448059082
    <class 'numpy.ndarray'> - 1 - [439.5 383.5 204.8] - (1000, 1000, 1) - 0.03068852424621582
    <class 'numpy.ndarray'> - 1 - [301.5 435.5 202.4] - (1000, 1000, 1) - 0.026761770248413086
    <class 'numpy.ndarray'> - 1 - [404.5 510.5 198.5] - (1000, 1000, 1) - 0.036835432052612305
    <class 'numpy.ndarray'> - 1 - [330.5 445.5 206. ] - (1000, 1000, 1) - 0.027587175369262695
    <class 'numpy.ndarray'> - 1 - [643.5 287.5  74.9] - (1000, 1000, 1) - 0.01343536376953125
    <class 'numpy.ndarray'> - 1 - [581.5 455.5  74.3] - (1000, 1000, 1) - 0.01354670524597168
    <class 'numpy.ndarray'> - 1 - [659.5 355.5  74.5] - (1000, 1000, 1) - 0.013075590133666992
    <class 'numpy.ndarray'> - 1 - [637.5 379.5  73.8] - (1000, 1000, 1) - 0.013816595077514648
    <class 'numpy.ndarray'> - 1 - [590.5 585.5 380.9] - (1000, 1000, 1) - 12.186649799346924
    <class 'numpy.ndarray'> - 1 - [700.5 294.5  75.9] - (1000, 1000, 1) - 0.013877630233764648
    <class 'numpy.ndarray'> - 1 - [474.5 475.5 358. ] - (1000, 1000, 1) - 57.795464754104614
    <class 'numpy.ndarray'> - 1 - [567.5 559.5 339.2] - (1000, 1000, 1) - 53.71345329284668
    <class 'numpy.ndarray'> - 1 - [537.5 590.5 324.9] - (1000, 1000, 1) - 54.272087812423706
    <class 'numpy.ndarray'> - 1 - [408.5 432.5 288.2] - (1000, 1000, 1) - 54.03179574012756
    <class 'numpy.ndarray'> - 1 - [499.5 549.5 394.2] - (1000, 1000, 1) - 55.88441061973572
    <class 'numpy.ndarray'> - 1 - [605.5 376.5 341.9] - (1000, 1000, 1) - 59.294281244277954
    <class 'numpy.ndarray'> - 1 - [401.5 461.5 273.4] - (1000, 1000, 1) - 57.52807307243347
    <class 'numpy.ndarray'> - 1 - [434.5 557.5 334.7] - (1000, 1000, 1) - 54.34955954551697
    <class 'numpy.ndarray'> - 1 - [645.5 520.5 311. ] - (1000, 1000, 1) - 53.408681869506836
    <class 'numpy.ndarray'> - 1 - [630.5 423.5 323.5] - (1000, 1000, 1) - 53.50788140296936
    <class 'numpy.ndarray'> - 1 - [540.5 623.5 288.8] - (1000, 1000, 1) - 53.4837760925293
    <class 'numpy.ndarray'> - 1 - [552.5  29.5  61.3] - (1000, 1000, 1) - 0.01313924789428711
    <class 'numpy.ndarray'> - 1 - [975.5 390.5 578.3] - (1000, 1000, 1) - 0.025887489318847656
    <class 'numpy.ndarray'> - 1 - [433.5 433.5 205. ] - (1000, 1000, 1) - 0.03818011283874512
    <class 'numpy.ndarray'> - 1 - [442.5 494.5 473.6] - (1000, 1000, 1) - 0.016605854034423828
    <class 'numpy.ndarray'> - 1 - [333.5 427.5 195.3] - (1000, 1000, 1) - 0.0321347713470459
    <class 'numpy.ndarray'> - 1 - [453.5 478.5 208.1] - (1000, 1000, 1) - 0.03302001953125
    <class 'numpy.ndarray'> - 1 - [347.5 489.5 180.9] - (1000, 1000, 1) - 0.08790373802185059
    <class 'numpy.ndarray'> - 1 - [808.5  98.5 790.7] - (1000, 1000, 1) - 0.28476715087890625
    <class 'numpy.ndarray'> - 1 - [611.5 520.5 213.5] - (1000, 1000, 1) - 0.03004932403564453
    <class 'numpy.ndarray'> - 1 - [517.5 412.5 557. ] - (1000, 1000, 1) - 0.018262147903442383
    <class 'numpy.ndarray'> - 1 - [484.5 556.5 202.8] - (1000, 1000, 1) - 0.027700424194335938
    <class 'numpy.ndarray'> - 1 - [266.5  29.5 267.2] - (1000, 1000, 1) - 0.01931929588317871
    <class 'numpy.ndarray'> - 1 - [576.5 636.5 207.4] - (1000, 1000, 1) - 0.03546643257141113
    <class 'numpy.ndarray'> - 1 - [520.5 589.5 208.6] - (1000, 1000, 1) - 0.03601813316345215
    <class 'numpy.ndarray'> - 1 - [291.5  56.5  73.2] - (1000, 1000, 1) - 0.014656305313110352
    <class 'numpy.ndarray'> - 1 - [577.5 417.5 206.6] - (1000, 1000, 1) - 0.0305478572845459
    <class 'numpy.ndarray'> - 1 - [572.5 417.5 205.5] - (1000, 1000, 1) - 0.02998065948486328
    <class 'numpy.ndarray'> - 1 - [611.5 373.5 211. ] - (1000, 1000, 1) - 0.02584099769592285
    <class 'numpy.ndarray'> - 1 - [559.5 417.5 206.9] - (1000, 1000, 1) - 0.03819704055786133
    <class 'numpy.ndarray'> - 1 - [612.5 365.5 212.1] - (1000, 1000, 1) - 0.03682661056518555
    <class 'numpy.ndarray'> - 1 - [570.5 398.5 204.8] - (1000, 1000, 1) - 0.035253047943115234
    <class 'numpy.ndarray'> - 1 - [552.5 405.5 205.1] - (1000, 1000, 1) - 0.032732248306274414
    <class 'numpy.ndarray'> - 1 - [566.5 391.5 209.6] - (1000, 1000, 1) - 0.029366493225097656
    <class 'numpy.ndarray'> - 1 - [548.5 400.5 209.5] - (1000, 1000, 1) - 0.028014183044433594
    <class 'numpy.ndarray'> - 1 - [600.5 355.5 212.3] - (1000, 1000, 1) - 0.03273415565490723
    <class 'numpy.ndarray'> - 1 - [545.5 402.5 207.7] - (1000, 1000, 1) - 0.030631303787231445
    <class 'numpy.ndarray'> - 1 - [340.5 558.5 295.3] - (1000, 1000, 1) - 55.74726104736328
    <class 'numpy.ndarray'> - 1 - [667.5 612.5 308. ] - (1000, 1000, 1) - 53.23933482170105
    <class 'numpy.ndarray'> - 1 - [767.5 122.5 748.9] - (1000, 1000, 1) - 0.1342782974243164
    <class 'numpy.ndarray'> - 1 - [611.5 588.5 209. ] - (1000, 1000, 1) - 0.03591299057006836
    <class 'numpy.ndarray'> - 1 - [526.5 556.5 211.5] - (1000, 1000, 1) - 0.02966451644897461
    <class 'numpy.ndarray'> - 1 - [592.5 663.5 208.3] - (1000, 1000, 1) - 0.03002786636352539
    <class 'numpy.ndarray'> - 1 - [516.5 635.5 185. ] - (1000, 1000, 1) - 0.0969243049621582
    <class 'numpy.ndarray'> - 1 - [829.5 100.5 812.5] - (1000, 1000, 1) - 0.8487277030944824
    <class 'numpy.ndarray'> - 1 - [423.5 383.5 213.9] - (1000, 1000, 1) - 0.02947688102722168
    <class 'numpy.ndarray'> - 1 - [409.5 498.5 470.9] - (1000, 1000, 1) - 0.21541905403137207
    <class 'numpy.ndarray'> - 1 - [312.5 415.5 214. ] - (1000, 1000, 1) - 0.024572372436523438
    <class 'numpy.ndarray'> - 1 - [506.5 510.5 483. ] - (1000, 1000, 1) - 0.02805042266845703
    <class 'numpy.ndarray'> - 1 - [440.5 451.5 213.9] - (1000, 1000, 1) - 0.03511953353881836
    <class 'numpy.ndarray'> - 1 - [336.5 463.5 212.7] - (1000, 1000, 1) - 0.03721785545349121
    <class 'numpy.ndarray'> - 1 - [667.5 442.5  77.7] - (1000, 1000, 1) - 0.018876314163208008
    <class 'numpy.ndarray'> - 1 - [649.5 439.5  80.4] - (1000, 1000, 1) - 0.018026351928710938
    <class 'numpy.ndarray'> - 1 - [652.5 488.5  78.9] - (1000, 1000, 1) - 0.016166210174560547
    <class 'numpy.ndarray'> - 1 - [703.5 427.5  79.2] - (1000, 1000, 1) - 0.01594257354736328
    <class 'numpy.ndarray'> - 1 - [494.5 503.5 475.9] - (1000, 1000, 1) - 6.7182934284210205
    <class 'numpy.ndarray'> - 1 - [554.5 441.5  80.7] - (1000, 1000, 1) - 0.019139766693115234
    <class 'numpy.ndarray'> - 1 - [549.5 587.5 344.1] - (1000, 1000, 1) - 55.50206899642944
    <class 'numpy.ndarray'> - 1 - [452.5 554.5 332. ] - (1000, 1000, 1) - 55.31743931770325
    <class 'numpy.ndarray'> - 1 - [552.5 451.5 384.4] - (1000, 1000, 1) - 52.93966817855835
    <class 'numpy.ndarray'> - 1 - [372.5 524.5 336.9] - (1000, 1000, 1) - 52.24394607543945
    <class 'numpy.ndarray'> - 1 - [474.5 575.5 386.6] - (1000, 1000, 1) - 56.27398347854614
    <class 'numpy.ndarray'> - 1 - [322.5 428.5 261.5] - (1000, 1000, 1) - 56.38656044006348
    <class 'numpy.ndarray'> - 1 - [619.5 535.5 324.9] - (1000, 1000, 1) - 56.61888599395752
    <class 'numpy.ndarray'> - 1 - [482.5 413.5 351.3] - (1000, 1000, 1) - 49.48043131828308
    <class 'numpy.ndarray'> - 1 - [626.5 600.5 344.1] - (1000, 1000, 1) - 46.943273067474365
    <class 'numpy.ndarray'> - 1 - [365.5 461.5 333.5] - (1000, 1000, 1) - 45.89050650596619
    <class 'numpy.ndarray'> - 1 - [509.5 482.5 433.3] - (1000, 1000, 1) - 30.465527296066284
    <class 'numpy.ndarray'> - 1 - [849.5 228.5 741.1] - (1000, 1000, 1) - 0.02493882179260254
    <class 'numpy.ndarray'> - 1 - [415.5 455.5 217.7] - (1000, 1000, 1) - 0.03599286079406738
    <class 'numpy.ndarray'> - 1 - [316.5 444.5 215.1] - (1000, 1000, 1) - 0.02917933464050293
    <class 'numpy.ndarray'> - 1 - [122.5 891.5  88.3] - (1000, 1000, 1) - 0.020218849182128906
    <class 'numpy.ndarray'> - 1 - [452.5 457.5 219.9] - (1000, 1000, 1) - 0.03384065628051758
    <class 'numpy.ndarray'> - 1 - [355.5 453.5 220.8] - (1000, 1000, 1) - 0.08139204978942871
    <class 'numpy.ndarray'> - 1 - [394.5 281.5 371. ] - (1000, 1000, 1) - 0.023867130279541016
    <class 'numpy.ndarray'> - 1 - [571.5 561.5 215.6] - (1000, 1000, 1) - 0.03637504577636719
    <class 'numpy.ndarray'> - 1 - [625.5 137.5 829.9] - (1000, 1000, 1) - 0.025828123092651367
    <class 'numpy.ndarray'> - 1 - [495.5 555.5 215.2] - (1000, 1000, 1) - 0.026091337203979492
    <class 'numpy.ndarray'> - 1 - [674.5 166.5 800.8] - (1000, 1000, 1) - 0.0969083309173584
    <class 'numpy.ndarray'> - 1 - [595.5 631.5 217.4] - (1000, 1000, 1) - 0.037203311920166016
    <class 'numpy.ndarray'> - 1 - [497.5 630.5 210.5] - (1000, 1000, 1) - 0.030076265335083008
    <class 'numpy.ndarray'> - 1 - [623.5 552.5 314.9] - (1000, 1000, 1) - 55.67029571533203
    <class 'numpy.ndarray'> - 1 - [381.5 494.5 252.9] - (1000, 1000, 1) - 56.5454466342926
    <class 'numpy.ndarray'> - 1 - [255.5 745.5 233.7] - (1000, 1000, 1) - 0.06834101676940918
    <class 'numpy.ndarray'> - 1 - [698.5 277.5 687.6] - (1000, 1000, 1) - 0.042342424392700195
    <class 'numpy.ndarray'> - 1 - [596.5 587.5 219.7] - (1000, 1000, 1) - 0.032930612564086914
    <class 'numpy.ndarray'> - 1 - [464.5 400.5 507.2] - (1000, 1000, 1) - 0.020345687866210938
    <class 'numpy.ndarray'> - 1 - [534.5 518.5 222.9] - (1000, 1000, 1) - 0.026459217071533203
    <class 'numpy.ndarray'> - 1 - [559.5 532.5 550. ] - (1000, 1000, 1) - 0.019244670867919922
    <class 'numpy.ndarray'> - 1 - [607.5 653.5 218.3] - (1000, 1000, 1) - 0.03579878807067871
    <class 'numpy.ndarray'> - 1 - [513.5 652.5 211.4] - (1000, 1000, 1) - 0.09022235870361328
    <class 'numpy.ndarray'> - 1 - [817.5 173.5 793.2] - (1000, 1000, 1) - 0.582385778427124
    <class 'numpy.ndarray'> - 1 - [399.5 434.5 219.4] - (1000, 1000, 1) - 0.03202056884765625
    <class 'numpy.ndarray'> - 1 - [504.5 509.5 480.9] - (1000, 1000, 1) - 0.44025588035583496
    <class 'numpy.ndarray'> - 1 - [315.5 415.5 220.2] - (1000, 1000, 1) - 0.027338027954101562
    <class 'numpy.ndarray'> - 1 - [410.5 582.5 386.8] - (1000, 1000, 1) - 0.10566520690917969
    <class 'numpy.ndarray'> - 1 - [413.5 497.5 224.3] - (1000, 1000, 1) - 0.03608107566833496
    <class 'numpy.ndarray'> - 1 - [312.5 507.5 219.1] - (1000, 1000, 1) - 0.029001951217651367
    <class 'numpy.ndarray'> - 1 - [743.5 385.5  85.1] - (1000, 1000, 1) - 0.012241840362548828
    <class 'numpy.ndarray'> - 1 - [635.5 428.5  85.3] - (1000, 1000, 1) - 0.015013694763183594
    <class 'numpy.ndarray'> - 1 - [508.5 477.5  85.7] - (1000, 1000, 1) - 0.015085935592651367
    <class 'numpy.ndarray'> - 1 - [487.5 353.5  87.2] - (1000, 1000, 1) - 0.013847589492797852
    <class 'numpy.ndarray'> - 1 - [538.5 550.5 416.8] - (1000, 1000, 1) - 2.442312002182007
    <class 'numpy.ndarray'> - 1 - [450.5 506.5  87.8] - (1000, 1000, 1) - 0.013612031936645508
    <class 'numpy.ndarray'> - 1 - [456.5 491.5 360.3] - (1000, 1000, 1) - 55.21931266784668
    <class 'numpy.ndarray'> - 1 - [631.5 596.5 323.6] - (1000, 1000, 1) - 52.49200701713562
    <class 'numpy.ndarray'> - 1 - [546.5 629.5 324.9] - (1000, 1000, 1) - 54.45923686027527
    <class 'numpy.ndarray'> - 1 - [580.5 600.5 310.1] - (1000, 1000, 1) - 52.36912226676941
    <class 'numpy.ndarray'> - 1 - [365.5 556.5 341.9] - (1000, 1000, 1) - 54.55448865890503
    <class 'numpy.ndarray'> - 1 - [530.5 633.5 291.4] - (1000, 1000, 1) - 53.90721416473389
    <class 'numpy.ndarray'> - 1 - [441.5 493.5 288.2] - (1000, 1000, 1) - 57.20547413825989
    <class 'numpy.ndarray'> - 1 - [314.5 531.5 272.6] - (1000, 1000, 1) - 53.19579243659973
    <class 'numpy.ndarray'> - 1 - [393.5 586.5 301.5] - (1000, 1000, 1) - 53.938227891922
    <class 'numpy.ndarray'> - 1 - [448.5 563.5 310.1] - (1000, 1000, 1) - 53.23050594329834
    <class 'numpy.ndarray'> - 1 - [586.5 346.5 252.9] - (1000, 1000, 1) - 57.34580087661743
    <class 'numpy.ndarray'> - 1 - [384.5 501.5 288.2] - (1000, 1000, 1) - 55.027130365371704
    <class 'numpy.ndarray'> - 1 - [559.5 505.5 414.8] - (1000, 1000, 1) - 35.794716596603394
    <class 'numpy.ndarray'> - 1 - [618.5 363.5 594.8] - (1000, 1000, 1) - 0.02331686019897461
    <class 'numpy.ndarray'> - 1 - [399.5 452.5 224.7] - (1000, 1000, 1) - 0.03359556198120117
    <class 'numpy.ndarray'> - 1 - [327.5 452.5 225. ] - (1000, 1000, 1) - 0.029801607131958008
    <class 'numpy.ndarray'> - 1 - [437.5 523.5 224.9] - (1000, 1000, 1) - 0.03723621368408203
    <class 'numpy.ndarray'> - 1 - [330.5 537.5 226.7] - (1000, 1000, 1) - 0.0819694995880127
    <class 'numpy.ndarray'> - 1 - [853.5 134.5 833.5] - (1000, 1000, 1) - 0.2674739360809326
    <class 'numpy.ndarray'> - 1 - [612.5 524.5 231. ] - (1000, 1000, 1) - 0.036695241928100586
    <class 'numpy.ndarray'> - 1 - [400.5 386.5 583. ] - (1000, 1000, 1) - 0.023836851119995117
    <class 'numpy.ndarray'> - 1 - [496.5 562.5 226.7] - (1000, 1000, 1) - 0.02723860740661621
    <class 'numpy.ndarray'> - 1 - [576.5 635.5 223.3] - (1000, 1000, 1) - 0.035218000411987305
    <class 'numpy.ndarray'> - 1 - [489.5 624.5 224.6] - (1000, 1000, 1) - 0.031089067459106445
    <class 'numpy.ndarray'> - 1 - [471.5 539.5 354.5] - (1000, 1000, 1) - 54.33361601829529
    <class 'numpy.ndarray'> - 1 - [609.5 373.5 288.2] - (1000, 1000, 1) - 56.596476316452026
    <class 'numpy.ndarray'> - 1 - [573.5 347.5 311.5] - (1000, 1000, 1) - 54.998395681381226
    <class 'numpy.ndarray'> - 1 - [441.5 499.5 323.6] - (1000, 1000, 1) - 57.244996547698975
    <class 'numpy.ndarray'> - 1 - [491.5 542.5 380.8] - (1000, 1000, 1) - 57.38055086135864
    <class 'numpy.ndarray'> - 1 - [538.5 446.5 366.6] - (1000, 1000, 1) - 56.34163975715637
    <class 'numpy.ndarray'> - 1 - [580.5 390.5 313.7] - (1000, 1000, 1) - 45.95479345321655
    <class 'numpy.ndarray'> - 1 - [325.5 677.5 227.5] - (1000, 1000, 1) - 0.025115013122558594
    <class 'numpy.ndarray'> - 1 - [466.5 657.5 230.7] - (1000, 1000, 1) - 0.027589082717895508
    <class 'numpy.ndarray'> - 1 - [584.5 718.5 225.9] - (1000, 1000, 1) - 0.026874065399169922
    <class 'numpy.ndarray'> - 1 - [731.5 711.5 235.3] - (1000, 1000, 1) - 0.027145862579345703
    <class 'numpy.ndarray'> - 1 - [549.5 518.5 361.1] - (1000, 1000, 1) - 56.621410846710205
    <class 'numpy.ndarray'> - 1 - [262.5 632.5 228.8] - (1000, 1000, 1) - 0.027457714080810547
    <class 'numpy.ndarray'> - 1 - [335.5 621.5 231.1] - (1000, 1000, 1) - 0.024728059768676758
    <class 'numpy.ndarray'> - 1 - [440.5 648.5 226. ] - (1000, 1000, 1) - 0.026103973388671875
    <class 'numpy.ndarray'> - 1 - [558.5 670.5 229.5] - (1000, 1000, 1) - 0.026404380798339844
    <class 'numpy.ndarray'> - 1 - [683.5 693.5 228.7] - (1000, 1000, 1) - 0.0240175724029541
    <class 'numpy.ndarray'> - 1 - [735.5 412.5 552.2] - (1000, 1000, 1) - 0.05063891410827637
    <class 'numpy.ndarray'> - 1 - [272.5 521.5 233.4] - (1000, 1000, 1) - 0.02669548988342285
    <class 'numpy.ndarray'> - 1 - [414.5 526.5 234. ] - (1000, 1000, 1) - 0.027920007705688477
    <class 'numpy.ndarray'> - 1 - [484.5 619.5 229.7] - (1000, 1000, 1) - 0.03190493583679199
    <class 'numpy.ndarray'> - 1 - [623.5 628.5 230.3] - (1000, 1000, 1) - 0.024868488311767578
    <class 'numpy.ndarray'> - 1 - [703.5 156.5 683.4] - (1000, 1000, 1) - 0.3205733299255371
    <class 'numpy.ndarray'> - 1 - [727.5 668.5 232.9] - (1000, 1000, 1) - 0.025986671447753906
    <class 'numpy.ndarray'> - 1 - [238.5 520.5 231.6] - (1000, 1000, 1) - 0.029259443283081055
    <class 'numpy.ndarray'> - 1 - [380.5 523.5 231. ] - (1000, 1000, 1) - 0.03371596336364746
    <class 'numpy.ndarray'> - 1 - [481.5 565.5 233.6] - (1000, 1000, 1) - 0.02984452247619629
    <class 'numpy.ndarray'> - 1 - [686.5 254.5 663. ] - (1000, 1000, 1) - 0.027632474899291992
    <class 'numpy.ndarray'> - 1 - [648.5 521.5 236.5] - (1000, 1000, 1) - 0.02853107452392578
    <class 'numpy.ndarray'> - 1 - [859.5   6.5 847.9] - (1000, 1000, 1) - 0.08589458465576172
    <class 'numpy.ndarray'> - 1 - [701.5 573.5 189.8] - (1000, 1000, 1) - 0.02869558334350586
    <class 'numpy.ndarray'> - 1 - [252.5 448.5 235.1] - (1000, 1000, 1) - 0.028528451919555664
    <class 'numpy.ndarray'> - 1 - [377.5 472.5 231. ] - (1000, 1000, 1) - 0.025413036346435547
    <class 'numpy.ndarray'> - 1 - [476.5 514.5 230.9] - (1000, 1000, 1) - 0.03474593162536621
    <class 'numpy.ndarray'> - 1 - [775.5 239.5 615.5] - (1000, 1000, 1) - 0.02543473243713379
    <class 'numpy.ndarray'> - 1 - [600.5 493.5 193.4] - (1000, 1000, 1) - 0.026931285858154297
    <class 'numpy.ndarray'> - 1 - [454.5 806.5 546.1] - (1000, 1000, 1) - 0.022521257400512695
    <class 'numpy.ndarray'> - 1 - [738.5 549.5 232.4] - (1000, 1000, 1) - 0.029941082000732422
    <class 'numpy.ndarray'> - 1 - [294.5 430.5 234.9] - (1000, 1000, 1) - 0.024034500122070312
    <class 'numpy.ndarray'> - 1 - [595.5  32.5 933.1] - (1000, 1000, 1) - 0.02588486671447754
    <class 'numpy.ndarray'> - 1 - [425.5 444.5 231.3] - (1000, 1000, 1) - 0.030086517333984375
    <class 'numpy.ndarray'> - 1 - [313.5 600.5 384.4] - (1000, 1000, 1) - 0.024771928787231445
    <class 'numpy.ndarray'> - 1 - [548.5 460.5 230.5] - (1000, 1000, 1) - 0.03226423263549805
    <class 'numpy.ndarray'> - 1 - [658.5 494.5 237. ] - (1000, 1000, 1) - 0.030545711517333984
    <class 'numpy.ndarray'> - 1 - [475.5 439.5 233.4] - (1000, 1000, 1) - 0.030675649642944336
    <class 'numpy.ndarray'> - 1 - [610.5 138.5 827.9] - (1000, 1000, 1) - 1.8789010047912598
    <class 'numpy.ndarray'> - 1 - [342.5 327.5 242.6] - (1000, 1000, 1) - 0.02548813819885254
    <class 'numpy.ndarray'> - 1 - [404.5 671.5 419.2] - (1000, 1000, 1) - 0.024164676666259766
    <class 'numpy.ndarray'> - 1 - [428.5 382.5 234.9] - (1000, 1000, 1) - 0.02910017967224121
    <class 'numpy.ndarray'> - 1 - [302.5 723.5 676.4] - (1000, 1000, 1) - 0.023247480392456055
    <class 'numpy.ndarray'> - 1 - [549.5 402.5 235. ] - (1000, 1000, 1) - 0.0335390567779541
    <class 'numpy.ndarray'> - 1 - [701.5 374.5 241.3] - (1000, 1000, 1) - 0.02838134765625
    <class 'numpy.ndarray'> - 1 - [250.5 301.5 239.6] - (1000, 1000, 1) - 0.02828812599182129
    <class 'numpy.ndarray'> - 1 - [ 58.5 803.5 781.6] - (1000, 1000, 1) - 5.099250316619873
    <class 'numpy.ndarray'> - 1 - [315.5 316.5 236.9] - (1000, 1000, 1) - 0.028528451919555664
    <class 'numpy.ndarray'> - 1 - [422.5 433.5 405.5] - (1000, 1000, 1) - 0.02231431007385254
    <class 'numpy.ndarray'> - 1 - [446.5 338.5 235.9] - (1000, 1000, 1) - 0.031496286392211914
    <class 'numpy.ndarray'> - 1 - [568.5 366.5 234.1] - (1000, 1000, 1) - 0.029977798461914062
    <class 'numpy.ndarray'> - 1 - [657.5 391.5 231.6] - (1000, 1000, 1) - 0.028259754180908203
    <class 'numpy.ndarray'> - 1 - [257.5 245.5 232.5] - (1000, 1000, 1) - 0.030965805053710938
    <class 'numpy.ndarray'> - 1 - [391.5 606.5 579.8] - (1000, 1000, 1) - 0.20736169815063477
    <class 'numpy.ndarray'> - 1 - [424.5 213.5 242.6] - (1000, 1000, 1) - 0.031389713287353516
    <class 'numpy.ndarray'> - 1 - [533.5 273.5 245.9] - (1000, 1000, 1) - 0.02604532241821289
    <class 'numpy.ndarray'> - 1 - [615.5 352.5  93.5] - (1000, 1000, 1) - 0.015076160430908203
    <class 'numpy.ndarray'> - 1 - [593.5 315.5  92.5] - (1000, 1000, 1) - 0.016647815704345703
    <class 'numpy.ndarray'> - 1 - [534.5 451.5  90.2] - (1000, 1000, 1) - 0.019681215286254883
    <class 'numpy.ndarray'> - 1 - [586.5 340.5  94.2] - (1000, 1000, 1) - 0.019324541091918945
    <class 'numpy.ndarray'> - 1 - [593.5 621.5 344.6] - (1000, 1000, 1) - 5.504838466644287
    <class 'numpy.ndarray'> - 1 - [522.5 419.5  95.9] - (1000, 1000, 1) - 0.017519712448120117
    <class 'numpy.ndarray'> - 1 - [565.5 628.5 288.2] - (1000, 1000, 1) - 56.1322922706604
    <class 'numpy.ndarray'> - 1 - [613.5 536.5 522.1] - (1000, 1000, 1) - 0.025838613510131836
    <class 'numpy.ndarray'> - 1 - [933.5 228.5  56.1] - (1000, 1000, 1) - 0.012070178985595703
    <class 'numpy.ndarray'> - 1 - [921.5  31.5 106.7] - (1000, 1000, 1) - 0.017765283584594727
    <class 'numpy.ndarray'> - 1 - [254.5 330.5  60.2] - (1000, 1000, 1) - 0.014133453369140625
    <class 'numpy.ndarray'> - 1 - [266.5 949.5 100.8] - (1000, 1000, 1) - 0.01753997802734375
    <class 'numpy.ndarray'> - 1 - [530.5 407.5 320.5] - (1000, 1000, 1) - 46.42959547042847
    <class 'numpy.ndarray'> - 1 - [624.5 333.5 247.8] - (1000, 1000, 1) - 0.03729510307312012
    <class 'numpy.ndarray'> - 1 - [528.5 331.5 248.4] - (1000, 1000, 1) - 0.035808563232421875
    <class 'numpy.ndarray'> - 1 - [596.5 454.5 241.8] - (1000, 1000, 1) - 0.03575849533081055
    <class 'numpy.ndarray'> - 1 - [257.5 853.5 717.5] - (1000, 1000, 1) - 0.024783611297607422
    <class 'numpy.ndarray'> - 1 - [500.5 447.5 241.6] - (1000, 1000, 1) - 0.09708929061889648
    <class 'numpy.ndarray'> - 1 - [625.5 327.5 252.7] - (1000, 1000, 1) - 0.031558990478515625
    <class 'numpy.ndarray'> - 1 - [491.5 363.5 244. ] - (1000, 1000, 1) - 0.03095841407775879
    <class 'numpy.ndarray'> - 1 - [581.5 435.5 245.2] - (1000, 1000, 1) - 0.042731285095214844
    <class 'numpy.ndarray'> - 1 - [108.5 755.5 731.8] - (1000, 1000, 1) - 0.029768705368041992
    <class 'numpy.ndarray'> - 1 - [489.5 436.5 243.2] - (1000, 1000, 1) - 0.03483724594116211"""

    lines = s.split("\n")
    print(len(lines))

    total = 0
    cnt = 0

    for l in lines:

        try:

            abc = l.split("-")

            if len(abc) == 5:

                ll = float(abc[4].strip())

                total += ll
                cnt += 1

            # print( ll )
        except Exception as ex:

            print( l )
            raise ex

    print( total )
    print( cnt )
    print( total / cnt )

    print( (17000-cnt) * 14 / 3600 )
    return


@app.cell
def _():

    15.5*17000 / 3600

    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
