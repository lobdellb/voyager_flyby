import marimo

__generated_with = "0.14.17"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import sqlite3, time, random
    import vicar
    import pvl
    import os
    import sys

    sys.path.insert(0, os.getcwd())

    os.chdir("../")
    return os, pvl


@app.cell
def _(pvl):

    fn = "cache/tar_members/VGISS_6101/DATA/C27830XX/C2783018_GEOMED.LBL"

    with open( fn ) as fp:

        # Loading these takes all day, so lets look-up the product_id before we
        # go and bother to load the file.

        metadata = pvl.loads( fp.read() )
    return (metadata,)


@app.cell
def _(metadata):
    metadata
    return


@app.cell
def _(os):
    os.getcwd()
    return


@app.cell
def _():
    from models.image import VoyagerImage
    return (VoyagerImage,)


@app.cell
def _(VoyagerImage):
    for m in dir( VoyagerImage ):

        if not m.startswith("_") and m[0].upper() == m[0]:

            attr = getattr( VoyagerImage, m )

            # print( type( attr ) )
            print( attr.type )
            # print( type(m) )
    return


@app.cell
def _(VoyagerImage, metadata):
    # for mm in dir( attr ):
    #     print( mm )

    for element in metadata:

        k = element[0]
        v = element[1]
    
        # print( k )

        try:
            attr2 = getattr( VoyagerImage, k )
    
            print( f"{k}: {type(v)} --> {attr2.type}")

        except:
            pass
        
        # break
    return (element,)


@app.cell
def _(element):
    type( element[1] )

    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
