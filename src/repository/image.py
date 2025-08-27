from sqlalchemy.orm import Session
from sqlalchemy import select
import pvl
import datetime
import json

from models.image import VoyagerImage


# def get_user(db: Session, user_id: int):
#     return db.query(User).filter(User.id == user_id).first()





class DatetimeReprEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime.datetime):
            return repr(obj)  # Return the __repr__() string of the datetime object
        return s
    


def upsert_image_metadata(session: Session, product_id: str, fn: str ):
    """
    Insert or update a Foobar record.
    
    Args:
        session (Session): SQLAlchemy session.
        foobar (Foobar): A Foobar ORM object to insert/update.
    """
    
    existing = get_voyager_image_by_product_id(session, product_id )

    if existing:
        # # update existing record
        # for attr, value in voyager_image_obj.__dict__.items():
        #     if attr.startswith("_"):  # skip SQLAlchemy internals
        #         continue
        #     setattr(existing, attr, value)
        
        # # commit happens in the caller
        # # session.commit()
        # return existing

        return existing

    else:
        # insert new record
        # session.add(voyager_image_obj)
        # commit happens in the caller
        # session.commit()

        with open( fn ) as fp:
            
            # Loading these takes all day, so lets look-up the product_id before we
            # go and bother to load the file.
            
            metadata = pvl.loads( fp.read() )

        flattened_dict = handle_special_cases( 
            flatten_vicar_object(metadata, to_exclude=["^VICAR_HEADER", "^IMAGE", "SOURCE_PRODUCT_ID"])
        )

        voyager_image_obj = voyager_image_from_dict( d=flattened_dict )

        session.add(voyager_image_obj)

        try:
            session.commit()
        except Exception as e:
            print("*"*50)
            print( product_id ), 
            print( fn )
            print( json.dumps( flattened_dict, cls=DatetimeReprEncoder ) )
            print("*"*50)
            raise e

        return voyager_image_obj





def voyager_image_from_dict(d: dict ) -> VoyagerImage:
    """
    Create a VoyagerImage instance from a dict of PVL/PDS fields.
    Extra keys are ignored; missing keys default to None.
    """
    cols = {c.name for c in VoyagerImage.__table__.columns}
    cols.discard("id")  # don't allow external 'id' to be set

    # if a datetime is "UNK" then we make it non

    payload = {k: d.get(k) for k in cols}

    voyager_image_obj = VoyagerImage(**payload)

    return voyager_image_obj





def handle_special_cases( flattened_vicar: dict ):

    # case 1
    for k in flattened_vicar:

        if k.endswith("TIME") and flattened_vicar[k] == "UNK":
            flattened_vicar[k] = None

    return flattened_vicar 






def flatten_vicar_object( o: pvl.collections.PVLModule | pvl.collections.PVLObject , to_exclude: list) -> dict:
    """The intent is to accept a vicar PVLModule object and flatten it into a python dict."""

    out = {}
    
    for element in o:

        key = element[0]
        val = element[1]

        if key in to_exclude:
            continue

        # print( f"--{key}--" )

        if isinstance( val, pvl.collections.PVLObject ) or isinstance( val, pvl.collections.PVLModule ):
            subdict = flatten_vicar_object( val, to_exclude=to_exclude )

            # print( subdict )
            # raise Exception("stop")
            
            for k,v in subdict.items():
                out[f"{key}_{k}"] = v

        elif type(val) in [ str, int, bool, float, datetime.datetime ]:
            out[ key ] = val
        elif isinstance( val, pvl.collections.Quantity ):
            out[ f"{key}_value" ] = val.value
            out[ f"{key}_units" ] = val.units
        elif isinstance( val, list ):

            if key not in ["^VICAR_HEADER","^IMAGE","SOURCE_PRODUCT_ID"]:
                raise Exception(f"flatten_vicar_object: found an unfamiliar key pointing to a list --{key}-- with data --{val}--.") 

            # Not going to keep these, not interesting.
        
        else:
            raise TypeError(f"flatten_vicar_object: Call with unhandled type {type(val)} for key --{key}--.")    

    return out


def get_voyager_image_by_product_id(session: Session, product_id: str):
    """
    Looks up a VoyagerImage by PRODUCT_ID. If it exists, returns a populated instance of VoyagerImage.
    If it does not exist, returns False.

    Args:
        session (Session): SQLAlchemy session.
        product_id (str): The PRODUCT_ID to look up.

    Returns:
        VoyagerImage | bool: A populated VoyagerImage instance if found, otherwise False.
    """
    stmt = select(VoyagerImage).where(VoyagerImage.PRODUCT_ID == product_id)
    result = session.execute(stmt).scalar_one_or_none()

    if result:
        return result
    return False
