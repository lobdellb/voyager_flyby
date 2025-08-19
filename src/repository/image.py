from sqlalchemy.orm import Session
from sqlalchemy import select
import pvl
import datetime

from models.image import VoyagerImage


# def get_user(db: Session, user_id: int):
#     return db.query(User).filter(User.id == user_id).first()

def create_user_from_dict(db: Session, d: dict ):
    obj = voyager_image_from_dict(d)
    db.add(obj)
    db.commit()
    db.refresh(obj)
    return obj




def upsert_foobar(db: Session, voyager_image_dict: dict ):
    """
    Insert or update a Foobar record.
    
    Args:
        session (Session): SQLAlchemy session.
        foobar (Foobar): A Foobar ORM object to insert/update.
    """

    voyager_image_obj = voyager_image_from_dict(voyager_image_dict)
    
    stmt = select(VoyagerImage).where(VoyagerImage.PRODUCT_ID == voyager_image_obj.PRODUCT_ID)
    existing = db.execute(stmt).scalar_one_or_none()

    if existing:
        # update existing record
        for attr, value in voyager_image_obj.__dict__.items():
            if attr.startswith("_"):  # skip SQLAlchemy internals
                continue
            setattr(existing, attr, value)
        
        db.commit()
        return existing

    else:
        # insert new record
        db.add(voyager_image_obj)
        db.commit()
        db.refresh(voyager_image_obj)
        return voyager_image_obj





def voyager_image_from_dict(d: dict ) -> VoyagerImage:
    """
    Create a VoyagerImage instance from a dict of PVL/PDS fields.
    Extra keys are ignored; missing keys default to None.
    """
    cols = {c.name for c in VoyagerImage.__table__.columns}
    cols.discard("id")  # don't allow external 'id' to be set
    payload = {k: d.get(k) for k in cols}

    return VoyagerImage(**payload)



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
