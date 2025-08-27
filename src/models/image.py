from sqlalchemy import (
    create_engine, Column, Integer, String, Float, DateTime
)

from sqlalchemy.orm import declarative_base
import database
# Base = declarative_base()

class VoyagerImage(database.Base):
    __tablename__ = "voyager_images"

    PDS_VERSION_ID = Column(String)
    RECORD_TYPE = Column(String)
    RECORD_BYTES = Column(Integer)
    FILE_RECORDS = Column(Integer)
    DATA_SET_ID = Column(String)
    PRODUCT_ID = Column(String, unique=True, index=True,primary_key=True)
    PRODUCT_CREATION_TIME = Column(DateTime(timezone=True))
    PRODUCT_TYPE = Column(String)
    INSTRUMENT_HOST_NAME = Column(String)
    INSTRUMENT_HOST_ID = Column(String)
    INSTRUMENT_NAME = Column(String)
    INSTRUMENT_ID = Column(String)
    MISSION_PHASE_NAME = Column(String)
    TARGET_NAME = Column(String)
    IMAGE_ID = Column(String)
    IMAGE_NUMBER = Column(String)
    IMAGE_TIME = Column(DateTime(timezone=True))
    EARTH_RECEIVED_TIME = Column(DateTime(timezone=True))
    SCAN_MODE_ID = Column(String)
    SHUTTER_MODE_ID = Column(String)
    GAIN_MODE_ID = Column(String)
    EDIT_MODE_ID = Column(String)
    FILTER_NAME = Column(String)
    FILTER_NUMBER = Column(String)

    EXPOSURE_DURATION_value = Column(Float)
    EXPOSURE_DURATION_units = Column(String)

    START_TIME = Column(DateTime(timezone=True))
    STOP_TIME = Column(DateTime(timezone=True))
    SPACECRAFT_CLOCK_START_COUNT = Column(String)
    SPACECRAFT_CLOCK_STOP_COUNT = Column(String)
    NOTE = Column(String)

    VICAR_HEADER_HEADER_TYPE = Column(String)
    VICAR_HEADER_BYTES = Column(Integer)
    VICAR_HEADER_RECORDS = Column(Integer)
    VICAR_HEADER_INTERCHANGE_FORMAT = Column(String)

    IMAGE_LINES = Column(Integer)
    IMAGE_LINE_SAMPLES = Column(Integer)
    IMAGE_SAMPLE_TYPE = Column(String)
    IMAGE_SAMPLE_BITS = Column(Integer)
    IMAGE_SAMPLE_DISPLAY_DIRECTION = Column(String)
    IMAGE_LINE_DISPLAY_DIRECTION = Column(String)

    IMAGE_HORIZONTAL_PIXEL_FOV_value = Column(Float)
    IMAGE_HORIZONTAL_PIXEL_FOV_units = Column(String)
    IMAGE_VERTICAL_PIXEL_FOV_value = Column(Float)
    IMAGE_VERTICAL_PIXEL_FOV_units = Column(String)

    IMAGE_HORIZONTAL_FOV_value = Column(Float)
    IMAGE_HORIZONTAL_FOV_units = Column(String)
    IMAGE_VERTICAL_FOV_value = Column(Float)
    IMAGE_VERTICAL_FOV_units = Column(String)

    IMAGE_REFLECTANCE_SCALING_FACTOR = Column(Float)
