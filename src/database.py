import logging

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base


import config

DATABASE_URL = None

engine = create_engine(config.db_conn_str, echo=True, future=True)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

Base = declarative_base()



# logfile = 'logs/sqlalchemy.log'

# logging.basicConfig()
# logger = logging.getLogger('sqlalchemy')
# logger.propagate = False
# handler = logging.FileHandler(logfile, mode='a')
# logger.setLevel(logging.DEBUG)

# logger.handlers.clear()

# handler.setLevel(logging.DEBUG)
# # logger.addHandler(handler)

# # Suppress output from child loggers
# logging.getLogger('sqlalchemy.engine').handlers.clear()
# logging.getLogger('sqlalchemy.pool').handlers.clear()
# logging.getLogger('sqlalchemy.orm').handlers.clear()


# logging.basicConfig(
#     level=logging.INFO,
#     handlers=[logging.FileHandler("logs/sqlalchemy.log", mode="a", encoding="utf-8")],
#     format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
#     force=True,   # <-- this is the key: wipes any console handlers others set up
# )

# 2) Make sure the right SQLAlchemy loggers are set (Engine logs SQL)
for name in ("sqlalchemy", "sqlalchemy.engine", "sqlalchemy.engine.Engine", "sqlalchemy.pool"):

    this_logger = logging.getLogger(name)

    this_logger.setLevel(logging.INFO)
    this_logger.propagate = False
    this_logger.handlers.clear()
    this_logger.addHandler(logging.FileHandler(f"logs/{name}.log", mode="a", encoding="utf-8"))
