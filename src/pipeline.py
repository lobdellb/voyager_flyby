import vicar
import matplotlib.pyplot as plt
import glob
import numpy as np
import re
import pathlib
import pandas as pd
import pvl
import tarfile
import re
import os
import tqdm
import pickle
import cv2
import sqlite3
import datetime
import warnings
import sys
import itertools
import logging
import time

# Some of the cv2 functionality will throw a warning when I want an exception.
warnings.filterwarnings("error", category=RuntimeWarning)

import pickle

logger = logging.getLogger(__name__)




class FileNameParser:

    def __init__(self, filename):
        self.filename = filename
        self.parts = self._parse_filename()

    def _parse_filename(self):
        pattern = re.compile(
            r"(?P<mission>VGISS_\d+)/DATA/(?P<directory>C\d+)/(?P<basename>C\d+)(?:_(?P<suffix>[A-Z]+))?\.(?P<extension>[A-Z]+)"
        )
        match = pattern.match(self.filename)
        if not match:
            raise ValueError(f"Filename '{self.filename}' does not match expected pattern.")
        return match.groupdict()

    def get_part(self, part_name):
        return self.parts.get(part_name)

    def __repr__(self):
        return f"FileNameParser({self.filename}) -> {self.parts}"
















# class Item:
#     def __init__(self, db_path, cache_dir):
#         self.db_path = db_path
#         self.cache_dir = pathlib.Path(cache_dir)
#         self.cache_dir.mkdir(parents=True, exist_ok=True)
#         self._initialize_db()

#     def _initialize_db(self):
#         with sqlite3.connect(self.db_path) as conn:
#             cursor = conn.cursor()
#             cursor.execute("""
#                 CREATE TABLE IF NOT EXISTS items (
#                     item_id INTEGER PRIMARY KEY AUTOINCREMENT,
#                     run_id INTEGER,
#                     code_version TEXT,
#                     key TEXT UNIQUE,
#                     metadata TEXT,
#                     file_path TEXT,
#                     FOREIGN KEY(run_id) REFERENCES runs(run_id)
#                 )
#             """)
#             conn.commit()

#     def save(self, run_id, key, data, metadata=None, code_version="unknown"):
#         file_path = self.cache_dir / f"{key}.pkl"
#         with open(file_path, "wb") as f:
#             pickle.dump(data, f)

#         with sqlite3.connect(self.db_path) as conn:
#             cursor = conn.cursor()
#             cursor.execute("""
#                 INSERT OR REPLACE INTO items (run_id, code_version, key, metadata, file_path)
#                 VALUES (?, ?, ?, ?, ?)
#             """, (run_id, code_version, key, pickle.dumps(metadata), str(file_path)))
#             conn.commit()

#     def load(self, key):
#         with sqlite3.connect(self.db_path) as conn:
#             cursor = conn.cursor()
#             cursor.execute("""
#                 SELECT file_path FROM items WHERE key = ?
#             """, (key,))
#             result = cursor.fetchone()

#         if result:
#             file_path = result[0]
#             with open(file_path, "rb") as f:
#                 return pickle.load(f)
#         else:
#             raise KeyError(f"Item with key '{key}' not found.")

#     def get_metadata(self, key):
#         with sqlite3.connect(self.db_path) as conn:
#             cursor = conn.cursor()
#             cursor.execute("""
#                 SELECT metadata FROM items WHERE key = ?
#             """, (key,))
#             result = cursor.fetchone()

#         if result:
#             return pickle.loads(result[0])
#         else:
#             raise KeyError(f"Metadata for key '{key}' not found.")




# class Step:
#     def __init__(self, name):
#         self.name = name

#     def run(self):
#         pass

        

# class Pipeline:
#     def __init__(self, db_path ):
#         self.db_path = db_path
#         self.steps = []
#         self._initialize_db()

#     def _initialize_db(self):
#         with sqlite3.connect(self.db_path) as conn:
#             cursor = conn.cursor()
#             cursor.execute("""
#                 CREATE TABLE IF NOT EXISTS runs (
#                     run_id INTEGER PRIMARY KEY,
#                     start_time TEXT,
#                     end_time TEXT,
#                     status TEXT
#                 )
#             """)
#             cursor.execute("""
#                 CREATE TABLE IF NOT EXISTS steps (
#                     step_id INTEGER PRIMARY KEY AUTOINCREMENT,
#                     run_id INTEGER,
#                     step_name TEXT,
#                     start_time TEXT,
#                     end_time TEXT,
#                     status TEXT,
#                     FOREIGN KEY(run_id) REFERENCES runs(run_id)
#                 )
#             """)
#             conn.commit()

#     def add_step(self, step):
#         if not isinstance(step, Step):
#             raise ValueError("Only Step instances can be added to the pipeline.")
#         self.steps.append(step)

#     def start_run(self):
#         with sqlite3.connect(self.db_path) as conn:
#             cursor = conn.cursor()
#             cursor.execute("""
#                 INSERT INTO runs (start_time, status) VALUES (?, ?)
#             """, (datetime.datetime.now().isoformat(), "running"))
#             conn.commit()
#             return cursor.lastrowid

#     def end_run(self, run_id, status="completed"):
#         with sqlite3.connect(self.db_path) as conn:
#             cursor = conn.cursor()
#             cursor.execute("""
#                 UPDATE runs SET end_time = ?, status = ? WHERE run_id = ?
#             """, (datetime.datetime.now().isoformat(), status, run_id))
#             conn.commit()

#     def start_step(self, run_id, step_name):
#         with sqlite3.connect(self.db_path) as conn:
#             cursor = conn.cursor()
#             cursor.execute("""
#                 INSERT INTO steps (run_id, step_name, start_time, status) VALUES (?, ?, ?, ?)
#             """, (run_id, step_name, datetime.datetime.now().isoformat(), "running"))
#             conn.commit()
#             return cursor.lastrowid

#     def end_step(self, step_id, status="completed", metadata=None):
#         with sqlite3.connect(self.db_path) as conn:
#             cursor = conn.cursor()
#             cursor.execute("""
#                 UPDATE steps SET end_time = ?, status = ?, metadata = ? WHERE step_id = ?
#             """, (datetime.datetime.now().isoformat(), status, metadata, step_id))
#             conn.commit()

#     def get_run_metadata(self, run_id):
#         with sqlite3.connect(self.db_path) as conn:
#             cursor = conn.cursor()
#             cursor.execute("""
#                 SELECT * FROM runs WHERE run_id = ?
#             """, (run_id,))
#             run_data = cursor.fetchone()
#             cursor.execute("""
#                 SELECT * FROM steps WHERE run_id = ?
#             """, (run_id,))
#             steps_data = cursor.fetchall()
#             return {"run": run_data, "steps": steps_data}

#     def run_pipeline(self):
#         run_id = self.start_run()
#         try:
#             for step in self.steps:
#                 step_id = self.start_step(run_id, step.name)
#                 try:
#                     metadata = step.run()
#                     self.end_step(step_id, metadata=metadata)
#                 except Exception as e:
#                     self.end_step(step_id, status="failed", metadata=str(e))
#                     raise
#             self.end_run(run_id)
#         except Exception:
#             self.end_run(run_id, status="failed")
#             raise










# How I know I've succeeded:
# - Work won't be redone unless relevant code changes.
# - All steps have access to any metadata created by previous steps, forward and backward.
# - All the objects and their state are persisted in the data store.


# What I want this pipeline to look like:

# List fetcher yeilds items --> processing step processes items and yeilds items --> and so-on












# Some changes to make
# - store more metadata
#   - the target object
# - put the original fn, the time, and the object in the filename
# - store the original 16 bit samples in .p format as well

# what I want is to find every LBL and _GEOMED.IMG file, and load them together, though not necessariy at the same time


# Goals in order:
# - Generate the list of .tar.gz files
# - Open those and flow out the file lists
# - Filter the files list for those we want
# - Unzip the files list




class Pipeline:

    def __init__(self):
        self.tasks = []

    def add_task(self, task):
        if not isinstance(task, Task):
            raise ValueError("Only Task instances can be added to the pipeline.")
        self.tasks.append(task)

    def run(self, initial_items ):

        # Workflow
        # - Loop over all tasks
        #   - Initial state: Last output
        #   - For each over the last output
        #     - Call the task's process method
        #     - Collect the results and store into the last output. 

        if len(initial_items) < 1:
            raise ValueError("Initial items must contain at least one item.")

        items = initial_items

        for task in self.tasks:

            logger.info(f"Starting task: {task.name}")
            start_time = time.time()

            new_items = []

            for item in tqdm.tqdm(items):
                new_items.append( task.process(item) )
            
            run_time = time.time() - start_time
            logger.info(f"Task {task.name} completed in {run_time:.1f} seconds.")
            
            items = itertools.chain.from_iterable( new_items )

        return items

        



class Task:

    # three types:
    # 0 in --> lots out
    # lots in --> lots out (not necessarily 1:1)
    # lots in --> few out

    def __init__(self, name):
        self.name = name

    def process(self, item ):
        raise NotImplementedError("Subclasses should implement this method.")


