# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
This is a Python project for processing Voyager planetary encounter data to create flyby videos. The project extracts and processes raw images and metadata from tar.gz archives containing VICAR image files from NASA's Voyager missions, specifically targeting Saturn flyby data.

## Dependencies and Setup
- Install dependencies with: `pip install -r requirements.txt`
- Key dependencies include: marimo, rms-vicar, pvl, numpy, matplotlib, opencv-python, pandas
- Uses VICAR format for NASA image processing
- Source data stored in `source_files/` as VGISS_*.tar.gz archives

## Project Structure
- `src/main.py` - Entry point with pipeline orchestration and tar file processing logic
- `src/pipeline.py` - Core pipeline framework with Task and Pipeline classes
- `src/config.py` - Path configuration (cache, source, output directories)
- `src/saturn_flyby_pipeline_notebook.py` - Marimo notebook for interactive analysis
- `src/analysis.py` - Currently empty analysis module
- `cache/` - Cached processing results and extracted files
- `output_files/` - Final processed outputs
- `source_files/` - Input tar.gz archives with raw Voyager image data

## Architecture
The project uses a task-based pipeline architecture:
- `Pipeline` class orchestrates sequential task execution
- `Task` base class defines processing interface with `process(item)` method
- Tasks can be 0→many, many→many, or many→few transformations
- `FileNameParser` handles VICAR filename parsing with pattern matching
- Caching system prevents reprocessing of extracted tar members

## Key Data Flow
1. List tar.gz files in source_files/
2. Extract and cache tar file member information
3. Filter for GEOMED IMG/LBL file pairs
4. Parse VICAR image data and PVL metadata
5. Generate normalized JPG outputs and preserve raw data

## Running the Project
- Main pipeline: `python src/main.py`
- Interactive notebook: `marimo run src/saturn_flyby_pipeline_notebook.py`
- Run tests: `python -m pytest` or `python -m pytest tests/test_helpers.py -v`

## File Processing Details
- Targets files matching pattern: `VGISS_*/DATA/C*/C*_GEOMED.{IMG,LBL}`
- IMG files contain VICAR image data, LBL files contain PVL metadata
- Metadata includes: IMAGE_TIME, FILTER_NAME, TARGET_NAME, EXPOSURE_DURATION
- Images are normalized and saved as both JPG and pickled numpy arrays