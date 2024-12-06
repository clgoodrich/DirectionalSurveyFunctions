"""
mainDX.py
Author: Colton Goodrich
Date: 11/10/2024
Python Version: 3.12
Directional well survey and plat clearance calculation system.

Single-line summary: Processes directional well surveys and calculates plat boundaries.

This module integrates directional survey processing and plat clearance calculations,
providing a comprehensive system for wellbore trajectory analysis. It handles both
the technical aspects of survey calculations and the spatial relationship with
lease boundaries.

The system consists of two main components:
    - Directional survey processing (magnetic field calculations, interpolation)
    - Plat boundary clearance calculations (minimum distances to boundaries)

Main features include:
    - Survey interpolation using minimum curvature
    - Magnetic field reference handling
    - Coordinate system transformations
    - Plat boundary segmentation and analysis
    - Multi-well clearance calculations
    - Support for adjacent plat relationships

Typical usage example:
    survey = DXSurvey(survey_df, start_nev=[0,0,0])
    results = survey.process_trajectory()

    clearance = ClearanceProcess(results, plat_df, adjacent_df)
    clearances = clearance.clearance_data

Dependencies:
    numpy
    pandas
    welleng
    pyproj
    shapely
    pygeomag
    scipy

Notes:
    Requires specific DataFrame structures for both survey and plat data.
    Handles multiple coordinate reference systems and magnetic corrections.
"""
import sqlite3
from welltrajconvert.wellbore_trajectory import *
from shapely import wkt
from DXClearance import ClearanceProcess
from DXSurveys import SurveyProcess
import pandas as pd
import time

import polars as pl
import pyproj
import shapely
from shapely.geometry import Polygon, Point
import sqlite3
from welltrajconvert.wellbore_trajectory import *
import pandas as pd
from shapely import wkt
from DXClearance import ClearanceProcess
from DXSurveys import SurveyProcess



"""Main directional drilling data processing script. This is an example of using the process, 
and is attached with an example well

This script handles SQL database connections, data transformations, and processing 
for directional drilling survey and plat clearance calculations.

Dependencies:
    - pandas
    - sqlite3
    - json
    - shapely.wkt
    - custom modules (SurveyProcess, ClearanceProcess)
"""
def dict_to_interval(d: Union[str, dict]) -> pd.Interval:
    """Converts dictionary or JSON string to pandas Interval object.

    Args:
        d: Dictionary or JSON string containing interval data with 'left',
           'right' and 'closed' keys

    Returns:
        pd.Interval: Pandas interval object representing the range
    """
    d = json.loads(d) if isinstance(d, str) else d
    return pd.Interval(d['left'], d['right'], closed=d['closed'])

"""Input Data Format Examples:

Plat Boundary DataFrame (plat_df):
  Required columns:
    - label (str): Section/concentration identifier
    - geometry (Polygon): Shapely polygon of boundary coordinates

  Example:
|    | label     | geometry                                                                                                                 |
|---:|:---------|:-------------------------------------------------------------------------------------------------------------------------|
|  0 | section1 | POLYGON ((47.38315 -102.457789, 47.38315 -102.455789, 47.38115 -102.455789, 47.38115 -102.457789, 47.38315 -102.457789)) |
|  1 | section2 | POLYGON ((47.38115 -102.457789, 47.38115 -102.455789, 47.37915 -102.455789, 47.37915 -102.457789, 47.38115 -102.457789)) |

Directional Survey DataFrame (dx_df_orig): 
  Required columns:
    - measured_depth (float): Measured depth in feet 
    - Inclination (float): Inclination angle in degrees
    - Azimuth (float): Azimuth angle in degrees
    - lat (float): Latitude coordinate in decimal degrees
    - lon (float): Longitude coordinate in decimal degrees

  Example values from 0-10000' MD with 1000' stations
|    |   measured_depth |   Inclination |   Azimuth |     lat |      lon |
|---:|----------------:|--------------:|----------:|--------:|---------:|
|  0 |               0 |             0 |       175 | 47.3822 | -102.457 |
|  1 |            1000 |             5 |       175 | 47.3811 | -102.457 |
|  2 |            2000 |            15 |       175 | 47.3802 | -102.457 |
|  3 |            3000 |            45 |       175 | 47.3792 | -102.457 |
|  4 |            4000 |            85 |       175 | 47.3781 | -102.457 |
|  5 |            5000 |            89 |       175 | 47.3772 | -102.457 |
|  6 |            6000 |            89 |       175 | 47.3762 | -102.457 |
|  7 |            7000 |            89 |       175 | 47.3751 | -102.457 |
|  8 |            8000 |            89 |       175 | 47.3742 | -102.457 |
|  9 |            9000 |            89 |       175 | 47.3732 | -102.457 |
| 10 |           10000 |            89 |       175 | 47.3721 | -102.457 |

Casing/interval Data (df_depths):
  Required columns:
    - feature (str): interval identifier (e.g. 'Cond', 'Surf')
    - casing_bottom (float): Bottom depth of casing in feet MD
    - interval (str): Depth range interval in format '[start, end)'

  Example shows standard casing program:
    - Conductor: 0-100'
    - Surface: 100-1700'
    - Production: 1700-6300'
    - Open Hole: 6300-20000'
ex:
|    | feature   |   casing_bottom | interval           |
|---:|:----------|---------------:|:-------------------|
|  0 | Cond      |           100   | [0.0, 100.0)        |
|  1 | Surf      |         1700   | [100.0, 1700.0)     |
|  2 | Prod      |         6300   | [1700.0, 6300.0)   |
|  3 | P2        |        20000.0 | [6300.0, 20000.00) |
"""

# Process survey data and calculate clearances
dx_df = SurveyProcess(
    df_referenced=dx_df_orig,
    elevation=5515)

kop = dx_df.find_kick_off_point(dx_df.true_dx)
lp = dx_df.find_landing_point(dx_df.true_dx)

df_test = dx_df.drilled_depths_process(dx_df.true_dx, df_depths)
clear_df = ClearanceProcess(dx_df.true_dx, plat_df)
processed_dx_df, footages = clear_df.clearance_data, clear_df.whole_df
