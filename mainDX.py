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

from shapely.geometry import Polygon
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


# Configure pandas display options
pd.set_option('display.max_columns', None)  # Show all columns when displaying DataFrames
pd.options.mode.chained_assignment = None  # Suppress chained assignment warnings

# Initialize database connection
conn_sample = sqlite3.connect('DX_sample.db')

# Load plat data and convert geometry columns
query = "select * from PlatDF"
plat_df = pd.read_sql(query, conn_sample)  # Plat definitions
query = "select * from PlatAdj"
plats_adjacent = pd.read_sql(query, conn_sample)  # Adjacent plat relationships
query = "select * from DX_Original"
dx_df_orig = pd.read_sql(query, conn_sample)  # Original directional survey data

# Load and process depth interval data
query = "select * from Depths"
df_depths = pd.read_sql(query, conn_sample)
df_depths['Interval'] = df_depths['Interval'].apply(dict_to_interval)

# Convert WKT strings to shapely geometry objects
plat_df['geometry'] = plat_df['geometry'].apply(lambda row: wkt.loads(row))
plat_df['centroid'] = plat_df['centroid'].apply(lambda row: wkt.loads(row))
plats_adjacent['geometry'] = plats_adjacent['geometry'].apply(lambda row: wkt.loads(row))
plats_adjacent['centroid'] = plats_adjacent['centroid'].apply(lambda row: wkt.loads(row))

survey_data = {
    'MeasuredDepth': [0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000],
    'Inclination': [0, 5, 15, 45, 85, 89, 89, 89, 89, 89, 89],
    'Azimuth': [175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175],
    'lat': [
        47.382150, 47.381150, 47.380150, 47.379150,
        47.378150, 47.377150, 47.376150, 47.375150,
        47.374150, 47.373150, 47.372150
    ],
    'lon': [
        -102.456789, -102.456789, -102.456789, -102.456789,
        -102.456789, -102.456789, -102.456789, -102.456789,
        -102.456789, -102.456789, -102.456789
    ]
}
dx_df_orig = pd.DataFrame(survey_data)
section1_coords = [
    (47.383150, -102.457789),
    (47.383150, -102.455789),
    (47.381150, -102.455789),
    (47.381150, -102.457789),
    (47.383150, -102.457789)
]

# Section 2 coordinates (1 mile square)
section2_coords = [
    (47.381150, -102.457789),
    (47.381150, -102.455789),
    (47.379150, -102.455789),
    (47.379150, -102.457789),
    (47.381150, -102.457789)
]

used_plats = {'label': ['section1', 'section2'],'geometry':[Polygon(section1_coords), Polygon(section2_coords)]}
plat_df = pd.DataFrame(used_plats)

# Process survey data and calculate clearances
dx_df = SurveyProcess(
    df_referenced=dx_df_orig,
    elevation=5515
)

kop = dx_df.find_kop(dx_df.true_dx)
print('kop', kop)
df_test = dx_df.drilled_depths_process(dx_df.true_dx, df_depths)
# clear_df = ClearanceProcess(dx_df.true_dx, plat_df, plats_adjacent)
# processed_dx_df, footages = clear_df.clearance_data, clear_df.whole_df
