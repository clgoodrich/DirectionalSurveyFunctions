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
import pyproj
from shapely.geometry import Polygon
import sqlite3
from welltrajconvert.wellbore_trajectory import *
import pandas as pd
from shapely import wkt
from DXClearance import ClearanceProcess
from DXSurveys import SurveyProcess
import math
import copy
import utm
import os
import os
from sqlalchemy import create_engine
from urllib.parse import quote_plus
from functools import lru_cache
import logging
from sqlalchemy.orm import sessionmaker
from contextlib import contextmanager
import pyodbc
from pyproj import Geod, Proj, CRS

class DatabaseManager:
    def __init__(self):
        # Initialize the connection
        self.connector = SQLConnector()
        self.engine = self.connector.get_engine()

        # Create a session factory
        self.Session = sessionmaker(bind=self.engine)

    @contextmanager
    def get_session(self):
        """Context manager for database sessions"""
        session = self.Session()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    # Example 1: Raw SQL query
    def execute_raw_query(self, query, params=None):
        with self.engine.connect() as connection:
            result = connection.execute(query, params or {})
            return result.fetchall()

    # Example 2: Query using pandas
    def query_to_dataframe(self, query):
        return pd.read_sql_query(query, self.engine)

    # Example 3: Execute with parameters
    def get_well_data(self, well_id):
        query = """
        SELECT *
        FROM Wells
        WHERE WellID = :well_id
        """
        return self.execute_raw_query(query, {'well_id': well_id})


class SQLConnector:
    def __init__(self, login_file="logininfo.txt"):
        self.login_file = login_file
        self.logger = logging.getLogger(__name__)
        self.engine = self._create_sql_connection()

    def _parse_credentials(self, content):
        """Parse username and password from file content."""
        try:
            lines = content.strip().split('\n')
            return {
                'username': lines[0].split(':')[1].strip(),
                'password': lines[1].split(':')[1].strip()
            }
        except (IndexError, KeyError) as e:
            self.logger.error(f"Error parsing credentials: {e}")
            return None

    def _get_credentials(self):
        """Read credentials from file."""
        try:
            file_path = os.path.join(os.getcwd(), self.login_file)
            with open(file_path, 'r') as file:
                content = file.read()
            return self._parse_credentials(content)
        except FileNotFoundError:
            self.logger.warning(f"Credentials file not found: {self.login_file}")
            return None

    @lru_cache(maxsize=1)
    def _create_connection_string(self):
        """Create connection string with caching."""
        credentials = self._get_credentials()

        if credentials:
            # Production connection
            params = {
                'driver': '{SQL Server}',
                'server': 'oilgas-sql-prod.ogm.utah.gov',
                'database': 'UTRBDMSNET',
                'uid': credentials['username'],
                'pwd': credentials['password']
            }

            conn_str = (
                "DRIVER={driver};"
                "SERVER={server};"
                "DATABASE={database};"
                "UID={uid};"
                "PWD={pwd}"
            ).format(**params)
        else:
            # Fallback to local connection
            conn_str = (
                "Driver={SQL Server};"
                r"Server=CGLAPTOP\SQLEXPRESS;"
                "Database=UTRBDMSNET;"
                "Trusted_Connection=yes;"
            )

        return quote_plus(conn_str)

    def _create_sql_connection(self):
        """Create SQLAlchemy engine."""
        try:
            connection_string = self._create_connection_string()
            return create_engine(
                f"mssql+pyodbc:///?odbc_connect={connection_string}",
                pool_pre_ping=True,
                pool_recycle=3600
            )
        except Exception as e:
            self.logger.error(f"Failed to create SQL connection: {e}")
            raise

    def get_engine(self):
        """Return SQLAlchemy engine instance."""
        return self.engine



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
import glob
def search_py_files_for_kop(folder_path):
    """
    Search through all .py files in a folder for the string 'kop'

    Args:
        folder_path (str): Path to the folder to search

    Returns:
        list: List of file paths containing 'kop'
    """
    # Get list of all .py files in folder and subfolders
    py_files = glob.glob(os.path.join(folder_path, '**/*.py'), recursive=True)

    files_with_kop = []

    # Search each file
    for file_path in py_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().lower()
                if 'kop' in content:
                    files_with_kop.append(file_path)
        except Exception as e:
            print(f"Error reading {file_path}: {str(e)}")
    print(files_with_kop)
    return files_with_kop


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

def processAllData():
    def utm_to_latlon_row(row, utm_zone=12, hemisphere='N'):
        utm_proj = f'+proj=utm +zone={utm_zone} +{hemisphere} +ellps=WGS84 +datum=WGS84 +units=m +no_defs'
        transformer = pyproj.Transformer.from_proj(
            pyproj.Proj(utm_proj),
            pyproj.Proj('+proj=longlat +datum=WGS84 +no_defs'),
            always_xy=True
        )
        lon, lat = transformer.transform(row['X'], row['Y'])
        return pd.Series({'Latitude': lat, 'Longitude': lon})

    db = DatabaseManager()
    query = f"""SELECT dsh.APINumber, MeasuredDepth, Inclination, Azimuth, CitingType, dsd.X, dsd.Y, dsh.SurveySurfaceElevation,
CASE
                When vsl.ConstructType = 'D' then 'DIRECTIONAL'
                When vsl.ConstructType = 'H' then 'HORIZONTAL'
                When vsl.ConstructType = 'V' then 'VERTICAL'
            END as 'Slant'

            FROM DirectionalSurveyHeader dsh
            JOIN DirectionalSurveyData dsd on dsd.DirectionalSurveyHeaderKey = dsh.Pkey
			join Well w on w.WellID = dsh.APINumber
			LEFT JOIN WellHistory wh on wh.WellKey = w.pkey
			LEFT JOIN vw_DON_WH_SLANT vsl on vsl.SlantHistKey = wh.PKey
            order by dsh.APINumber, CitingType, MeasuredDepth"""

    df = db.query_to_dataframe(query)

    # df[['lat', 'long']] = df.apply(utm_to_latlon_row, axis=1)
    # print(df)
    counter = 0
    grouped = df.groupby(['APINumber', 'CitingType'])
    total_things = len(grouped)
    print(total_things)

    for group_name, group_data in grouped:
        slants = group_data['Slant'].unique()
        group_data = group_data.drop(columns=['Slant'])
        # if 'VERTICAL' not in slants:
        #
        if '4301354249' not in group_name and 'VERTICAL' not in slants:
        # if group_name == ('4301354249', 'Planned') and 'VERTICAL' not in slants:
            group_data = group_data.drop_duplicates(keep="first")
            print(group_name, counter, f"""/""", total_things)
            # print(group_data)
            elevation = group_data['SurveySurfaceElevation'].iloc[0]
            group_data[['lat', 'lon']] = group_data.apply(utm_to_latlon_row, axis=1)
            new_df = group_data[['MeasuredDepth', 'Inclination', 'Azimuth', 'lat', 'lon']]
            dx_df_test = SurveyProcess(
                df_referenced=new_df,
                elevation=elevation)

            try:
                kop_test = dx_df_test.find_kop(dx_df_test.true_dx)
                lp_test = dx_df_test.find_landing_point(dx_df_test.true_dx)
            except IndexError:
                print(slants)
                print(counter)
                print(group_data)
                print(foo)
        counter += 1
        # utm_proj = f'+proj=utm +zone={12} +{'N'} +ellps=WGS84 +datum=WGS84 +units=m +no_defs'
        # print(utm_proj)
        # # Create projection transformers
        # utm_to_wgs84 = pyproj.Transformer.from_proj(
        #     pyproj.Proj(utm_proj),
        #     pyproj.Proj('+proj=longlat +datum=WGS84 +no_defs'),
        #     always_xy=True
        # )
        #
        # # Convert coordinates
        # lons, lats = utm_to_wgs84.transform(
        #     i['Easting'].values,
        #     i['Northing'].values
        # )

        # Add new columns to dataframe
        # df['Longitude'] = lons
        # df['Latitude'] = lats
        # print(df)


# Configure pandas display options
pd.set_option('display.max_columns', None)  # Show all columns when displaying DataFrames
pd.options.mode.chained_assignment = None  # Suppress chained assignment warnings

processAllData()
print(foo)
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

# survey_data = {
#     'MeasuredDepth': [0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000],
#     'Inclination': [0, 5, 15, 45, 85, 89, 89, 89, 89, 89, 89],
#     'Azimuth': [175, 175, 175, 175, 175, 175, 175, 175, 175, 175, 175],
#     'lat': [
#         47.382150, 47.381150, 47.380150, 47.379150,
#         47.378150, 47.377150, 47.376150, 47.375150,
#         47.374150, 47.373150, 47.372150
#     ],
#     'lon': [
#         -102.456789, -102.456789, -102.456789, -102.456789,
#         -102.456789, -102.456789, -102.456789, -102.456789,
#         -102.456789, -102.456789, -102.456789
#     ]
# }
# dx_df_orig = pd.DataFrame(survey_data)
# section1_coords = [
#     (47.383150, -102.457789),
#     (47.383150, -102.455789),
#     (47.381150, -102.455789),
#     (47.381150, -102.457789),
#     (47.383150, -102.457789)
# ]
#
# # Section 2 coordinates (1 mile square)
# section2_coords = [
#     (47.381150, -102.457789),
#     (47.381150, -102.455789),
#     (47.379150, -102.455789),
#     (47.379150, -102.457789),
#     (47.381150, -102.457789)
# ]
# # search_py_files_for_kop(r"C:\Work\RewriteAPD")
#
# used_plats = {'label': ['section1', 'section2'],'geometry':[Polygon(section1_coords), Polygon(section2_coords)]}
# plat_df = pd.DataFrame(used_plats)

# Process survey data and calculate clearances
print(dx_df_orig)
dx_df = SurveyProcess(
    df_referenced=dx_df_orig,
    elevation=5515)

kop = dx_df.find_kop(dx_df.true_dx)
lp = dx_df.find_landing_point(dx_df.true_dx)
print('kop', kop)
print('lp', lp)
df_test = dx_df.drilled_depths_process(dx_df.true_dx, df_depths)
# clear_df = ClearanceProcess(dx_df.true_dx, plat_df, plats_adjacent)
# processed_dx_df, footages = clear_df.clearance_data, clear_df.whole_df
