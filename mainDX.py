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
import time

import polars as pl
import pyproj
import shapely
from shapely.geometry import Polygon, Point
import sqlite3
from welltrajconvert.wellbore_trajectory import *
import pandas as pd
from shapely import wkt, LineString
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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
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
    # Only search the given folder, not subfolders
    py_files = glob.glob(os.path.join(folder_path, '*.py'))

    files_with_kop = []

    # Search each file
    for file_path in py_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().lower()
                if 'distance' in content:
                    files_with_kop.append(file_path)
                    print(content)
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
def analyzeTime2(function_call, args_list ):
    print(str(function_call))
    profiler = cProfile.Profile()
    profiler.runcall(function_call, *args_list )

    # Redirect pstats output to a string stream
    s = StringIO()
    ps = pstats.Stats(profiler, stream=s)
    ps.sort_stats('cumulative')
    ps.print_stats()

    # Get the string output and process it
    lines = s.getvalue().split('\n')
    data = []
    for line in lines[5:]:  # Skip the header lines
        if line.strip():
            fields = line.split(None, 5)
            if len(fields) == 6:
                ncalls, tottime, percall, cumtime, percall2, filename_lineno_function = fields
                data.append({
                    'ncalls': ncalls,
                    'tottime': float(tottime),
                    'percall': float(percall),
                    'cumtime': float(cumtime),
                    'percall2': float(percall2),
                    'filename_lineno_function': filename_lineno_function
                })

    # Create DataFrame
    df = pd.DataFrame(data)
    df.sort_values(by='cumtime')
    # df = df[(df['cumtime'] > 0.1)]

    # excluded_location = r'C:\Work\RewriteAPD\ven2'
    # filtered_df = df[~df['filename_lineno_function'].str.contains(excluded_location, case=False, regex=False)]
    # filtered_df = filtered_df[
    #     filtered_df['filename_lineno_function'].str.contains(r'C:\Work\RewriteAPD', case=False, regex=False)]

    print(df.head(10).to_markdown())


def processAllData():
    def equationDistance(x1, y1, x2, y2):
        return math.sqrt((float(x2) - float(x1)) ** 2 + (float(y2) - float(y1)) ** 2)
    def equationDistance2(points, x2, y2):
        # print([pt, x2, y2])
        # ls = LineString([pt, Point(x2, y2)])
        # # coord = shapely.get_coordinates(pt)
        # x1, y1 = shapely.get_coordinates(pt)```
        # # x1, y1 = coord[0], coord[1]

        # x1, y1 = shapely.get_coordinates(pt)
        # # Create LineString from coordinates rather than Point objects
        # ls = LineString([(x1, y1), (x2, y2)])
        # print(x2, y2)
        # utm_pt = Point(utm.from_latlon(x2, y2)[:2])
        # print(points)
        # print(x2, y2)
        # target = Point(x2, y2)
        return [LineString([pt, Point(x2, y2)]).length for pt in points]
        # return ls.length
        # return math.sqrt((float(x2) - float(x1)) ** 2 + (float(y2) - float(y1)) ** 2)

    # def haversine_distance_from_geom(points_series, target_x, target_y):
    #     """
    #     Calculate haversine distance from a series of geometry points
    #     points_series: pandas series containing shapely Points
    #     target_lat, target_lon: coordinates of target point
    #     """
    #     # Extract coordinates from geometry column
    #     lats = points_series.y  # latitude is y
    #     lons = points_series.x  # longitude is x
    #
    #     # Convert decimal degrees to radians
    #     lats, lons, target_lat, target_lon = map(
    #         np.radians, [lats, lons, target_lat, target_lon]
    #     )
    #
    #     # Haversine formula
    #     dlat = target_lat - lats
    #     dlon = target_lon - lons
    #
    #     a = np.sin(dlat / 2) ** 2 + np.cos(lats) * np.cos(target_lat) * np.sin(dlon / 2) ** 2
    #     c = 2 * np.arcsin(np.sqrt(a))
    #
    #     # Radius of earth in kilometers
    #     r = 6371
    #
    #     return c * r


    def geometryTransform(df):
        def transform_string(s):
            part1 = str(int(s[:2]))
            part2 = str(int(s[2:4])) + s[4]
            part3 = str(int(s[5:7])) + s[7]
            part4 = s[-1]

            return f"{part1} {part2} {part3} {part4}"

        # Create 'geometry' column once
        df['geometry'] = [Point(e, n) for e, n in zip(df['Easting'], df['Northing'])]

        # Group by 'Conc' to create polygons
        polygons = (
            df.groupby('Conc')
            .apply(lambda x: Polygon(zip(x['Easting'], x['Northing'])))
            .reset_index(name='geometry')
        )

        # Add centroid and label in one step
        polygons['centroid'] = polygons['geometry'].apply(lambda geom: geom.centroid)
        polygons['label'] = polygons['Conc'].apply(transform_string)

        df_new = polygons


        # df['geometry'] = df.apply(lambda row: Point(row['Easting'], row['Northing']), axis=1)
        # used_fields = df[['Conc', 'Easting', 'Northing', 'geometry']]
        # used_fields['geometry'] = used_fields.apply(lambda row: Point(row['Easting'], row['Northing']), axis=1)
        # polygons = used_fields.groupby('Conc').apply(lambda x: Polygon(zip(x['Easting'], x['Northing']))).reset_index()
        # merged_data = used_fields.merge(polygons, on='Conc')
        # merged_data = merged_data.drop('geometry', axis=1).rename(columns={0: 'geometry'})
        # df_new = merged_data.groupby('Conc').apply(lambda x: Polygon(zip(x['Easting'], x['Northing']))).reset_index()
        # df_new.columns = ['Conc', 'geometry']
        # df_new['centroid'] = df_new.apply(lambda x: x['geometry'].centroid, axis=1)
        # df_new['label'] = df_new.apply(lambda x: transform_string(x['Conc']), axis=1)
        return df_new

    def utm_to_latlon_row(row, utm_zone=12, hemisphere='N'):
        utm_proj = f'+proj=utm +zone={utm_zone} +{hemisphere} +ellps=WGS84 +datum=WGS84 +units=m +no_defs'
        transformer = pyproj.Transformer.from_proj(
            pyproj.Proj(utm_proj),
            pyproj.Proj('+proj=longlat +datum=WGS84 +no_defs'),
            always_xy=True
        )
        try:
            lon, lat = transformer.transform(row['X'], row['Y'])
        except KeyError:
            lon, lat = transformer.transform(row['Easting'], row['Northing'])
        return pd.Series({'Latitude': lat, 'Longitude': lon})

    def processSections():
        conn = sqlite3.connect(r'C:\Work\databases\Board_DB_Plss_Sections.db')
        query = f"""SELECT * FROM BaseData"""
        plats = pd.read_sql(query, conn)
        plats = geometryTransform(plats)
        return plats


    plats_all_df = processSections()
    plats_all = ['3205S03WU', '3107S23ES', '1222S16ES', '2412S23ES', '1603S06WU', '1109S16ES', '0802S01EU', '1712S24ES', '1809S16ES', '1410S18ES', '3412S01ES', '1002S03WU', '1505S03WU', '3302S03WU', '2009S16ES', '1505S05WU', '0215S03ES', '1605S04WU', '0716S26ES', '1715S21ES', '0605S03WU', '2702S01EU', '0403S02EU', '1303S06WU', '2222S17ES', '3240S25ES', '3403S04WU', '1210S24ES', '0709S19ES', '0805S03WU', '2625S16ES', '1409S18ES', '1908S17ES', '1809S21ES', '0612S25ES', '3020S04ES', '0622S17ES', '2204S01WU', '0404S01EU', '0302S01WU', '1210S23ES', '2103S01EU', '1609S16ES', '2105S04WU', '2840S25ES', '1804S02WU', '2112S23ES', '3004S01WU', '0503S02WU', '0109S15ES', '2230S23ES', '2202S02WU', '2002S02WU', '2901S03WU', '3203S02WU', '2503S05WU', '3409S22ES', '0303S01WU', '0804S05WU', '2307S23ES', '2609S20ES', '1807S24ES', '1902S01EU', '2404S02WU', '1005S03WU', '2425S16ES', '2003S01WU', '0508S25ES', '1607S20ES', '2608S15ES', '2102S04WU', '1204S02EU', '1504S04WU', '0204S01EU', '2501S03WU', '1641S24ES', '2009S17ES', '1904S04WU', '1409S21ES', '1134S24ES', '2504S03WU', '0508S23ES', '1209S16ES', '2203S03WU', '3405S06WU', '2812S24ES', '1322S17ES', '2103S04WU', '2601S02WU', '3503S02EU', '2309S25ES', '1111S24ES', '2502S05WU', '0811S25ES', '3302S05WU', '2505S06WU', '3125S20ES', '0505S05WU', '1205S02EU', '1203S05WU', '0309S25ES', '1709S22ES', '2316S25ES', '1408S21ES', '2203S04WU', '0302S02WU', '0341S24ES', '3304S05WU', '1540S24ES', '2508S21ES', '2304S05WU', '3109S21ES', '0204S02WU', '1309S19ES', '1908S22ES', '0504S04WU', '2602S04WU', '0609S22ES', '2507S21ES', '1302S02WU', '1007S22ES', '2004S01EU', '1902S02WU', '3608S21ES', '0422S17ES', '3503S01WU', '1940S25ES', '2501S01WU', '0207S22ES', '2104S02WU', '0704S02WU', '0204S02EU', '1904S03WU', '2907S20ES', '0603S02EU', '2003S02EU', '2508S16ES', '2112S24ES', '0611S25ES', '3004S05WU', '2603S06WU', '1612S24ES', '1708S20ES', '1004S02WU', '2429S21ES', '1312S23ES', '3410S22ES', '1812S25ES', '2426S19ES', '2803S04WU', '2402S01EU', '3007S24ES', '2603S04WU', '2205S03WU', '2309S22ES', '0710S25ES', '1604S01WU', '1604S03WU', '1309S17ES', '1002S01EU', '3607S24ES', '1010S21ES', '1308S24ES', '1926S20ES', '2030S23ES', '2125S19ES', '2611S23ES', '3310S24ES', '0707S23ES', '2709S21ES', '3340S25ES', '2704S01EU', '1610S24ES', '0211S24ES', '2004S04WU', '1202S04WU', '2241S25ES', '1204S01EU', '0802S03WU', '3103S02WU', '1126S19ES', '1402S03WU', '1906S23ES', '0303S04WU', '3010S25ES', '1403S03WU', '2108S17ES', '1810S24ES', '2109S18ES', '3411S23ES', '2212S23ES', '1311S22ES', '2801S01EU', '3433S24ES', '2303S03WU', '0510S24ES', '1704S02WU', '1822S17ES', '1003S01EU', '1041S25ES', '2804S01WU', '0407S20ES', '1802S01WU', '0602S02WU', '3102S01WU', '1711S24ES', '2301S03WU', '1104S01EU', '2111S23ES', '2512S23ES', '0709S23ES', '3525S19ES', '2908S16ES', '3609S24ES', '3540S24ES', '2212S24ES', '3401S01EU', '0603S01WU', '1207S24ES', '2922S18ES', '1902S04WU', '2804S03EU', '2703S03WU', '2522S16ES', '0109S20ES', '3122S18ES', '2602S01EU', '0902S01WU', '2403S03WU', '3605S04WU', '1716S26ES', '2509S17ES', '1312S24ES', '2806S23ES', '3003S05WU', '0611S24ES', '1510S24ES', '1307S19ES', '0905S03WU', '3022S18ES', '0212S24ES', '0906S04WU', '2940S25ES', '1902S03WU', '1803S03WU', '3109S16ES', '2326S19ES', '3007S23ES', '0334S24ES', '3108S16ES', '3507S23ES', '1002S04WU', '0108S21ES', '2407S19ES', '1104S02WU', '3308S21ES', '3204S02EU', '1811S25ES', '2207S24ES', '2009S25ES', '2404S05WU', '1511S24ES', '3614S22ES', '3111S23ES', '0841S25ES', '2611S24ES', '0604S03WU', '1026S19ES', '2207S22ES', '2003S05WU', '0941S24ES', '0511S23ES', '0605S04WU', '1307S22ES', '0703S02EU', '0512S25ES', '2004S03EU', '0803S05WU', '2103S05WU', '1705S04WU', '1401S01WU', '2922S17ES', '3304S01EU', '0608S22ES', '2301S02WU', '2903S03WU', '2508S18ES', '3507S24ES', '1641S25ES', '2608S16ES', '2802S01WU', '3503S06WU', '3303S03WU', '0129S21ES', '1329S21ES', '2404S01EU', '2129S22ES', '2804S02WU', '3408S16ES', '3401S02WU', '1720S04ES', '3206S24ES', '1110S21ES', '2706S23ES', '2303S06WU', '3002S02EU', '0309S17ES', '0812S25ES', '1806S23ES', '3508S15ES', '0620S04ES', '0407S24ES', '0409S16ES', '0404S05WU', '3525S20ES', '0205S01EU', '2901S02WU', '1315S22ES', '2420S03ES', '1808S23ES', '2407S23ES', '3101S02WU', '3209S22ES', '0423S18ES', '3022S17ES', '1307S21ES', '3608S20ES', '1012S24ES', '2302S01WU', '2509S21ES', '1403S05WU', '1030S24ES', '0912S25ES', '1309S20ES', '2408S20ES', '2901S01EU', '3012S23ES', '0609S19ES', '3207S21ES', '0903S05WU', '1017S24ES', '2609S22ES', '1308S16ES', '2808S17ES', '1506S22ES', '0704S02EU', '2203S05WU', '1507S22ES', '0209S24ES', '1301S01WU', '3509S24ES', '3105S03WU', '1708S17ES', '2503S02WU', '1722S17ES', '3306S22ES', '2110S23ES', '2811S23ES', '1107S24ES', '2308S22ES', '2101S03WU', '2305S05WU', '2707S20ES', '0510S21ES', '0505S03EU', '1603S02WU', '1908S16ES', '1508S24ES', '1304S03WU', '0509S19ES', '1109S25ES', '2802S01EU', '2601S01WU', '2709S24ES', '3403S05WU', '0812S24ES', '3211S25ES', '0929S22ES', '2801S01WU', '0707S20ES', '1504S02EU', '2808S18ES', '3106S22ES', '2104S05WU', '2408S21ES', '2802S03WU', '2907S22ES', '1207S23ES', '0809S17ES', '1203S02WU', '1511S23ES', '1109S17ES', '2111S24ES', '2007S24ES', '0704S05WU', '0305S02EU', '3110S24ES', '0412S23ES', '0226S20ES', '3603S02WU', '1110S18ES', '1511S25ES', '2503S03WU', '0222S17ES', '1209S17ES', '2405S05WU', '3508S22ES', '3004S02EU', '0504S02EU', '1102S01WU', '3015S23ES', '0229S21ES', '0420S04ES', '2902S01WU', '0404S01WU', '2603S03WU', '2502S03WU', '1712S23ES', '2008S16ES', '0608S20ES', '0608S24ES', '1311S24ES', '2508S22ES', '1426S20ES', '0503S03WU', '1630S23ES', '2503S01WU', '2026S20ES', '2520S03ES', '1502S01WU', '1602S01WU', '0611S23ES', '3001S01EU', '2416S25ES', '0909S18ES', '3107S20ES', '0805S03EU', '2725S20ES', '3222S17ES', '1722S18ES', '0909S20ES', '1306S22ES', '2115S23ES', '0908S21ES', '1503S05WU', '3112S24ES', '1408S22ES', '0810S24ES', '2505S04WU', '0209S22ES', '3601S01WU', '0504S03WU', '2311S22ES', '1823S01WS', '0204S05WU', '2203S06WU', '1404S01WU', '2404S02EU', '1602S01EU', '0409S25ES', '0923S18ES', '0108S22ES', '2808S22ES', '2209S21ES', '3221S17ES', '3410S24ES', '3308S16ES', '3509S25ES', '1816S26ES', '1504S02WU', '2611S22ES', '0304S04WU', '1215S22ES', '0209S16ES', '1606S22ES', '2206S22ES', '2604S05WU', '3408S20ES', '1409S15ES', '3304S03WU', '2403S01EU', '2104S03WU', '3606S22ES', '1704S03EU', '1823S18ES', '0304S01EU', '0409S21ES', '0107S19ES', '0412S24ES', '0508S22ES', '1409S25ES', '1911S24ES', '2702S02WU', '1907S25ES', '3421S17ES', '1908S21ES', '0103S02WU', '2701S01WU', '0626S17ES', '0705S03WU', '2309S17ES', '0405S05WU', '0506S05WU', '2308S17ES', '2001S01WU', '0808S22ES', '0103S04WU', '1904S02WU', '3225S20ES', '2205S05WU', '2805S04WU', '2925S19ES', '1909S20ES', '2903S01EU', '2811S24ES', '3002S01EU', '3311S24ES', '3602S01WU', '2402S04WU', '1204S04WU', '2508S15ES', '1907S23ES', '2322S17ES', '1804S01EU', '3106S23ES', '0803S01EU', '2203S01EU', '0408S21ES', '1009S22ES', '1003S05WU', '3503S05WU', '0411S22ES', '0502S02WU', '1503S01WU', '1509S15ES', '1607S24ES', '1608S24ES', '1603S01EU', '2606S19ES', '0912S24ES', '0608S25ES', '1909S25ES', '2626S19ES', '2909S25ES', '0809S21ES', '1404S02WU', '1326S19ES', '1711S23ES', '2709S22ES', '0604S01EU', '1003S03WU', '1404S01EU', '2411S23ES', '2302S04WU', '3114S23ES', '1040S24ES', '3403S02EU', '1412S24ES', '2909S16ES', '3016S26ES', '2607S24ES', '3202S01WU', '3207S24ES', '3304S04WU', '2307S22ES', '2208S16ES', '0941S25ES', '2141S25ES', '2907S24ES', '2415S22ES', '2304S01WU', '3407S20ES', '1608S22ES', '1703S01WU', '2803S02WU', '2804S01EU', '1012S23ES', '2820S04ES', '0803S04WU', '0104S02WU', '2409S25ES', '2702S05WU', '0803S02WU', '3306S24ES', '0107S24ES', '2603S02EU', '3601S03WU', '1804S02EU', '0204S06WU', '0408S22ES', '2312S24ES', '0105S02EU', '2909S22ES', '1309S18ES', '2111S22ES', '1503S04WU', '2009S20ES', '3603S01WU', '1709S20ES', '3422S17ES', '2506S22ES', '0604S05WU', '2907S21ES', '2005S04WU', '1205S04WU', '2303S01EU', '0512S24ES', '0309S18ES', '0805S05WU', '1303S01WU', '0202S01WU', '2904S03EU', '2012S24ES', '0411S25ES', '2412S22ES', '2704S03WU', '2522S17ES', '2202S01EU', '3604S01EU', '2409S24ES', '1409S17ES', '0808S23ES', '1404S04WU', '2304S04WU', '3610S24ES', '2904S01WU', '1605S03WU', '0609S25ES', '0604S02EU', '2011S25ES', '3506S19ES', '2604S04WU', '0103S05WU', '3603S04WU', '1723S01WS', '1508S16ES', '0404S04WU', '1803S02WU', '2211S25ES', '2302S02WU', '2007S25ES', '2403S05WU', '3307S21ES', '2006S23ES', '0110S23ES', '2708S18ES', '3607S21ES', '0202S02WU', '1204S02WU', '0702S03WU', '0207S24ES', '2407S21ES', '1609S21ES', '2002S01EU', '1829S22ES', '3603S01EU', '1726S20ES', '2509S15ES', '0908S20ES', '1505S04WU', '3602S01EU', '0405S03WU', '1105S05WU', '2705S05WU', '0106S06WU', '3303S01EU', '0904S01EU', '0809S22ES', '0605S03EU', '0209S15ES', '0105S01EU', '1941S25ES', '1111S23ES', '3608S16ES', '3501S03WU', '0526S17ES', '1302S03WU', '2209S24ES', '3003S02EU', '3301S03WU', '0826S20ES', '2602S01WU', '0402S01EU', '1408S24ES', '3103S03WU', '0109S24ES', '3303S05WU', '2204S04WU', '3308S25ES', '3440S24ES', '0904S02EU', '1502S02WU', '3607S22ES', '0109S18ES', '2204S03WU', '3409S20ES', '1408S20ES', '2804S05WU', '1501S01WU', '2511S24ES', '0504S01WU', '1706S23ES', '2110S24ES', '1504S01WU', '1822S18ES', '1002S02WU', '1404S02EU', '1604S02WU', '2902S03WU', '2308S21ES', '0823S18ES', '0802S01WU', '1729S22ES', '3611S23ES', '2204S05WU', '1304S02EU', '1541S25ES', '0203S03WU', '0341S25ES', '1102S02WU', '1508S20ES', '3108S23ES', '2708S16ES', '1412S23ES', '3611S24ES', '3604S04WU', '2402S02WU', '2605S06WU', '1704S01EU', '1523S17ES', '0304S03WU', '3304S03EU', '1526S19ES', '2609S24ES', '0602S03WU', '1303S02WU', '1503S06WU', '1109S20ES', '0904S01WU', '1929S22ES', '2309S20ES', '2608S22ES', '1208S20ES', '3010S23ES', '0704S01EU', '0807S22ES', '3510S22ES', '0907S24ES', '3410S23ES', '1529S22ES', '2508S17ES', '0815S23ES', '0529S22ES', '3208S18ES', '2510S23ES', '2703S02EU', '2907S23ES', '3502S02WU', '3101S03WU', '1811S23ES', '0207S21ES', '0820S04ES', '0915S23ES', '2711S25ES', '1612S23ES', '1004S02EU', '0930S23ES', '2803S02EU', '0712S25ES', '0904S05WU', '3103S04WU', '2909S20ES', '3406S24ES', '1503S02EU', '1703S01EU', '0309S15ES', '3009S17ES', '2209S20ES', '3508S20ES', '3104S02EU', '1209S21ES', '2808S21ES', '2807S20ES', '3407S22ES', '0407S22ES', '1011S24ES', '3409S24ES', '1904S02EU', '2903S02WU', '3009S21ES', '1211S23ES', '2541S24ES', '3503S02WU', '3307S23ES', '2203S02EU', '1306S05WU', '0110S24ES', '3208S20ES', '0809S20ES', '3604S03WU', '0607S23ES', '1103S02WU', '1809S18ES', '1430S23ES', '0904S04WU', '1905S04WU', '1223S17ES', '1008S21ES', '2509S24ES', '3609S15ES', '0211S22ES', '0103S06WU', '1904S01WU', '2340S24ES', '3040S25ES', '2409S18ES', '0703S01WU', '0441S25ES', '2902S04WU', '1509S20ES', '2608S17ES', '1208S23ES', '1623S17ES', '0541S25ES', '3208S19ES', '2703S05WU', '1702S04WU', '3303S06WU', '2130S23ES', '0409S22ES', '0107S22ES', '2111S25ES', '1709S16ES', '1103S05WU', '1709S21ES', '1509S18ES', '2602S03WU', '1305S05WU', '1803S02EU', '2604S02WU', '2806S22ES', '2802S02WU', '1709S18ES', '2622S17ES', '0603S02WU', '1009S20ES', '1210S25ES', '2206S19ES', '1910S25ES', '3402S01EU', '3625S19ES', '3208S22ES', '0923S17ES', '1808S17ES', '1104S03WU', '1311S23ES', '3140S25ES', '0304S01WU', '1517S24ES', '1615S23ES', '1029S21ES', '3002S01WU', '1617S24ES', '2203S01WU', '1203S06WU', '3622S17ES', '0709S21ES', '1405S02EU', '3008S18ES', '1607S22ES', '0703S02WU', '1003S02EU', '0305S05WU', '2912S24ES', '2809S20ES', '2525S16ES', '2401S03WU', '1003S06WU', '0811S24ES', '2608S20ES', '1025S13ES', '0404S03WU', '2711S22ES', '1709S25ES', '1125S13ES', '1609S20ES', '1108S22ES', '0609S20ES', '2235S26ES', '2825S19ES', '1141S24ES', '2908S17ES', '0703S05WU', '0903S02EU', '0715S23ES', '0209S18ES', '1209S25ES', '1440S24ES', '3203S01WU', '1507S20ES', '3125S17ES', '1807S20ES', '1904S03EU', '2906S23ES', '0908S24ES', '0508S20ES', '0302S04WU', '0710S24ES', '1710S24ES', '1009S17ES', '3612S23ES', '2011S23ES', '1004S01WU', '3610S23ES', '3602S05WU', '1707S23ES', '2702S03WU', '1708S21ES', '0411S24ES', '1802S02EU', '0508S21ES', '2704S02EU', '0903S01EU', '2102S02WU', '2002S03WU', '0310S25ES', '1015S03ES', '2803S06WU', '0305S04WU', '1204S05WU', '0234S24ES', '0711S25ES', '1611S24ES', '2809S25ES', '0110S21ES', '0803S02EU', '3502S01EU', '1515S23ES', '3005S03WU', '3010S24ES', '1102S03WU', '2107S20ES', '0829S22ES', '1809S25ES', '0909S17ES', '0604S01WU', '3402S02WU', '0105S04WU', '2609S16ES', '1009S18ES', '0705S03EU', '3603S06WU', '0204S04WU', '2709S20ES', '1702S02WU', '2605S04WU', '2703S06WU', '2502S04WU', '3108S22ES', '3003S04WU', '2202S05WU', '1604S04WU', '3602S03WU', '1904S01EU', '2035S26ES', '1415S22ES', '0303S01EU', '3425S20ES', '2309S16ES', '2406S22ES', '2208S21ES', '1203S03WU', '1104S01WU', '1735S26ES', '2603S02WU', '2341S24ES', '0411S23ES', '2706S22ES', '1109S15ES', '1703S03WU', '2003S01EU', '3008S17ES', '0708S22ES', '2226S19ES', '1103S06WU', '0104S01WU', '1208S21ES', '3111S24ES', '0209S20ES', '3009S22ES', '3311S25ES', '0711S24ES', '0910S25ES', '1809S19ES', '0308S23ES', '1504S03WU', '0403S01WU', '0304S02WU', '3409S25ES', '2002S04WU', '3407S21ES', '2303S02EU', '2502S01WU', '2225S19ES', '1604S02EU', '3408S17ES', '3309S21ES', '2002S02EU', '2904S02WU', '0702S02WU', '1022S17ES', '1112S23ES', '1410S23ES', '0609S17ES', '3002S02WU', '0208S24ES', '2409S15ES', '0307S23ES', '1209S22ES', '1707S24ES', '2509S16ES', '3526S19ES', '1603S02EU', '1009S25ES', '0410S21ES', '1604S05WU', '1909S22ES', '0702S01EU', '0108S20ES', '2804S02EU', '2007S23ES', '3322S18ES', '1809S22ES', '1815S21ES', '0415S23ES', '1007S21ES', '2106S22ES', '2920S04ES', '2402S03WU', '2403S06WU', '2610S24ES', '3508S21ES', '1309S21ES', '1104S02EU', '2841S25ES', '0707S22ES', '2526S19ES', '0109S22ES', '3507S21ES', '1840S25ES', '1104S06WU', '1906S22ES', '1122S17ES', '2905S04WU', '2809S24ES', '2403S01WU', '2805S03WU', '2908S22ES', '2205S04WU', '1208S24ES', '1034S24ES', '2101S02WU', '3103S05WU', '0104S06WU', '3109S25ES', '0526S20ES', '1602S03WU', '1105S04WU', '0503S05WU', '3101S01WU', '0511S24ES', '2601S03WU', '0208S21ES', '1111S22ES', '2203S02WU', '2701S03WU', '1611S22ES', '1522S17ES', '1902S02EU', '2911S24ES', '1209S15ES', '2902S02WU', '2905S06WU', '0610S24ES', '3204S03WU', '3202S04WU', '1620S04ES', '0803S03WU', '3608S17ES', '0604S02WU', '3102S02EU', '2908S21ES', '1702S01WU', '3303S01WU', '2540S24ES', '1603S01WU', '1603S05WU', '1504S01EU', '1908S18ES', '0406S03WU', '2011S24ES', '0326S20ES', '3407S24ES', '1110S19ES', '1202S03WU', '1841S25ES', '1303S03WU', '1804S04WU', '2707S24ES', '3303S02EU', '2708S22ES', '2109S21ES', '0305S03WU', '1912S24ES', '2709S18ES', '3206S22ES', '2104S02EU', '2201S03WU', '0722S17ES', '1004S03WU', '2309S21ES', '2103S06WU', '1609S18ES', '1607S23ES', '2810S24ES', '1009S16ES', '2109S17ES', '2422S17ES', '2204S02WU', '1806S22ES', '2308S16ES', '0108S24ES', '1207S21ES', '3203S04WU', '0112S24ES', '2802S02EU', '1226S19ES', '3302S01EU', '0917S24ES', '3009S16ES', '2703S01EU', '0107S21ES', '3025S20ES', '1202S01EU', '2022S17ES', '1802S03WU', '1402S01WU', '1005S04WU', '3509S15ES', '0109S16ES', '2103S02WU', '1704S02EU', '1912S23ES', '0210S21ES', '3504S02EU', '1509S25ES', '1815S23ES', '1502S04WU', '3008S16ES', '2104S01EU', '1107S21ES', '3221S19ES', '0409S17ES', '2801S03WU', '0209S21ES', '0603S04WU', '0441S24ES', '2805S06WU', '1126S20ES', '3206S23ES', '0207S19ES', '0912S23ES', '3603S05WU', '1129S21ES', '1307S23ES', '1108S20ES', '2703S01WU', '3507S20ES', '0108S23ES', '1441S24ES', '2102S02EU', '0804S02WU', '1011S22ES', '3502S04WU', '2603S01EU', '1535S26ES', '1609S24ES', '2405S04WU', '0309S21ES', '3511S23ES', '2407S22ES', '2240S24ES', '1103S01WU', '2507S20ES', '3509S22ES', '0804S02EU', '0302S03WU', '2009S22ES', '3307S24ES', '1003S04WU', '3310S22ES', '0911S25ES', '2503S02EU', '2606S22ES', '0607S22ES', '0503S04WU', '3104S03WU', '2903S05WU', '0308S20ES', '3104S04WU', '1608S21ES', '0503S01WU', '1110S23ES', '1507S21ES', '1005S05WU', '3311S23ES', '0510S25ES', '3503S01EU', '2641S24ES', '2411S22ES', '0103S03WU', '2206S23ES', '2315S03ES', '0902S02WU', '1804S03WU', '2211S24ES', '3608S22ES', '2211S23ES', '2803S03WU', '2605S05WU', '1109S21ES', '1707S22ES', '1406S22ES', '2702S01WU', '3222S18ES', '0326S19ES', '3440S25ES', '2202S03WU', '3409S15ES', '1512S23ES', '1629S22ES', '3408S21ES', '0406S19ES', '3306S23ES', '3101S01EU', '3604S02EU', '2702S04WU', '1405S04WU', '2906S22ES', '0509S25ES', '2625S19ES', '3409S21ES', '0603S05WU', '2104S04WU', '1312S22ES', '0520S04ES', '0502S02EU', '0603S01EU', '3203S02EU', '0209S25ES', '1009S24ES', '3308S22ES', '1410S24ES', '3004S04WU', '2808S16ES', '2725S16ES', '1907S22ES', '0304S06WU', '1103S01EU', '3006S23ES', '1309S24ES', '2408S16ES', '1508S23ES', '3402S01WU', '2302S01EU', '3607S20ES', '3108S19ES', '2408S17ES', '1911S23ES', '3408S25ES', '3203S05WU', '0809S18ES', '2610S22ES', '0602S02EU', '1803S04WU', '2510S24ES', '2307S21ES', '0523S18ES', '1503S01EU', '0605S05WU', '3303S02WU', '0112S23ES', '0503S02EU', '0323S17ES', '3011S23ES', '1807S25ES', '1411S22ES', '0905S04WU', '1908S23ES', '3407S23ES', '1402S04WU', '2507S24ES', '2904S04WU', '2126S20ES', '2208S20ES', '1808S22ES', '3301S01WU', '1409S24ES', '0209S17ES', '2404S03WU', '3108S17ES', '2725S19ES', '2708S17ES', '0307S21ES', '1304S01WU', '2409S22ES', '3026S20ES', '1403S01EU', '1915S23ES', '1309S16ES', '1107S22ES', '0303S03WU', '3607S23ES', '1203S01EU', '2008S22ES', '1203S01WU', '0703S03WU', '1708S22ES', '2209S15ES', '0203S05WU', '2005S05WU', '3609S21ES', '1909S21ES', '2008S18ES', '2440S24ES', '1907S20ES', '0102S02WU', '1209S24ES', '1403S02WU', '2004S01WU', '2608S18ES', '3508S16ES', '3502S01WU', '0615S23ES', '2303S04WU', '3109S22ES', '1202S05WU', '0511S25ES', '2830S23ES', '1209S18ES', '1901S01WU', '1903S05WU', '0809S25ES', '2511S22ES', '2516S25ES', '2009S18ES', '2612S23ES', '3601S02WU', '1504S05WU', '0626S20ES', '0207S23ES', '1903S02WU', '2311S23ES', '3521S17ES', '3404S02EU', '0706S05WU', '3501S02WU', '3507S22ES', '2503S04WU', '2807S22ES', '2804S03WU', '2208S22ES', '2925S20ES', '3504S03WU', '2305S04WU', '1108S21ES', '1407S23ES', '1916S26ES', '2422S16ES', '1702S01EU', '2807S23ES', '2903S01WU', '2904S03WU', '0110S20ES', '1803S01EU', '2808S20ES', '0907S20ES', '0905S05WU', '1903S01WU', '1220S03ES', '3210S24ES', '2740S24ES', '2410S24ES', '0506S19ES', '0404S02WU', '1105S02EU', '3007S25ES', '0409S20ES', '3208S25ES', '1208S22ES', '0722S18ES', '2022S18ES', '3211S24ES', '2710S22ES', '3510S24ES', '1811S24ES', '0704S04WU', '3403S02WU', '1826S20ES', '3308S17ES', '2201S01WU', '1240S24ES', '1809S17ES', '1512S24ES', '2006S22ES', '0205S05WU', '3110S25ES', '2202S01WU', '0504S02WU', '1203S04WU', '1140S24ES', '0507S22ES', '0804S03WU', '2801S02WU', '2902S01EU', '2603S01WU', '0609S16ES', '0302S01EU', '1041S24ES', '0723S18ES', '1403S01WU', '2303S02WU', '2508S20ES', '2108S16ES', '1820S04ES', '1102S04WU', '0607S20ES', '2825S20ES', '2107S24ES', '2406S21ES', '0709S25ES', '0310S24ES', '1515S03ES', '1003S01WU', '3004S02WU', '1530S23ES', '1703S05WU', '1206S04WU', '3309S24ES', '3505S06WU', '1123S17ES', '1229S21ES', '2511S23ES', '0909S21ES', '0510S22ES', '2604S01EU', '1222S17ES', '1704S01WU', '0402S01WU', '2302S03WU', '3041S25ES', '2802S04WU', '2207S23ES', '1004S05WU', '1426S19ES', '3302S04WU', '1526S20ES', '1626S20ES', '1303S04WU', '0904S03WU', '1807S23ES', '0505S03WU', '2004S02EU', '3108S18ES', '1209S19ES', '3325S19ES', '1914S15ES', '0502S01WU', '0303S02EU', '2404S01WU', '1520S04ES', '0112S22ES', '0103S01EU', '2103S01WU', '0408S23ES', '1602S02EU', '3606S19ES', '1704S04WU', '2609S17ES', '3003S01EU', '3204S03EU', '3207S20ES', '1812S24ES', '0807S23ES', '2711S24ES', '1604S01EU', '1904S05WU', '1507S24ES', '0707S25ES', '1320S03ES', '2304S02WU', '1705S05WU', '1323S17ES', '0704S03EU', '0404S02EU', '1603S03WU', '2604S03WU', '1008S24ES', '2504S02EU', '2803S05WU', '2502S01EU', '1715S23ES', '2910S24ES', '0105S05WU', '1315S03ES', '0908S22ES', '3207S23ES', '1608S23ES', '1304S02WU', '2640S24ES', '2803S01EU', '2002S01WU', '0507S24ES', '2604S01WU', '3004S03EU', '3009S25ES', '0312S24ES', '0407S23ES', '1302S05WU', '3003S03WU', '1003S02WU', '0111S22ES', '2304S02EU', '3525S16ES', '1103S02EU', '0303S05WU', '1608S20ES', '3401S01WU', '1541S24ES', '0205S02EU', '2329S21ES', '2411S24ES', '2109S25ES', '2101S01WU', '0103S01WU', '3203S03WU', '0210S24ES', '0702S04WU', '1506S19ES', '0911S23ES', '0504S05WU', '2107S23ES', '2312S23ES', '3209S24ES', '3002S03WU', '1308S22ES', '0205S04WU', '3504S05WU', '0715S21ES', '0126S19ES', '2509S25ES', '1809S20ES', '0107S23ES', '0903S04WU', '2921S19ES', '0822S17ES', '3608S18ES', '2907S25ES', '2210S24ES', '1502S01EU', '2709S15ES', '1510S23ES', '3640S24ES', '2809S22ES', '1812S23ES', '1308S23ES', '0705S05WU', '2104S01WU', '1422S17ES', '1905S03WU', '1403S02EU', '0204S03WU', '3202S02WU', '1303S05WU', '2709S16ES', '3201S01EU', '2310S23ES', '0807S24ES', '1409S20ES', '3002S04WU', '2109S20ES', '2908S18ES', '0311S22ES', '0622S18ES', '2401S02WU', '3308S18ES', '1408S16ES', '0804S01WU', '0304S02EU', '0702S02EU', '3506S22ES', '3522S17ES', '1309S22ES', '2109S22ES', '0402S03WU', '3103S01WU', '0806S04WU', '0308S22ES', '0804S04WU', '2926S20ES', '1408S17ES', '2408S15ES', '0203S04WU', '0202S04WU', '1602S02WU', '2209S25ES', '2107S21ES', '1402S01EU', '2910S23ES', '0104S04WU', '0403S03WU', '3208S16ES', '0405S04WU', '0709S17ES', '2707S23ES', '1308S21ES', '3207S22ES', '0111S23ES', '2402S05WU', '2404S04WU', '2504S01EU', '1008S22ES', '1107S19ES', '0311S23ES', '1503S03WU', '1415S03ES', '2710S24ES', '1703S02EU', '2510S22ES', '1602S04WU', '1810S25ES', '2509S20ES', '3202S05WU', '2003S03WU', '2409S16ES', '3005S04WU', '2914S15ES', '0702S01WU', '3305S03WU', '1909S17ES', '3208S17ES', '1707S25ES', '1204S01WU', '2502S02WU', '1302S04WU', '1007S24ES', '0523S17ES', '0509S16ES', '3122S17ES', '2310S24ES', '1202S02WU', '2220S04ES', '3008S21ES', '2803S01WU', '0708S23ES', '0211S23ES', '3608S15ES', '3501S01WU', '2804S04WU', '0906S19ES', '2712S23ES', '2807S24ES', '3408S18ES', '3203S01EU', '3209S25ES', '3102S04WU', '2811S25ES', '0208S20ES', '0522S17ES', '0608S23ES', '2809S21ES', '3209S21ES', '2307S24ES', '1407S22ES', '3201S01WU', '0909S22ES', '1008S20ES', '2202S04WU', '1407S21ES', '1802S01EU', '2807S21ES', '0720S04ES', '0805S04WU', '2701S02WU', '3301S01EU', '2408S22ES', '0503S01EU', '3625S16ES', '2903S02EU', '0803S01WU', '1903S01EU', '0102S01WU', '3509S20ES', '1010S19ES', '2209S16ES', '2402S01WU', '2707S22ES', '2912S23ES', '3108S21ES', '3201S03WU', '2201S02WU', '0707S24ES', '3007S22ES', '1107S23ES', '0802S02EU', '2709S25ES', '0309S16ES', '0911S24ES', '2304S03WU', '1509S17ES', '3504S01EU', '1303S01EU', '0807S20ES', '1309S25ES', '1920S04ES', '0203S06WU', '2503S06WU', '1903S02EU', '3608S24ES', '3503S03WU', '0903S01WU', '2809S19ES', '0110S25ES', '2003S04WU', '2107S22ES', '1704S03WU', '2309S24ES', '0741S25ES', '3322S17ES', '1605S05WU', '1002S01WU', '2704S05WU', '0808S24ES', '3202S01EU', '0911S22ES', '3403S01EU', '2607S20ES', '3406S21ES', '1703S02WU', '1907S24ES', '2004S02WU', '0602S01WU', '2302S05WU', '0406S04WU', '3207S25ES', '1402S05WU', '2504S05WU', '3008S22ES', '3102S03WU', '1411S24ES', '3510S23ES', '2102S01WU', '0102S01EU', '0102S03WU', '0208S22ES', '2105S05WU', '1704S05WU', '3508S17ES', '2409S17ES', '2304S01EU', '2106S23ES', '2004S03WU', '1204S06WU', '1008S23ES', '0208S23ES', '0903S02WU', '0602S01EU', '0423S17ES', '3309S22ES', '0908S23ES', '1805S03WU', '1215S03ES', '1108S23ES', '2504S01WU', '1101S01WU', '2606S21ES', '3310S23ES', '1307S20ES', '3102S01EU', '0926S20ES', '0203S02WU', '2822S17ES', '3103S01EU', '2901S01WU', '2609S15ES', '1307S24ES', '2120S04ES', '0609S18ES', '1903S03WU', '0726S20ES', '3025S17ES', '3107S25ES', '0009S19ES', '3102S02WU', '0641S25ES', '2303S05WU', '0709S20ES', '0111S24ES', '3411S25ES', '1001S01WU', '2507S22ES', '0504S01EU', '0408S24ES', '2710S23ES', '1211S22ES', '0704S01WU', '3508S18ES', '3111S25ES', '0123S17ES', '2410S23ES', '1910S24ES', '0922S17ES', '0307S24ES', '1108S24ES', '3001S01WU', '0507S20ES', '3102S05WU', '3511S22ES', '3302S01WU', '1404S03WU', '1808S24ES', '3602S04WU', '2122S17ES', '0205S03WU', '2911S23ES', '3403S01WU', '3303S04WU', '2308S20ES', '0502S01EU', '1509S22ES', '0115S22ES', '0309S20ES', '1435S26ES', '1723S18ES', '1404S05WU', '1010S24ES', '2001S02WU', '0902S04WU', '2904S01EU', '0215S22ES', '0508S24ES', '0122S17ES', '3402S04WU', '0705S04WU', '3201S02WU', '3308S20ES', '0502S03WU', '2704S04WU', '3404S05WU', '1511S22ES', '2911S25ES', '0709S18ES', '0703S04WU', '0403S02WU', '0408S20ES', '0703S01EU', '2703S04WU', '0802S04WU', '3506S21ES', '2704S02WU', '0104S03WU', '1409S22ES', '3108S25ES', '0708S24ES', '1004S04WU', '2109S24ES', '1911S25ES', '0403S01EU', '0409S18ES', '0403S05WU', '0607S24ES', '1109S18ES', '1023S17ES', '2211S22ES', '1304S06WU', '2012S23ES', '2409S21ES', '1026S20ES', '0303S02WU', '0308S24ES', '1509S16ES', '1508S17ES', '2509S22ES', '3302S02WU', '1903S04WU', '1502S03WU', '0310S21ES', '2008S23ES', '1115S22ES', '0910S24ES', '3404S01EU', '0708S25ES', '1608S17ES', '1402S02WU', '0509S17ES', '0712S23ES', '0610S22ES', '2122S18ES', '0808S20ES', '2707S21ES', '3204S01EU', '2015S23ES', '0512S23ES', '3214S23ES', '0104S02EU', '0109S17ES', '1104S05WU', '2504S02WU', '3506S23ES', '0309S22ES', '3011S24ES', '1730S23ES', '2204S02EU', '1209S20ES', '3422S18ES', '0629S22ES', '3106S24ES', '2226S20ES', '3104S03EU', '2309S15ES', '1207S22ES', '1804S03EU', '2108S18ES', '2103S03WU', '0505S04WU', '1609S17ES', '1807S22ES', '0509S21ES', '1212S22ES', '0402S02WU', '2822S18ES', '1009S15ES', '1909S16ES', '1507S23ES', '0609S21ES', '2525S19ES', '2603S05WU', '1403S06WU', '3511S24ES', '1305S04WU', '1609S22ES', '1030S23ES', '0935S26ES', '0402S04WU', '2303S01WU', '1212S23ES', '3502S05WU', '2204S01EU', '1503S02WU', '1212S24ES', '0920S04ES', '1322S16ES', '3110S23ES', '1902S01WU', '0605S02EU', '3404S03WU', '3402S03WU', '2905S03WU', '0904S02WU', '3609S25ES', '3011S25ES', '1020S04ES', '1609S25ES', '0102S04WU', '1429S21ES', '0604S03EU', '0307S22ES', '2403S02WU', '3621S17ES', '2209S17ES', '2607S22ES', '2407S24ES', '2607S21ES', '0212S23ES', '1803S01WU', '3404S04WU', '0509S18ES', '0210S25ES', '0903S03WU', '0426S20ES', '0623S18ES', '3502S03WU', '2608S21ES', '2108S21ES', '1423S17ES', '2004S05WU', '2103S02EU', '2029S22ES', '2708S20ES', '2325S16ES', '0109S21ES', '3208S21ES', '0410S25ES', '3004S01EU', '0134S24ES', '2209S22ES', '2108S22ES', '1802S04WU', '1611S25ES', '2209S18ES', '3003S01WU', '3402S05WU', '2904S05WU', '1417S24ES', '2902S02EU', '3012S24ES', '2604S02EU', '1302S01WU', '2810S23ES', '2505S05WU', '1508S22ES', '1702S03WU', '1909S18ES', '2610S23ES', '0711S23ES', '3104S05WU', '1309S15ES', '0902S03WU', '3604S05WU', '1803S05WU', '0609S23ES', '2241S24ES', '1011S23ES', '1115S03ES', '3006S22ES', '2020S04ES', '3007S20ES', '1304S05WU', '1110S24ES', '2007S22ES', '3609S20ES', '0830S23ES', '0410S24ES', '0403S04WU', '2311S24ES', '2041S25ES', '2010S24ES', '1508S21ES', '2609S25ES', '3008S23ES', '3204S05WU', '0729S22ES', '1901S02WU', '1706S22ES', '0203S01EU', '3512S23ES', '1006S19ES', '2704S01WU', '0709S16ES', '2105S03WU', '1622S17ES', '0802S02WU', '0203S01WU', '2504S04WU', '1606S19ES', '0704S03WU', '2307S20ES', '2809S18ES', '3605S06WU', '0809S16ES', '3003S02WU', '1705S03WU', '0612S24ES', '2809S16ES', '1204S03WU', '2503S01EU', '1741S25ES', '0322S17ES', '2306S22ES', '0210S19ES', '1015S23ES', '0202S01EU', '0907S22ES', '0507S23ES', '1104S04WU', '3301S02WU', '1407S20ES', '0811S23ES', '3307S22ES', '1509S24ES', '1310S23ES', '0505S02EU', '0610S25ES', '3211S23ES', '1802S02WU', '1109S24ES', '1703S04WU', '2909S21ES', '0610S21ES', '1205S05WU', '0812S23ES', '1241S24ES', '3504S04WU', '3602S02WU', '1805S04WU', '2008S17ES', '1409S16ES', '1709S17ES', '1922S17ES', '3325S20ES', '3606S21ES', '2722S17ES', '3509S21ES', '0808S21ES', '1405S05WU', '2409S20ES', '0612S23ES', '1529S21ES', '2507S23ES', '1635S26ES', '1103S04WU', '3403S03WU', '0223S17ES', '0709S22ES', '1112S24ES', '1304S04WU', '3107S24ES', '2501S02WU', '0712S24ES', '1611S23ES', '0604S04WU', '1922S18ES', '2003S02WU', '3533S24ES', '2210S23ES', '0126S16ES', '2108S20ES', '2135S26ES', '1207S19ES', '0810S25ES', '1004S01EU', '0907S23ES', '1403S04WU', '3210S23ES', '1411S23ES', '2904S02EU', '3202S03WU', '0603S03WU', '3406S23ES', '2441S24ES', '1909S23ES', '1806S05WU', '3610S22ES', '1009S21ES', '1708S23ES', '0312S23ES', '0304S05WU', '1004S06WU', '1202S01WU', '2506S21ES', '3126S20ES', '3408S22ES', '0909S16ES', '1804S01WU', '2102S01EU', '2009S21ES', '3103S02EU', '0412S25ES', '2602S05WU', '1702S02EU', '3204S04WU', '1341S24ES', '3626S19ES', '0202S03WU', '0804S01EU', '3425S16ES', '0226S19ES', '1711S25ES', '3202S02EU', '1007S23ES', '0902S01EU', '3425S19ES', '2102S03WU', '1102S01EU', '2711S23ES', '0308S21ES', '0141S24ES', '1407S24ES', '3611S22ES', '0509S22ES', '2208S17ES', '2216S25ES', '3411S24ES', '1708S24ES', '2903S04WU', '0705S02EU', '2207S21ES', '3001S02WU', '2703S02WU', '2602S02WU', '1109S22ES', '3004S03WU', '3321S17ES', '2309S18ES', '3603S02EU', '1310S24ES', '3307S20ES', '2005S03WU', '1103S03WU', '3205S06WU', '3603S03WU', '2007S20ES', '1804S05WU', '0311S24ES', '0509S20ES', '0909S25ES', '1211S24ES', '1509S21ES', '2812S23ES', '2403S04WU', '1809S23ES', '1302S01EU', '3403S06WU', '3309S25ES', '0515S23ES', '2609S21ES', '3406S22ES', '1603S04WU', '1707S20ES', '2607S23ES', '2802S05WU', '0104S01EU', '3401S03WU', '2109S16ES', '2941S25ES', '0204S01WU', '3503S04WU', '1340S24ES', '2207S20ES', '0241S24ES', '2403S02EU', '3107S22ES']
    plats_all_df = plats_all_df[plats_all_df['Conc'].isin(plats_all)]

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
    counter = 0
    all_plats = []
    grouped = df.groupby(['APINumber', 'CitingType'])
    total_things = len(grouped)
    for group_name, group_data in grouped:
        slants = group_data['Slant'].unique()
        group_data = group_data.drop(columns=['Slant'])
        max_x = group_data['X'].iloc[0] - group_data['X'].iloc[-1]
        max_y = group_data['Y'].iloc[0] - group_data['Y'].iloc[-1]
        max_distance = (max_x**2 + max_y**2)**0.5

        if '4301354249' not in group_name and 'VERTICAL' not in slants and max_distance  > 20 and counter >= 531:
            print(group_name, counter, f"""/""", total_things)
            group_data = group_data.drop_duplicates(keep="first")
            elevation = group_data['SurveySurfaceElevation'].iloc[0]
            group_data[['lat', 'lon']] = group_data.apply(utm_to_latlon_row, axis=1)
            x2, y2 = group_data['X'].mean(), group_data['Y'].mean()
            # Convert centroid column to x,y coordinates
            # Assuming centroid column contains shapely Points
            x1 = np.array([p.x for p in plats_all_df['centroid']])
            y1 = np.array([p.y for p in plats_all_df['centroid']])
            # Vectorized distance calculation
            plats_all_df['min_distance'] = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
            tot_distance = LineString([Point(group_data['X'].iloc[0],group_data['Y'].iloc[0]), Point(group_data['X'].iloc[-1],group_data['Y'].iloc[-1])]).length
            plats_all_df_filtered = plats_all_df[plats_all_df['min_distance']<= tot_distance * 2    ]
            all_plats.extend(plats_all_df_filtered['Conc'].unique())
            # print(plats_all_df_filtered['Conc'].unique())
            new_df = group_data[['MeasuredDepth', 'Inclination', 'Azimuth', 'lat', 'lon']]
            dx_df_test = SurveyProcess(
                df_referenced=new_df,
                elevation=elevation)
            try:
                kop_test = dx_df_test.find_kick_off_point(dx_df_test.true_dx)
                lp_test = dx_df_test.find_landing_point(dx_df_test.true_dx)
            except IndexError:
                print(slants)
                print(counter)
                print(dx_df_test.true_dx)
                print(foo)
            try:
                clear_df = ClearanceProcess(dx_df_test.true_dx, plats_all_df_filtered)
                processed_dx_df_test, footages_test = clear_df.clearance_data, clear_df.whole_df
            except (ValueError, IndexError):
                print(plats_all_df_filtered)
                # print(foo)
            # print(footages_test)
            # print(foo)
        counter += 1
    print(list(set(all_plats)))
def generateSampleSection():
    # Create measured depth points every 100 ft
    md = np.arange(0, 12000, 100)

    # Generate realistic inclination and azimuth profiles
    inclination = np.zeros_like(md, dtype=float)
    azimuth = np.zeros_like(md, dtype=float)

    # Vertical section (0-2000 ft)
    inclination[:20] = 0
    azimuth[:20] = 0

    # Build section (2000-4000 ft)
    build_idx = np.where((md >= 2000) & (md <= 4000))[0]
    inclination[build_idx] = np.linspace(0, 90, len(build_idx))
    azimuth[build_idx] = 45

    # Lateral section (4000-12000 ft)
    lateral_idx = np.where(md > 4000)[0]
    inclination[lateral_idx] = 90
    azimuth[lateral_idx] = 45

    # Calculate northing and easting using basic trig
    northing = np.zeros_like(md)
    easting = np.zeros_like(md)

    for i in range(1, len(md)):
        delta_md = md[i] - md[i - 1]
        avg_inc = np.radians((inclination[i] + inclination[i - 1]) / 2)
        avg_az = np.radians((azimuth[i] + azimuth[i - 1]) / 2)

        northing[i] = northing[i - 1] + delta_md * np.sin(avg_inc) * np.cos(avg_az)
        easting[i] = easting[i - 1] + delta_md * np.sin(avg_inc) * np.sin(avg_az)

    # Convert northing/easting to lat/lon (assuming starting point and rough conversion)
    start_lat = 43.0
    start_lon = -108.0
    lat_conv = 1 / 364000
    lon_conv = 1 / 288200
    lat = start_lat + northing * lat_conv
    lon = start_lon + easting * lon_conv

    # Create DataFrame
    df = pd.DataFrame({
        'MeasuredDepth': md,
        'Inclination': inclination,
        'Azimuth': azimuth,
        'Latitude': lat,
        'Longitude': lon,
        'Township': 'T38N',
        'Range': 'R89W'
    })
    # Create sections DataFrame
    section_size = 5280  # 1 mile in feet
    min_easting = min(easting)
    max_easting = max(easting)
    min_northing = min(northing)
    max_northing = max(northing)

    # Find section boundaries (round down to nearest section)
    start_easting = (min_easting // section_size) * section_size
    start_northing = (min_northing // section_size) * section_size

    # Create empty lists to store section data
    section_ids = []
    section_polygons = []

    # Generate sections and their polygons
    for i in range(2):  # 2 sections east-west
        for j in range(2):  # 2 sections north-south
            # Calculate corner coordinates for this section
            x1 = start_easting + i * section_size
            y1 = start_northing + j * section_size
            x2 = x1 + section_size
            y2 = y1 + section_size

            # Create polygon (5 points to close the rectangle)
            polygon = Polygon([
                (x1, y1),  # bottom left
                (x2, y1),  # bottom right
                (x2, y2),  # top right
                (x1, y2),  # top left
                (x1, y1)  # back to bottom left to close polygon
            ])

            section_ids.append(f'Sec {i + 1}-{j + 1} T38N-R89W')
            section_polygons.append(polygon)

    # Create sections DataFrame
    sections_df = pd.DataFrame({
        'Section_ID': section_ids,
        'Geometry': section_polygons
    })
    print(sections_df)
    print(df)
    # # Create X-Y plot with sections
    # plt.figure(figsize=(12, 8))
    #
    # # Define section dimensions (5280 ft = 1 mile = 1 section)
    # section_size = 5280
    #
    # # Calculate which sections we need based on well path extent
    # min_easting = min(easting)
    # max_easting = max(easting)
    # min_northing = min(northing)
    # max_northing = max(northing)
    #
    # # Find section boundaries (round down to nearest section)
    # start_easting = (min_easting // section_size) * section_size
    # start_northing = (min_northing // section_size) * section_size
    #
    # # Draw sections
    # for i in range(2):  # 2 sections east-west
    #     for j in range(2):  # 2 sections north-south
    #         section = Rectangle(
    #             (start_easting + i * section_size, start_northing + j * section_size),
    #             section_size, section_size,
    #             fill=False, color='gray', linestyle='--'
    #         )
    #         plt.gca().add_patch(section)
    #         # Add section labels
    #         plt.text(
    #             start_easting + i * section_size + section_size / 2,
    #             start_northing + j * section_size + section_size / 2,
    #             f'Sec {i + 1}-{j + 1}\nT38N-R89W',
    #             horizontalalignment='center',
    #             verticalalignment='center'
    #         )
    #
    # # Plot well path
    # plt.plot(easting, northing, 'b-', linewidth=2)
    # plt.scatter(easting[0], northing[0], color='green', s=100, label='Surface Location')
    # plt.scatter(easting[-1], northing[-1], color='red', s=100, label='TD')
    #
    # plt.xlabel('East-West Distance (ft)')
    # plt.ylabel('North-South Distance (ft)')
    # plt.title('Well Path - Plan View with Sections')
    # plt.grid(True)
    # plt.axis('equal')
    # plt.legend()
    #
    # # Ensure all sections are visible
    # plt.xlim(start_easting - section_size * 0.1, start_easting + section_size * 2.1)
    # plt.ylim(start_northing - section_size * 0.1, start_northing + section_size * 2.1)
    #
    # plt.show()
    #
    # print("\nWell Data Sample:")
    # print(df.head())
# Configure pandas display options
pd.set_option('display.max_columns', None)  # Show all columns when displaying DataFrames
pd.options.mode.chained_assignment = None  # Suppress chained assignment warnings
# search_py_files_for_kop(r"C:\Work\RewriteAPD")
# processAllData()
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
# dx_df_orig = pd.DataFrame(survey_data)
generateSampleSection()
print(foo)
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
used_plats = {'Conc': ['section1', 'section2'],'geometry':[Polygon(section1_coords), Polygon(section2_coords)]}
plat_df = pd.DataFrame(used_plats)
# Process survey data and calculate clearances
dx_df = SurveyProcess(
    df_referenced=dx_df_orig,
    elevation=5515)
# plat_df = plat_df.drop(columns=['centroid', 'label'])

kop = dx_df.find_kick_off_point(dx_df.true_dx)
lp = dx_df.find_landing_point(dx_df.true_dx)
print('kop', kop)
print('lp', lp)
df_test = dx_df.drilled_depths_process(dx_df.true_dx, df_depths)
clear_df = ClearanceProcess(dx_df.true_dx, plat_df, plats_adjacent)
processed_dx_df, footages = clear_df.clearance_data, clear_df.whole_df
print(processed_dx_df)
