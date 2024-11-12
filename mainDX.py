import sqlite3
from scipy.spatial import ConvexHull
from rdp import rdp
import copy
from welltrajconvert.wellbore_trajectory import *
from shapely.geometry import Point, LineString, MultiPoint, Polygon
import welleng as we
from pyproj import Proj, Geod, Transformer, transform
import matplotlib.pyplot as plt
import utm
# import ModuleAgnostic as ma
from pyproj import CRS, Proj, Transformer, CRS
import pandas as pd
import numpy as np
import math
import time
from shapely import wkt
from DXClearence import ClearanceProcess
from DXSurveys import SurveyProcess






def dict_to_interval(d):
    d = json.loads(d) if isinstance(d, str) else d
    return pd.Interval(d['left'], d['right'], closed=d['closed'])


pd.set_option('display.max_columns', None)
pd.options.mode.chained_assignment = None
conn_sample = sqlite3.connect('DX_sample.db')

query = "select * from PlatDF"
plat_df = pd.read_sql(query, conn_sample)
query = "select * from PlatAdj"
plats_adjacent = pd.read_sql(query, conn_sample)
query = "select * from DX_Original"
dx_df_orig = pd.read_sql(query, conn_sample)


query = "select * from Depths"
df_depths = pd.read_sql(query, conn_sample)
df_depths['Interval'] = df_depths['Interval'].apply(dict_to_interval)

plat_df['geometry'] = plat_df['geometry'].apply(lambda row: wkt.loads(row))
plat_df['centroid'] = plat_df['centroid'].apply(lambda row: wkt.loads(row))
plats_adjacent['geometry'] = plats_adjacent['geometry'].apply(lambda row: wkt.loads(row))
plats_adjacent['centroid'] = plats_adjacent['centroid'].apply(lambda row: wkt.loads(row))

dx_df = SurveyProcess(df_referenced = dx_df_orig, drilled_depths = df_depths,elevation = 5515, coords_type = 'latlon')

clear_df = ClearanceProcess(dx_df.df_t, plat_df, plats_adjacent)
base_dx_df_planned, planned_footages = clear_df.clearance_data, clear_df.whole_df
print(base_dx_df_planned)
