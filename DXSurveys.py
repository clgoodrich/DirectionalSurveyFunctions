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
from shapely import wkt
from datetime import datetime
from welleng.survey import SurveyHeader
from pygeomag import GeoMag
from typing import Optional, Tuple, Union

class FastSurveyHeader(SurveyHeader):
    """A fast implementation of SurveyHeader for calculating magnetic field parameters.

    This class extends SurveyHeader to provide efficient magnetic field calculations
    and date-time utilities for survey operations. It includes methods for computing
    decimal years and retrieving magnetic field information for specific locations.

    Attributes:
        latitude (float): The latitude coordinate in degrees
        longitude (float): The longitude coordinate in degrees
        altitude (float): The altitude in meters
        b_total (Optional[float]): Total magnetic field strength
        dip (Optional[float]): Magnetic dip angle
        declination (Optional[float]): Magnetic declination
        convergence (float): Grid convergence angle
        vertical_inc_limit (float): Vertical inclination limit
        vertical_section_azimuth (float): Vertical section azimuth
    """

    @staticmethod
    def get_decimal_year() -> float:
        """Calculate the current year as a decimal value.

        Returns:
            float: The current year with decimal fraction representing the
                  elapsed portion of the year (e.g., 2023.5 for mid-year)

        Example:
            >>> FastSurveyHeader.get_decimal_year()
            2023.534246575342  # For July 14, 2023
        """
        now = datetime.now()
        year_start = datetime(now.year, 1, 1)
        year_end = datetime(now.year + 1, 1, 1)
        year_length = (year_end - year_start).total_seconds()
        year_elapsed = (now - year_start).total_seconds()
        return now.year + (year_elapsed / year_length)

    @staticmethod
    def get_magnetic_field_info(
            lat: float,
            lon: float,
            altitude: float = 0,
            current_date: Optional[float] = None
    ) -> Tuple[float, float, float]:
        """Retrieve magnetic field information for a specific location.

        Args:
            lat: Latitude in degrees
            lon: Longitude in degrees
            altitude: Altitude in meters above sea level (default: 0)
            current_date: Decimal year (default: None, uses current date)

        Returns:
            Tuple containing:
                - Magnetic declination (degrees)
                - Total field intensity
                - Magnetic inclination (degrees)

        Example:
            >>> FastSurveyHeader.get_magnetic_field_info(51.5074, -0.1278)
            (-0.23, 48950.1, 66.8)  # Example values for London
        """
        if current_date is None:
            current_date = FastSurveyHeader.get_decimal_year()
        geo_mag = GeoMag()
        result = geo_mag.calculate(glat=lat, glon=lon, alt=altitude, time=current_date)
        return result.dec, result.total_intensity, result.inclination

    def _get_mag_data(self, deg: bool) -> None:
        """Process and store magnetic field data for the survey location.

        Updates the instance magnetic field attributes (b_total, dip, declination)
        and converts angular measurements to radians if required.

        Args:
            deg: If True, input values are in degrees and will be converted to radians

        Note:
            This is an internal method used to initialize magnetic field parameters.
            It modifies the instance attributes in place.
        """
        # Calculate magnetic field parameters for the current location
        declination, total_intensity, inclination = self.get_magnetic_field_info(
            lat=self.latitude,
            lon=self.longitude,
            altitude=self.altitude
        )

        # Set total field intensity if not already defined
        if self.b_total is None:
            self.b_total = total_intensity

        # Set magnetic dip angle if not already defined
        if self.dip is None:
            self.dip = -inclination  # Note: Negative convention for dip
            if not deg:
                self.dip = math.radians(self.dip)

        # Set declination if not already defined
        if self.declination is None:
            self.declination = declination
            if not deg:
                self.declination = math.radians(self.declination)

        # Convert all angular measurements to radians if input was in degrees
        if deg:
            self.dip = math.radians(self.dip)
            self.declination = math.radians(self.declination)
            self.convergence = math.radians(self.convergence)
            self.vertical_inc_limit = math.radians(self.vertical_inc_limit)
            self.vertical_section_azimuth = math.radians(self.vertical_section_azimuth)


# class FastSurveyHeader(SurveyHeader):
#     @staticmethod
#     def get_decimal_year():
#         now = datetime.now()
#         year_start = datetime(now.year, 1, 1)
#         year_end = datetime(now.year + 1, 1, 1)
#         year_length = (year_end - year_start).total_seconds()
#         year_elapsed = (now - year_start).total_seconds()
#         return now.year + (year_elapsed / year_length)
#
#     @staticmethod
#     def get_magnetic_field_info(lat, lon, altitude=0, current_date=None):
#         if current_date is None:
#             current_date = FastSurveyHeader.get_decimal_year()
#         geo_mag = GeoMag()
#         result = geo_mag.calculate(glat=lat, glon=lon, alt=altitude, time=current_date)
#         return result.dec, result.total_intensity, result.inclination
#
#     def _get_mag_data(self, deg):
#         # Use the integrated fast implementation
#         declination, total_intensity, inclination = self.get_magnetic_field_info(
#             lat=self.latitude,
#             lon=self.longitude,
#             altitude=self.altitude
#         )
#
#         # Process the results
#         if self.b_total is None:
#             self.b_total = total_intensity
#
#         if self.dip is None:
#             self.dip = -inclination
#             if not deg:
#                 self.dip = math.radians(self.dip)
#
#         if self.declination is None:
#             self.declination = declination
#             if not deg:
#                 self.declination = math.radians(self.declination)
#
#         # Convert to radians if deg is True
#         if deg:
#             self.dip = math.radians(self.dip)
#             self.declination = math.radians(self.declination)
#             self.convergence = math.radians(self.convergence)
#             self.vertical_inc_limit = math.radians(self.vertical_inc_limit)
#             self.vertical_section_azimuth = math.radians(self.vertical_section_azimuth)

class SurveyProcess:
    def __init__(self,
                 df_referenced,
                 drilled_depths,
                 stepped_boo=False,
                 steps=10,
                 elevation = 0,
                 coords_type = 'latlon'):
        self.coords_type = coords_type
        self.elevation = elevation
        self.start_lat, self.start_lon = df_referenced[['lat', 'lon']].iloc[0].tolist()

        if self.coords_type == 'latlon':
            df_referenced = self.convertCoordsToNEV(df_referenced)
        self.start_n, self.start_e= df_referenced[['n', 'e']].iloc[0].tolist()
        # self.utm_data = df_referenced[['e', 'n']].iloc[0].tolist()
        self.steps = steps
        self.stepped_boo = stepped_boo
        for col in ['Azimuth', 'Inclination']:
            df_referenced[col] = np.radians(df_referenced[col])
        df_referenced = self.check_and_insert_zero_md(df_referenced)
        self.original = copy.deepcopy(df_referenced)
        self.df_referenced = df_referenced
        self.drilled_depths = drilled_depths
        self.df, self.kop_lp = pd.DataFrame(), pd.DataFrame()
        # self.conv_angle = self.get_convergence(self.utm_data[0], self.utm_data[1])
        self.conv_angle = self.get_convergence(self.start_lat, self.start_lon)
        self.start_nev = np.array([self.start_n, self.start_e, self.elevation])
        # self.start_nev = np.array([self.utm_data[0] / 0.3048, self.utm_data[1] / 0.3048, 0])

        self.df_t, self.kop_t, self.prop_azi_t = self.mainProcess('t')
        self.df_g, self.kop_g, self.prop_azi_g = self.mainProcess('g')

    def convertCoordsToNEV(self, df):
        proj = Proj(proj="aeqd", datum="WGS84", lat_0=self.start_lat, lon_0=self.start_lon, units="us-ft")
        df['n'], df['e'] = proj(df['lon'].values, df['lat'].values)
        return df

    def convertCoordsFromNEV(self, n, e):
        pts = list(zip(n,e))
        proj = Proj(proj="aeqd", datum="WGS84", lat_0=self.start_lat, lon_0=self.start_lon, units="us-ft")
        latlon_points = [proj(e, n, inverse=True) for n, e in pts]
        lats, lons = [i[1] for i in latlon_points], [i[0] for i in latlon_points]
        return lats, lons


    def drilledDepthsProcess(self, df):
        self.drilled_depths['Interval'] = pd.IntervalIndex(self.drilled_depths['Interval'], closed='left')
        interval_to_feature = dict(zip(self.drilled_depths['Interval'], self.drilled_depths['Feature']))
        df['Feature'] = df['MeasuredDepth'].apply(
            lambda x: next((interval_to_feature[interval] for interval in interval_to_feature if x in interval),
                           'Unknown'))
        return df

    def check_and_insert_zero_md(self, df):
        if df['MeasuredDepth'].iloc[0] != 0:
            # Get the first row
            first_row = df.iloc[0].copy()
            # Set MD to 0
            first_row['MeasuredDepth'] = 0
            # Insert the new row at the beginning
            df = pd.concat([pd.DataFrame([first_row]), df]).reset_index(drop=True)
        return df

    def mainProcess(self, ref_type):
        if ref_type == 't':
            north_ref = 't'
            rad_type = 'azi_true_rad'
            deg_type = 'azi_true_deg'
        else:
            north_ref = 'g'
            rad_type = 'azi_true_rad'
            deg_type = 'azi_true_deg'
        columns_to_round = ['MeasuredDepth', 'Inclination', 'Azimuth', 'TVD', 'N Offset', 'E Offset',
                            'Vertical Section', 'DeltaZ', 'DeltaY', 'DeltaX', 'DeltaMD']

        # lat, lon = utm.to_latlon(self.utm_data[0], self.utm_data[1], 12, 'T')
        header = FastSurveyHeader(azi_reference=self.return_spelled_north_ref(north_ref.lower()), deg=False,
                                  convergence=math.radians(self.conv_angle))
        self.kop_lp = self.find_kop_and_lp(self.df_referenced, north_ref, rad_type)

        self.df = pd.concat([self.df_referenced, self.kop_lp], ignore_index=True)
        self.df = self.df.drop_duplicates(subset='MeasuredDepth', keep='first')
        self.df = self.df.sort_values('MeasuredDepth').reset_index(drop=True)
        survey_used = we.survey.Survey(md=self.df['MeasuredDepth'], inc=self.df['Inclination'],
                                       azi=self.df['Azimuth'], start_nev=self.start_nev, deg=False, header=header,
                                       error_model='ISCWSA MWD Rev4')
        if self.stepped_boo:
            survey_used = survey_used.interpolate_survey(step=self.steps)
        proposed_azimuth = survey_used.survey_deg[-1][2]

        min_curve = we.utils.MinCurve(md=self.df['MeasuredDepth'], inc=self.df['Inclination'],
                                      azi=getattr(survey_used, rad_type), unit='feet', start_xyz=self.start_nev)
        utm_vals, latlons = self.solve_utm(self.df['MeasuredDepth'], self.start_lat, self.start_lon, min_curve)
        lats, lons = latlons.T
        # print(min_curve.poss)
        # foo = min_curve.poss.tolist()[-5:]
        x,y,z =  min_curve.poss.T
        # lats3, lons3 = self.convertCoordsFromNEV(y, x)
        # for i in foo:
        #     print(i)
        # print(survey_used.pos_nev.tolist())
        # print()
        # foo = survey_used.pos_nev.tolist()[-5:]
        # for i in foo:
        #     print(i)
        # n,e,z = survey_used.pos_nev.T
        # lats, lons = self.convertCoordsFromNEV(n, e)

        easting, northing = utm_vals.T
        tool_face = self.toolface_solve(survey_used)
        outputs = {
            'MeasuredDepth': survey_used.md, 'Inclination': survey_used.inc_deg,
            'Azimuth': getattr(survey_used, deg_type),
            'TVD': survey_used.tvd, 'RatioFactor': min_curve.rf, 'N Offset': survey_used.y, 'E Offset': survey_used.x,
            'ToolFace': tool_face, 'Vertical Section': survey_used.vertical_section, 'Easting': easting,
            'Northing': northing, 'Lat': lats, 'Lon':lons,
            'DeltaZ': min_curve.delta_z, 'DeltaY': min_curve.delta_y, 'DeltaX': min_curve.delta_x,
            'DeltaMD': min_curve.delta_md, 'DogLegSeverity': min_curve.dls, 'BuildRadius': survey_used.radius,
            'BuildRate': survey_used.build_rate, 'TurnRate': survey_used.turn_rate, 'PositionX': y, 'PositionY': x, 'DepthActual': z}
        df = pd.DataFrame(outputs)

        df[columns_to_round] = df[columns_to_round].round(2)
        if not self.stepped_boo:
            init_md = self.df['MeasuredDepth'].tolist() + self.kop_lp['MeasuredDepth'].tolist()
            df = df[df['MeasuredDepth'].isin(init_md)]
        df = df.reset_index(drop=True)
        df['shp_pt'] = df.apply(
            lambda row: Point(row['Easting'], row['Northing']), axis=1)
        df['point_index'] = df.index
        df = self.drilledDepthsProcess(df)
        df = df.reindex(columns=['Feature', 'MeasuredDepth', 'Inclination', 'Azimuth', 'TVD', 'Vertical Section','RatioFactor', 'DogLegSeverity', 'DeltaMD','BuildRadius', 'BuildRate', 'TurnRate',
                                 'DeltaX', 'DeltaY', 'DeltaZ', 'PositionX', 'PositionY', 'DepthActual', 'Easting', 'Northing','Lat', 'Lon','shp_pt', 'point_index'])
        return df, self.kop_lp, proposed_azimuth





    def grapher(self, df1, df2):
        points3 = [
            [578241.754639931, 4437797.33819129],
            [578241.754639931, 4437797.33819129],
            [578241.754639931, 4437797.33819129],
            [578241.754639931, 4437797.33819129],
            [578241.754639931, 4437797.33819129],
            [578241.754639931, 4437797.33819129],
            [578241.754639931, 4437797.33819129],
            [578241.754639931, 4437797.33819129],
            [578241.754639931, 4437797.33819129],
            [578241.754639931, 4437797.33819129],
            [578241.754639931, 4437797.33819129],
            [578241.752349962, 4437797.56017999],
            [577360.556053558, 4433648.4610295],
            [577350.20556953, 4433617.71819769]
        ]
        points3 = np.array([utm.to_latlon(i[0], i[1], 12, 'T') for i in points3])
        # Both together
        plt.plot(df1['Lat'], df1['Lon'], '-o')  # line with dots
        plt.plot(df1['Lat2'], df1['Lon2'], '-o')  # line with dots
        plt.plot(df1['Lat3'], df1['Lon3'], '-o')  # line with dots
        # plt.plot(df2['Lat'], df2['Lon'], '-o')  # line with dots
        # plt.plot(df2['Lat2'], df2['Lon2'], '-o')  # line with dots
        # plt.plot(df2['Lat3'], df2['Lon3'], '-o')  # line with dots

        plt.plot(points3[:, 0], points3[:, 1], '-o')  # line with dots
        plt.show()





    def find_kop_and_lp(self, df, north_ref, rad_type):
        def is_valid_number(x):
            return pd.notnull(x) and np.isreal(x) and x > 0

        header = FastSurveyHeader(
            azi_reference=self.return_spelled_north_ref(north_ref.lower()),
            deg=False,
            convergence=math.radians(self.conv_angle)
        )

        # Create Survey object

        s = we.survey.Survey(
            md=df['MeasuredDepth'].values,
            inc=df['Inclination'].values,
            azi=df['Azimuth'].values,
            start_nev=self.start_nev,
            deg=False,
            header=header,
            error_model='ISCWSA MWD Rev4'
        )

        # Interpolate survey with numpy operations
        md = np.arange(s.md[0], s.md[-1], 10)
        inc = np.interp(md, s.md, s.inc_rad)
        azi = np.interp(md, s.md, getattr(s, rad_type))

        dls = np.interp(md, s.md, s.dls)

        kop_index = np.argmax(dls > 1.5)
        lp_index = np.argmax((np.degrees(inc) > 85) & (dls < 1.0))

        # Create result DataFrame
        result = pd.DataFrame({
            'MeasuredDepth': [md[kop_index], md[lp_index]],
            'Inclination': [inc[kop_index], inc[lp_index]],
            'Azimuth': [azi[kop_index], azi[lp_index]],
            'Point': ['KOP', 'LP']
        })
        invalid_x = result[~result['MeasuredDepth'].apply(is_valid_number)]
        if not invalid_x.empty:
            VERTICAL_INC_THRESHOLD = 5.0  # Degrees below which the well is considered vertical
            DEVIATED_INC_THRESHOLD = 5.0  # Degrees above which the well is considered deviated
            # Interpolate survey with numpy operations
            md = np.arange(s.md[0], s.md[-1], 10)
            inc = np.interp(md, s.md, s.inc_rad)
            azi = np.interp(md, s.md, s.azi_true_rad)

            # Convert inclination to degrees for threshold comparison
            inc_deg = np.degrees(inc)

            # Identify KOP: First index where inclination exceeds DEVIATED_INC_THRESHOLD
            kop_candidates = np.where(inc_deg > DEVIATED_INC_THRESHOLD)[0]

            kop_index = kop_candidates[0]
            # Identify LP: First index after KOP where inclination falls below VERTICAL_INC_THRESHOLD
            lp_candidates = np.where(inc_deg[kop_index:] < VERTICAL_INC_THRESHOLD)[0]

            inc_rad = np.radians(inc_deg)
            lp_index = kop_index + lp_candidates[0]
            result = pd.DataFrame({
                'MeasuredDepth': [md[kop_index], md[lp_index]],
                'Inclination': [inc_rad[kop_index], inc_rad[lp_index]],
                'Azimuth': [azi[kop_index], azi[lp_index]],
                'Point': ['KOP', 'LP']
            })

        return result[['MeasuredDepth', 'Inclination', 'Azimuth', 'Point']]

    def return_spelled_north_ref(self, val):
        return 'true' if val.lower() == 't' else 'grid' if val.lower() == 'g' else val

    def solve_utm(self, md, lat1, lon1, min_curve):
        geod = Geod(ellps='WGS84')
        lst = [[lat1, lon1]]
        for i in range(len(md) - 1):
            d_x, d_y = min_curve.delta_x[i + 1], min_curve.delta_y[i + 1]
            lon_x, lat_y, _ = geod.fwd(abs(lon1) * -1, lat1, 90 if d_x >= 0 else 270, abs(d_x) * 0.3048)
            lon1, lat1, _ = geod.fwd(abs(lon_x) * -1, lat_y, 0 if d_y >= 0 else 180, abs(d_y) * 0.3048)
            lst.append([lat1, lon1])
        return np.array([utm.from_latlon(i[0], i[1])[:2] for i in lst]).astype(int), np.array(lst)

    def toolface_solve(self, survey):
        vectors, pos_nev = survey.vec_nev, survey.pos_nev
        tool_face = [math.degrees(we.utils.get_toolface(pos_nev[i], pos_nev[i + 1], vectors[i])) for i in
                     range(len(pos_nev) - 1)]
        return np.array(tool_face + [tool_face[-1]])

    # 578241, 4437797
    # def get_convergence2(self, x, y, from_crs='EPSG:32612', to_crs='EPSG:4326'):
    #     x, y = 578241, 4437797
    #     crs_nztm = CRS(from_crs)
    #     p = Proj(crs_nztm)
    #     transformer = Transformer.from_crs(from_crs, to_crs, always_xy=True)
    #     lon, lat = transformer.transform(x, y)
    #     declination = p.get_factors(lon, lat, False, True).meridian_convergence
    #     return declination
    # def get_convergence(self, lat, lon, from_crs='EPSG:32612'):
    #     crs_nztm = CRS(from_crs)
    #     p = Proj(crs_nztm)
    #     print(lat, lon)
    #     declination = p.get_factors(lon, lat, False, True).meridian_convergence
    #     return declination


    def get_convergence(self, lat, lon, from_crs='EPSG:32043'):
        crs_spcs = CRS(from_crs)  # Use Utah Central Zone in State Plane
        p = Proj(crs_spcs)
        declination = p.get_factors(lon, lat, False, True).meridian_convergence
        return declination
class ClearanceProcess:
    def __init__(self, df_used, df_plat, adjacent_plats):
        self.df = df_used
        self.plats = df_plat
        self.adjacent_plats = adjacent_plats
        self.df['Conc'] = self.df['shp_pt'].apply(self.find_conc)
        self.all_polygons_concs = df_used['Conc'].unique()
        self.whole_df = pd.DataFrame()
        self.clearance_data = self.mainClearance()
        self.used_conc = self.clearance_data['Conc'].unique().tolist()

    def mainClearance(self):
        for i in range(len(self.all_polygons_concs)):
            well_traj = self.df[self.df['Conc'] == self.all_polygons_concs[i]]
            well_trajectory = well_traj[['Easting', 'Northing', 'point_index']].values

            try:
                used_poly = list(
                    self.plats[self.plats['Conc'] == self.all_polygons_concs[i]]['geometry'].values.tolist()[
                        0].exterior.coords)
            except IndexError:
                print('index error on guisurvey tab')


            right_lst_segments, left_lst_segments, up_lst_segments, down_lst_segments = self.idSides(used_poly)
            left_df = self.resultsFinder(left_lst_segments, 'West', well_trajectory)
            right_df = self.resultsFinder(right_lst_segments, 'East', well_trajectory)
            down_df = self.resultsFinder(down_lst_segments, 'South', well_trajectory)
            up_df = self.resultsFinder(up_lst_segments, 'North', well_trajectory)

            up_down = pd.merge(up_df, down_df, on='point_index')
            left_right = pd.merge(left_df, right_df, on='point_index')

            up_down = up_down.drop(columns=['well_point_y'])
            left_right = left_right.drop(columns=['well_point_y'])
            all_data = pd.merge(up_down, left_right, on='point_index')

            all_data = all_data.drop(columns=['well_point_x_y'])
            all_data = all_data.rename(columns={'well_point_x_x': 'well_point'})

            self.whole_df = pd.concat([all_data, self.whole_df]).reset_index(drop=True)
        self.whole_df = self.whole_df.sort_values(by=self.whole_df.columns[0])
        self.whole_df = self.whole_df.rename(
            columns={'distance_East': 'FEL', 'distance_West': 'FWL', 'distance_North': 'FNL', 'distance_South': 'FSL'})
        edited_df = self.whole_df[['point_index', 'FNL', 'FSL', 'FEL', "FWL"]]
        result = pd.merge(self.df, edited_df, on='point_index')

        return result

    def find_conc(self, point):
        for idx, row in self.adjacent_plats.iterrows():
            if row['geometry'].contains(point):
                return row['Conc']
        return None  # Return None if no matching polygon is found

    def calculate_well_to_line_clearance_detailed(self, well_trajectory, line_points):
        well_trajectory = np.array(well_trajectory)
        line_points = np.array(line_points)
        p1, p2 = line_points
        line_vector = p2 - p1
        line_length_squared = np.sum(line_vector ** 2)

        detailed_results = []
        for i, well_point in enumerate(well_trajectory):
            # Vector from line start to well point
            t = np.divide(np.dot(well_point - p1, line_vector), line_length_squared,
                          where=line_length_squared != 0,
                          out=np.zeros_like(line_length_squared))
            # Clamp t to [0, 1] to keep point on the segment
            t = max(0, min(1, t))

            # Calculate the closest point on the line segment
            closest_point = p1 + t * line_vector
            # Calculate the distance
            distance = np.linalg.norm(well_point - closest_point)

            # Calculate the angle
            well_to_closest = well_point - closest_point
            epsilon = 1e-10  # Adjust this value as needed
            angle = np.degrees(np.arccos(np.dot(well_to_closest, line_vector) /
                                         (np.linalg.norm(well_to_closest) * np.linalg.norm(line_vector) + epsilon)))
            # Ensure the angle is the smaller one (acute angle)
            if angle > 90:
                angle = 180 - angle

            result = {
                "point_index": i,
                "well_point": well_point,
                "distance": distance,
                "closest_surface_point": closest_point,
                "intersection_angle": angle,
                "original_segment": line_points
            }
            detailed_results.append(result)

        return detailed_results

    def idSides(self, polygon):
        corners, sides_generated = self.cornerGeneratorProcess(polygon)
        sides_generated = [[j[:-1] for j in i] for i in sides_generated]
        left_lst, up_lst, right_lst, down_lst = sides_generated[0], sides_generated[1], sides_generated[2], \
            sides_generated[3]
        left_lst, up_lst, right_lst, down_lst = (sorted(left_lst, key=lambda x: x[1]),
                                                 sorted(up_lst, key=lambda x: x[0]),
                                                 sorted(right_lst, key=lambda x: x[1], reverse=True),
                                                 sorted(down_lst, key=lambda x: x[0], reverse=True))
        right_lst_segments = [[right_lst[i], right_lst[i + 1]] for i in range(len(right_lst) - 1)]
        left_lst_segments = [[left_lst[i], left_lst[i + 1]] for i in range(len(left_lst) - 1)]
        down_lst_segments = [[down_lst[i], down_lst[i + 1]] for i in range(len(down_lst) - 1)]
        up_lst_segments = [[up_lst[i], up_lst[i + 1]] for i in range(len(up_lst) - 1)]
        return right_lst_segments, left_lst_segments, up_lst_segments, down_lst_segments

    def cornerGeneratorProcess(self, data_lengths):
        corner_arrange = self.optimized_corner_process(data_lengths)
        centroid = Polygon(data_lengths).centroid
        centroid = [centroid.x, centroid.y]
        corner_arrange = self.reorganizeLstPointsWithAngle([i[:2] for i in corner_arrange], centroid)
        corners = sorted(corner_arrange, key=lambda r: r[-1])
        corners = self.removeDupesListOfLists(corners)
        data_lengths = self.reorganizeLstPointsWithAngle(data_lengths, centroid)
        data_lengths = sorted(data_lengths, key=lambda r: r[-1])
        corners = [list(i) for i in corners]
        all_data = self.regularCornerClass(corners, data_lengths)
        return corners, all_data

    def optimized_corner_process(self, trajectory):
        # Convert trajectory to numpy array
        trajectory = np.array(trajectory)

        # Calculate centroid
        centroid = Polygon(trajectory).centroid.coords[0]

        # Use ConvexHull for clockwise sorting
        hull = ConvexHull(trajectory)
        trajectory = trajectory[hull.vertices]

        # Simplify trajectory
        epsilon = 0.002 if (35 < trajectory[0, 0] < 55 or 35 < trajectory[0, 1] < 55) else 200
        simplified = rdp(np.vstack((trajectory, trajectory[0])), epsilon=epsilon)

        # Calculate angles
        vectors = np.diff(simplified, axis=0)
        angles = np.arctan2(vectors[:, 1], vectors[:, 0])
        angle_diffs = np.diff(angles, append=angles[0] - 2 * np.pi)
        angle_diffs = np.abs(np.where(angle_diffs > np.pi, angle_diffs - 2 * np.pi, angle_diffs))

        # Find corners
        corners = simplified[1:][angle_diffs > np.pi / 35]

        # Calculate angles from centroid
        centroid_vectors = corners - centroid
        centroid_angles = (np.degrees(np.arctan2(centroid_vectors[:, 1], centroid_vectors[:, 0])) + 360) % 360

        # Combine coordinates and angles
        result = np.column_stack((corners, centroid_angles))

        return result.tolist()

    def reorganizeLstPointsWithAngle(self, lst, centroid):
        lst_arrange = [list(i) + [(math.degrees(math.atan2(centroid[1] - i[1], centroid[0] - i[0])) + 360) % 360] for i
                       in
                       lst]

        return lst_arrange

    def removeDupesListOfLists(self, lst):
        dup_free = []
        dup_free_set = set()
        for x in lst:
            if tuple(x) not in dup_free_set:
                dup_free.append(x)
                dup_free_set.add(tuple(x))
        return dup_free

    def regularCornerClass(self, corners, data_lengths):
        def find_corner_point(corners, min_angle, max_angle):
            return next((i for i in corners if min_angle < i[-1] <= max_angle), None)

        def find_side_points(data_lengths, start_angle, end_angle, reverse=False):
            points = [i for i in data_lengths if start_angle <= i[-1] <= end_angle]
            return self.remove_duplicates_preserve_order(points[::-1] if reverse else points)

        def get_side(data_lengths, corners, start_angle, end_angle, reverse=False):
            start_point = find_corner_point(corners, start_angle, end_angle)
            end_point = find_corner_point(corners, start_angle - 90, start_angle)

            if start_point is None or end_point is None:
                return []

            start_idx = data_lengths.index(start_point)
            end_idx = data_lengths.index(end_point)

            side = find_side_points(data_lengths,
                                    data_lengths[end_idx][-1],
                                    data_lengths[start_idx][-1],
                                    reverse)

            return side

        # Main code
        south_side = get_side(data_lengths, corners, 90, 180, reverse=True)
        east_side = get_side(data_lengths, corners, 180, 270, reverse=True)
        north_side = get_side(data_lengths, corners, 270, 360, reverse=True)

        # West side needs special handling due to the angle wrap-around
        nw_point = find_corner_point(corners, 270, 360)
        sw_point = find_corner_point(corners, 0, 90)

        if nw_point is not None and sw_point is not None:
            nw_idx = data_lengths.index(nw_point)
            sw_idx = data_lengths.index(sw_point)
            west_side = [sw_point] + [i for i in data_lengths if
                                      (i[-1] > data_lengths[nw_idx][-1] or
                                       i[-1] < data_lengths[sw_idx][-1]) and
                                      i not in (east_side + south_side + north_side)] + [nw_point]
            west_side = self.remove_duplicates_preserve_order(west_side)
        else:
            west_side = []

        all_data = [west_side] + [north_side] + [east_side] + [south_side]
        return all_data

    def remove_duplicates_preserve_order(self, points_list):
        seen = set()
        result = []

        for point in points_list:
            point_tuple = tuple(point)
            if point_tuple not in seen:
                result.append(point_tuple)
                seen.add(point_tuple)

        return result

    def consolidate_columns(self, df, num_segments, dir_val):
        consolidated = []
        for _, row in df.iterrows():
            valid_indices = [i for i in range(1, num_segments + 1)
                             if pd.notna(row[f'distance{i}_{dir_val}'])]

            if not valid_indices:
                consolidated.append({
                    'distance': np.nan,
                    'closest_surface_point': np.nan,
                    'intersection_angle': np.nan,
                    'segments': np.nan,
                })
            else:
                min_distance_index = min(valid_indices,
                                         key=lambda i: row[f'distance{i}_{dir_val}'])
                consolidated.append({
                    f'distance_{dir_val}': row[f'distance{min_distance_index}_{dir_val}'],
                    f'closest_surface_point_{dir_val}': row[
                        f'closest_surface_point{min_distance_index}_{dir_val}'],
                    f'intersection_angle_{dir_val}': row[f'intersection_angle{min_distance_index}_{dir_val}'],
                    f'segments_{dir_val}': row[f'segments{min_distance_index}_{dir_val}']
                })

        return pd.DataFrame(consolidated)

    def process_row(self, row, num_segments, dir_val):

        angles = [row[f'intersection_angle{i}_{dir_val}'] for i in range(1, num_segments + 1)]
        closest_to_90 = min(range(len(angles)), key=lambda i: abs(angles[i] - 90))

        for i in range(1, num_segments + 1):
            if i != closest_to_90 + 1:
                row[f'distance{i}_{dir_val}'] = np.nan
                row[f'closest_surface_point{i}_{dir_val}'] = np.nan
                row[f'intersection_angle{i}_{dir_val}'] = np.nan
                row[f'segments{i}_{dir_val}'] = np.nan
        return row

    def resultsFinder(self, segments, dir_val, well_trajectory):
        all_results = []
        well_index, well_trajectory = [i[2] for i in well_trajectory], [i[:2] for i in well_trajectory]
        for i, segment in enumerate(segments):
            results = self.calculate_well_to_line_clearance_detailed(well_trajectory, segment)
            for j, result in enumerate(results):
                if i == 0:
                    all_results.append({'point_index': well_index[j], 'well_point': well_trajectory[j]})
                all_results[j][f'distance{i + 1}_{dir_val}'] = round(result['distance'] / 0.3048, 2)
                all_results[j][f'closest_surface_point{i + 1}_{dir_val}'] = result['closest_surface_point']
                all_results[j][f'intersection_angle{i + 1}_{dir_val}'] = result['intersection_angle']
                all_results[j][f'segments{i + 1}_{dir_val}'] = segments[0]

        df = pd.DataFrame(all_results)
        column_order = ['point_index', 'well_point'] + [
            f'{col}{i}_{dir_val}' for i in range(1, len(segments) + 1)
            for col in ['distance', 'closest_surface_point', 'intersection_angle', 'segments']]

        if df.empty:
            # Create a new DataFrame with empty columns but correct number of rows
            well_traj_pts = [[None, None] for i in well_trajectory]
            placeholder_data = {'point_index': well_index, 'well_point': well_trajectory,
                                f'distance1_{dir_val}': [None] * len(well_trajectory),
                                f'closest_surface_point1_{dir_val}': [None] * len(well_trajectory),
                                f'intersection_angle1_{dir_val}': [None] * len(well_trajectory),
                                f'segments1_{dir_val}': well_traj_pts}
            df = pd.DataFrame(placeholder_data)
            return df
        else:
            df = df[column_order]
            num_segments = len(segments)
            df = df.apply(lambda row: self.process_row(row, num_segments, dir_val), axis=1)
            consolidated_df = self.consolidate_columns(df, num_segments, dir_val)
            df = pd.concat([df[['point_index', 'well_point']], consolidated_df], axis=1)
            return df



def dict_to_interval(d):
    d = json.loads(d) if isinstance(d, str) else d
    return pd.Interval(d['left'], d['right'], closed=d['closed'])


pd.set_option('display.max_columns', None)
pd.options.mode.chained_assignment = None
conn_sample = sqlite3.connect('DX_sample.db')
# query = "select * from DX"
# dx_df = pd.read_sql(query, conn_sample)
query = "select * from PlatDF"
plat_df = pd.read_sql(query, conn_sample)
query = "select * from PlatAdj"
plats_adjacent = pd.read_sql(query, conn_sample)
query = "select * from DX_Original"
dx_df_orig = pd.read_sql(query, conn_sample)
# dx_df_orig[['lat', 'lon']] = dx_df_orig.apply(lambda row: pd.Series(utm.to_latlon(row['Easting'], row['Northing'], 12, 'T')), axis=1)
# dx_df_orig = dx_df_orig.drop(columns=['Easting', 'Northing', 'CitingType'], axis=1)
# conn_sample.execute('DROP TABLE IF EXISTS DX_Original')
# dx_df_orig.to_sql('DX_Original', conn_sample, index=False)
# print(dx_df_orig)


query = "select * from Depths"
df_depths = pd.read_sql(query, conn_sample)
df_depths['Interval'] = df_depths['Interval'].apply(dict_to_interval)

# dx_df['shp_pt'] = dx_df['shp_pt'].apply(lambda row: wkt.loads(row))
plat_df['geometry'] = plat_df['geometry'].apply(lambda row: wkt.loads(row))
plat_df['centroid'] = plat_df['centroid'].apply(lambda row: wkt.loads(row))
plats_adjacent['geometry'] = plats_adjacent['geometry'].apply(lambda row: wkt.loads(row))
plats_adjacent['centroid'] = plats_adjacent['centroid'].apply(lambda row: wkt.loads(row))
dx_df = SurveyProcess(df_referenced = dx_df_orig, drilled_depths = df_depths,elevation = 5515, coords_type = 'latlon')
clear_df = ClearanceProcess(dx_df.df_t, plat_df, plats_adjacent)
base_dx_df_planned, planned_footages = clear_df.clearance_data, clear_df.whole_df
