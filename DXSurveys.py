"""
DXSurveys.py
Author: Colton Goodrich
Date: 11/10/2024
Python Version: 3.12
Survey processing and manipulation for directional well data.

This module provides functionality for processing directional survey data with
magnetic field calculations, interpolation, and coordinate transformations.

Key Features:
    - Survey interpolation and minimum curvature calculations
    - Magnetic field reference handling
    - Coordinate system conversions (UTM, lat/lon)
    - Integration with welleng library for survey calculations
    - Support for various azimuth reference systems
    - Error modeling using ISCWSA MWD Rev4

Typical usage example:
    survey = DXSurvey(
        df=survey_df,
        start_nev=[0,0,0],
        conv_angle=1.5,
        interpolate=True
    )

    results = survey.process_trajectory()

Notes:
    - Requires input DataFrame with standard survey columns:
        * MeasuredDepth: Measured depth values
        * Inclination: Inclination angles
        * Azimuth: Azimuth angles
    - Handles both grid and true north references
    - Integrates with spatial analysis tools via shapely geometries

Dependencies:
    - welleng
    - numpy
    - pandas
    - pyproj
    - shapely
    - pygeomag
    - scipy
"""
from pandas import Series, DataFrame
from scipy.ndimage import gaussian_filter1d
import copy
from welltrajconvert.wellbore_trajectory import *
from shapely.geometry import Point
import welleng as we
from pyproj import Geod, Proj, CRS
import numpy.typing as npt
import pandas as pd
import numpy as np
import math
from datetime import datetime
from welleng.survey import SurveyHeader
from pygeomag import GeoMag
from typing import Optional, Tuple, Dict, Literal, Any
from scipy.signal import savgol_filter
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans

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


def _get_convergence(lat: float, lon: float, from_crs: str = 'EPSG:32043') -> float:
    """Calculate the meridian convergence angle for a given latitude/longitude coordinate.

    This function computes the meridian convergence (the angular difference between
    grid north and true north) at a specified location using the State Plane
    Coordinate System.

    Args:
        lat: Latitude coordinate in decimal degrees.
        lon: Longitude coordinate in decimal degrees.
        from_crs: EPSG code for the coordinate reference system. Defaults to
            'EPSG:32043' (Utah Central Zone State Plane).

    Returns:
        float: Meridian convergence angle in degrees.
            Positive values indicate convergence east of true north.
            Negative values indicate convergence west of true north.

    Examples:
        >>> _get_convergence(40.7608, -111.8910)  # Salt Lake City coordinates
        -0.324  # Example return value

    Notes:
        - The function uses the Utah Central Zone State Plane by default, which is
          appropriate for central Utah locations.
        - For locations outside Utah Central Zone, specify the appropriate EPSG code.
    """
    # Initialize the Coordinate Reference System using the specified EPSG code
    crs_spcs = CRS(from_crs)

    # Create a projection object for coordinate transformation and calculations
    p = Proj(crs_spcs)

    # Calculate meridian convergence using projection factors
    # Parameters: (longitude, latitude, radians=False, return_convergence=True)
    declination = p.get_factors(lon, lat, False, True).meridian_convergence

    return declination


def _check_and_insert_zero_md(df: pd.DataFrame) -> pd.DataFrame:
    """Insert a zero measured depth row if one doesn't exist at the start of the dataframe.

    This function checks if the first row of the dataframe has a measured depth of zero.
    If not, it creates a new row by copying the first row's values and setting its
    measured depth to zero, then prepends this row to the dataframe.

    Args:
        df: A pandas DataFrame containing a 'MeasuredDepth' column with survey data.
            The dataframe is expected to be sorted by measured depth in ascending order.

    Returns:
        pd.DataFrame: The modified dataframe with a zero measured depth row if one
            was needed, otherwise returns the original dataframe unchanged.

    Notes:
        - This function is intended for internal use (denoted by leading underscore)
        - The function preserves all other column values from the first row
        - Index will be reset after modification
        - The original dataframe is not modified in place

    Examples:
        >>> data = pd.DataFrame({'MeasuredDepth': [100, 200], 'Value': [1, 2]})
        >>> result = _check_and_insert_zero_md(data)
        >>> print(result['MeasuredDepth'].tolist())
        [0, 100, 200]
    """
    # Check if first row's measured depth is not zero
    if df['MeasuredDepth'].iloc[0] != 0:
        # Create a copy of the first row to preserve all other values
        first_row = df.iloc[0].copy()
        # Modify the measured depth to zero
        first_row['MeasuredDepth'] = 0
        # Concatenate the new zero MD row with the original dataframe
        df = pd.concat([pd.DataFrame([first_row]), df]).reset_index(drop=True)

    return df


def _return_spelled_north_ref(val: str) -> str:
    """Convert single letter north reference codes to full spelled-out versions.

    This helper function translates abbreviated north reference indicators to their
    full text equivalents. It specifically handles 'T' for 'true' north and 'G' for
    'grid' north, while passing through any other values unchanged.

    Args:
        val: A string containing the north reference indicator. Common values are:
            't' or 'T' for true north
            'g' or 'G' for grid north
            Any other string value will be returned unchanged

    Returns:
        str: The full spelled-out version of the north reference:
            'true' if input is 't' or 'T'
            'grid' if input is 'g' or 'G'
            Original input value for all other cases

    Examples:
        >>> _return_spelled_north_ref('T')
        'true'
        >>> _return_spelled_north_ref('g')
        'grid'
        >>> _return_spelled_north_ref('magnetic')
        'magnetic'
    """
    # Convert to lowercase for case-insensitive comparison and use conditional expressions
    return 'true' if val.lower() == 't' else 'grid' if val.lower() == 'g' else val

def _solve_utm(md: np.ndarray,
               lat1: float,
               lon1: float,
               min_curve: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Convert minimum curvature calculations to UTM coordinates via geodesic calculations.

    This function performs a step-by-step conversion of minimum curvature delta values
    into UTM coordinates by first converting through geographic coordinates using
    geodesic calculations on the WGS84 ellipsoid.

    Args:
        md: Array of measured depths corresponding to survey points.
        lat1: Initial latitude in decimal degrees.
        lon1: Initial longitude in decimal degrees.
        min_curve: Object containing minimum curvature calculations with attributes:
            - delta_x: Array of x-direction displacement values
            - delta_y: Array of y-direction displacement values

    Returns:
        Tuple containing:
            - np.ndarray: Array of UTM coordinates (easting, northing) as integers
            - np.ndarray: Array of intermediate lat/lon coordinates used in calculation

    Notes:
        - Uses WGS84 ellipsoid for geodesic calculations
        - Converts distances from feet to meters (0.3048 conversion factor)
        - Returns integer UTM coordinates
        - Assumes input coordinates are in decimal degrees
        - Preserves sign of longitude through calculations

    Examples:
        >>> md = np.array([0, 100, 200])
        >>> min_curve = MinCurvature(delta_x=[0, 10, 20], delta_y=[0, 5, 10])
        >>> utm_coords, latlon = _solve_utm(md, 40.0, -111.0, min_curve)
    """
    # Initialize geodesic calculator with WGS84 ellipsoid
    geod = Geod(ellps='WGS84')

    # Initialize list to store intermediate lat/lon coordinates
    lst = [[lat1, lon1]]

    # Process each segment of the wellbore
    for i in range(len(md) - 1):
        # Get delta values for current segment
        d_x, d_y = min_curve.delta_x[i + 1], min_curve.delta_y[i + 1]

        # Calculate intermediate point moving east/west
        # Convert feet to meters using 0.3048 conversion factor
        lon_x, lat_y, _ = geod.fwd(abs(lon1) * -1, lat1,
                                   90 if d_x >= 0 else 270,
                                   abs(d_x) * 0.3048)

        # Calculate final point moving north/south
        lon1, lat1, _ = geod.fwd(abs(lon_x) * -1, lat_y,
                                 0 if d_y >= 0 else 180,
                                 abs(d_y) * 0.3048)

        # Store calculated coordinates
        lst.append([lat1, lon1])

    # Convert all lat/lon pairs to UTM coordinates and return as integer arrays
    return (np.array([utm.from_latlon(i[0], i[1])[:2] for i in lst]).astype(int),
            np.array(lst))

def _get_reference_settings(ref_type: Literal['t', 'g']) -> Dict[str, str]:
    """Get reference system settings for survey calculations.

    Determines the appropriate reference settings based on whether true north
    or grid north is being used for azimuth calculations.

    Args:
        ref_type: Single character string indicating reference system:
            't': True north reference system
            'g': Grid north reference system

    Returns:
        Dict containing reference settings with keys:
            - north_ref: Reference system identifier ('t' or 'g')
            - rad_type: Type of azimuth measurement in radians
            - deg_type: Type of azimuth measurement in degrees

    Notes:
        - Currently both true and grid north use the same rad/deg types
        - This is a helper function to simplify reference system setup
        - Used primarily in survey calculations and transformations

    Examples:
        >>> settings = _get_reference_settings('t')
        >>> print(settings['north_ref'])
        't'
        >>> print(settings['rad_type'])
        'azi_true_rad'
    """
    # Return settings dictionary based on reference type
    if ref_type == 't':
        return {
            'north_ref': 't',
            'rad_type': 'azi_true_rad',
            'deg_type': 'azi_true_deg'
        }

    return {
        'north_ref': 'g',
        'rad_type': 'azi_true_rad',
        'deg_type': 'azi_true_deg'
    }

def _setup_survey_header(north_ref: Literal['t', 'g'], conv_angle: float) -> FastSurveyHeader:
    """Create a survey header with specified reference system and convergence angle.

    Initializes a FastSurveyHeader object with the appropriate azimuth reference
    system and convergence angle correction for survey calculations.

    Args:
        north_ref: Reference system identifier:
            't': True north reference
            'g': Grid north reference
        conv_angle: Convergence angle in degrees to be converted to radians
            for magnetic declination correction

    Returns:
        FastSurveyHeader: Configured survey header object with specified settings

    Notes:
        - Automatically converts convergence angle from degrees to radians
        - Forces degrees mode to False for internal calculations
        - North reference is converted to lowercase and spelled out form

    Examples:
        >>> header = _setup_survey_header('t', 1.5)
        >>> print(header.azi_reference)
        'true'
        >>> print(header.convergence)  # Will be in radians
        0.026179938779914945
    """
    return FastSurveyHeader(
        azi_reference=_return_spelled_north_ref(north_ref.lower()),
        deg=False,
        convergence=math.radians(conv_angle)
    )

def _create_survey(
        df: pd.DataFrame,
        start_nev: Tuple[float, float, float],
        header: 'FastSurveyHeader',
) -> 'we.survey.Survey':
    """Create and optionally interpolate a wellbore survey object.

    Initializes a Survey object from measured depth, inclination, and azimuth data
    with specified starting position and reference parameters.

    Args:
        df: DataFrame containing survey data with columns:
            - MeasuredDepth: Measured depth values
            - Inclination: Inclination angles
            - Azimuth: Azimuth angles
        start_nev: Tuple of starting position coordinates (north, east, vertical)
        header: FastSurveyHeader object containing reference system settings

    Returns:
        we.survey.Survey: Survey object containing well trajectory data and calculated parameters

    Notes:
        - All angular inputs should be in radians (deg=False)
        - Uses ISCWSA MWD Rev4 error model for uncertainty calculations
        - Interpolation creates regular spacing between survey points
        - Original survey points are preserved in interpolation

    Examples:
        >>> survey = _create_survey(df, (0,0,0), header, True, 100)
        >>> print(len(survey.md))  # Will show interpolated point count
    """
    # Initialize base survey object
    survey = we.survey.Survey(
        md=df['MeasuredDepth'],
        inc=df['Inclination'],
        azi=df['Azimuth'],
        start_nev=start_nev,
        deg=False,
        header=header,
        error_model='ISCWSA MWD Rev4'
    )

    return survey

def _process_min_curve(
        df: pd.DataFrame,
        survey: we.survey.Survey,
        rad_type: str,
        start_nev: tuple[float, float, float]
) -> 'we.utils.MinCurve':
    """Process minimum curvature calculations for wellbore trajectory.

    Creates a MinCurve object to calculate the minimum curvature solution
    for the well path, including position vectors, dogleg severity,
    and other geometric parameters.

    Args:
        df: DataFrame containing survey data with required column:
            - MeasuredDepth: Array of measured depth values
            - Inclination: Array of inclination angles
        survey: Survey object containing processed survey data
        rad_type: String specifying which azimuth attribute to use (e.g. 'azi_true_rad')
        start_nev: Tuple of starting position coordinates (north, east, vertical)

    Returns:
        MinCurve: Object containing minimum curvature calculations including:
            - delta_x/y/z: Position changes between survey stations
            - delta_md: Measured depth changes between stations
            - dls: Dogleg severity values
            - rf: Ratio factors
            - poss: Position vectors

    Notes:
        - All calculations assume units in feet
        - Azimuth values are extracted dynamically using getattr
        - Position calculations start from provided start_nev coordinates
        - Used primarily for well path position and geometric calculations

    Examples:
        >>> min_curve = _process_min_curve(df, survey, 'azi_true_rad', (0,0,0))
        >>> print(min_curve.dls)  # Get dogleg severity values
    """
    return we.utils.MinCurve(
        md=df['MeasuredDepth'],
        inc=df['Inclination'],
        azi=getattr(survey, rad_type),
        unit='feet',
        start_xyz=start_nev
    )


def _create_output_dict(
        survey: 'we.survey.Survey',
        min_curve: 'we.utils.MinCurve',
        utm_vals: npt.NDArray[np.float64],
        latlons: npt.NDArray[np.float64],
        deg_type: str
) -> Dict[str, npt.NDArray[np.float64]]:
    """Create a dictionary containing all computed survey and position data.

    Organizes survey calculations, position data, and derived measurements into a
    standardized dictionary format for analysis and export.

    Args:
        survey: Survey object containing basic survey calculations
        min_curve: MinCurve object with minimum curvature calculations
        utm_vals: Array of UTM coordinates (easting, northing pairs)
        latlons: Array of latitude/longitude coordinates
        deg_type: String specifying which degree attribute to use from survey

    Returns:
        Dictionary containing arrays for:
            - Survey measurements (MD, Inc, Azi, TVD)
            - Position data (N/E offsets, UTM, Lat/Lon)
            - Geometric calculations (DLS, Build Rate, Turn Rate)
            - Minimum curvature results (RF, deltas)
            - Tool orientation (Tool Face)
            - Cartesian positions (X, Y, Z)

    Notes:
        - All angular values are in degrees
        - Position values maintain original input units
        - Arrays are aligned by measured depth
        - X/Y/Z positions are transformed from min_curve coordinate system

    Examples:
        >>> output = _create_output_dict(survey, min_curve, utm, latlon, tf, 'azi_true_deg')
        >>> print(output['TVD'])  # Access true vertical depth array
    """
    # Unpack coordinate arrays for clarity
    lats, lons = latlons.T
    x, y, z = min_curve.poss.T
    easting, northing = utm_vals.T

    # Create comprehensive output dictionary with all survey and position data
    return {
        'MeasuredDepth': survey.md,
        'Inclination': survey.inc_deg,
        'Azimuth': getattr(survey, deg_type),
        'TVD': survey.tvd,
        'RatioFactor': min_curve.rf,
        'N Offset': survey.y,
        'E Offset': survey.x,
        'ToolFace': survey.toolface,
        'Vertical Section': survey.vertical_section,
        'Easting': easting,
        'Northing': northing,
        'Lat': lats,
        'Lon': lons,
        'DeltaZ': min_curve.delta_z,
        'DeltaY': min_curve.delta_y,
        'DeltaX': min_curve.delta_x,
        'DeltaMD': min_curve.delta_md,
        'DogLegSeverity': min_curve.dls,
        'BuildRadius': survey.radius,
        'BuildRate': survey.build_rate,
        'TurnRate': survey.turn_rate,
        'PositionX': y,  # Note coordinate system transformation
        'PositionY': x,  # Note coordinate system transformation
        'DepthActual': z
    }


class SurveyProcess:
    """Process and transform well survey data between different coordinate systems and reference frames.

    This class handles the conversion and processing of well survey data, including coordinate
    transformations, depth calculations, and survey interpolation. It supports both true north
    and grid north reference systems.

    Attributes:
        coords_type (str): Type of input coordinates ('latlon' or other)
        elevation (float): Surface elevation of the well in feet
        start_lat (float): Starting latitude in decimal degrees
        start_lon (float): Starting longitude in decimal degrees
        start_n (float): Starting northing coordinate
        start_e (float): Starting easting coordinate
        original (pd.DataFrame): Copy of original input data
        df_referenced (pd.DataFrame): Processed reference dataframe
        conv_angle (float): Convergence angle between true and grid north
        start_nev (np.ndarray): Starting position vector [north, east, vertical]
        df_t (pd.DataFrame): Processed data referenced to true north
        df_g (pd.DataFrame): Processed data referenced to grid north
        kop_t (pd.DataFrame): Kickoff points referenced to true north
        kop_g (pd.DataFrame): Kickoff points referenced to grid north
        prop_azi_t (float): Proposed azimuth referenced to true north
        prop_azi_g (float): Proposed azimuth referenced to grid north

    Args:
        df_referenced (pd.DataFrame): Input survey data containing at minimum:
            - lat: Latitude in decimal degrees
            - lon: Longitude in decimal degrees
            - Azimuth: Well azimuth in degrees
            - Inclination: Well inclination in degrees
        elevation (float, optional): Surface elevation in feet. Defaults to 0.
        coords_type (str, optional): Coordinate system type. Defaults to 'latlon'.

    Notes:
        - All angular inputs are expected in degrees and are converted to radians internally
        - The class processes surveys in both true north and grid north reference frames
        - A zero measured depth point is automatically added if not present
        - Coordinates are transformed to NEV (North-East-Vertical) system for calculations
    """

    def __init__(self,
                 df_referenced: pd.DataFrame,
                 elevation: float = 0) -> None:
        """Initialize the SurveyProcess class with survey data and processing parameters."""
        # Store initialization parameters
        self.elevation = elevation

        # Extract starting coordinates
        self.start_lat, self.start_lon = df_referenced[['lat', 'lon']].iloc[0].tolist()

        # # Convert coordinates
        df_referenced = self._convert_coords_to_nev(df_referenced)

        # Store starting NEV coordinates
        self.start_n, self.start_e = df_referenced[['n', 'e']].iloc[0].tolist()

        # Convert angular measurements to radians
        for col in ['Azimuth', 'Inclination']:
            df_referenced[col] = np.radians(df_referenced[col])

        # Ensure zero measured depth exists
        df_referenced = _check_and_insert_zero_md(df_referenced)

        # Store dataframes and parameters
        self.original = copy.deepcopy(df_referenced)
        self.df_referenced = df_referenced
        self.df, self.kop_lp = pd.DataFrame(), pd.DataFrame()

        # Calculate convergence angle and starting position
        self.conv_angle = _get_convergence(self.start_lat, self.start_lon)
        self.start_nev = np.array([self.start_n, self.start_e, self.elevation])

        # Process data for both true and grid north references
        self.true_dx, self.prop_azi_true = self._main_process('t')
        self.grid_dx, self.prop_azi_grid = self._main_process('g')

    def drilled_depths_process(self, df: pd.DataFrame, drilled_depths: List[float]) -> pd.DataFrame:
        """Process measured depths to assign formation/feature labels to survey points.

        Matches each survey point's measured depth to a corresponding geological feature
        or formation based on depth intervals. Uses left-closed intervals to ensure
        consistent feature assignment at boundary depths.

        Args:
            df: DataFrame containing at minimum:
                - MeasuredDepth: Survey measured depths

        Returns:
            pd.DataFrame: Input DataFrame with additional column:
                - Feature: String identifier of geological feature/formation

        Notes:
            - Uses self.drilled_depths which must contain:
                - Interval: Depth ranges for features
                - Feature: Names/labels for geological features
            - Intervals are converted to pandas IntervalIndex with left-closed bounds
            - Points not matching any interval are labeled as 'Unknown'
            - Each depth point can only belong to one feature interval

        Examples:
            >>> depths_df = pd.DataFrame({
            ...     'Interval': [(0, 100), (100, 200)],
            ...     'Feature': ['Surface', 'Reservoir']
            ... })
            >>> survey_df = pd.DataFrame({'MeasuredDepth': [50, 150]})
            >>> processed = drilled_depths_process(survey_df)
            >>> print(processed['Feature'])
            0    Surface
            1    Reservoir
            :param df:
            :param drilled_depths:
        """

        # Convert intervals to left-closed IntervalIndex
        drilled_depths['Interval'] = pd.IntervalIndex(
            drilled_depths['Interval'],
            closed='left'
        )

        # Create mapping dictionary from intervals to features
        interval_to_feature = dict(zip(
            drilled_depths['Interval'],
            drilled_depths['Feature']
        ))

        # Assign features based on depth intervals
        df['Feature'] = df['MeasuredDepth'].apply(
            lambda x: next((interval_to_feature[interval]
                            for interval in interval_to_feature
                            if x in interval),
                           'Unknown')
        )
        df = df[['Feature', 'MeasuredDepth']]
        return df


    def _convert_coords_to_nev(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert geographic coordinates (lat/lon) to local North-East-Vertical (NEV) coordinates.

        This method projects geographic coordinates onto a local azimuthal equidistant
        projection centered at the well's surface location. The projection preserves
        distances and azimuths from the center point, making it ideal for wellbore
        survey calculations.

        Args:
            df: A pandas DataFrame containing at minimum:
                - lat: Latitude values in decimal degrees
                - lon: Longitude values in decimal degrees

        Returns:
            pd.DataFrame: The input dataframe with two new columns added:
                - n: Northing coordinates in US survey feet
                - e: Easting coordinates in US survey feet

        Notes:
            - Uses an Azimuthal Equidistant projection (aeqd)
            - WGS84 datum is used as the reference ellipsoid
            - Output coordinates are in US survey feet
            - Projection is centered on well's surface location (self.start_lat/lon)
            - Original lat/lon columns are preserved

        Examples:
            >>> df = pd.DataFrame({'lat': [40.0, 40.1], 'lon': [-110.0, -110.1]})
            >>> converted = self._convert_coords_to_nev(df)
            >>> print(converted[['n', 'e']].head())
               n          e
            0  0.0        0.0
            1  12345.6    -6789.0
        """
        # Initialize the azimuthal equidistant projection centered on well location
        proj = Proj(proj="aeqd",
                    datum="WGS84",
                    lat_0=self.start_lat,
                    lon_0=self.start_lon,
                    units="us-ft")

        # Project coordinates and add to dataframe as new columns
        df['n'], df['e'] = proj(df['lon'].values, df['lat'].values)

        return df


    def _main_process(
            self,
            ref_type: str
    ) -> tuple[Any, Any]:
        """Execute main well trajectory processing pipeline.

        Processes survey data through multiple stages including survey creation,
        minimum curvature calculations, coordinate transformations, and data organization.

        Args:
            ref_type: Reference type string determining north reference system and angle types

        Returns:
            Tuple containing:
                - pd.DataFrame: Processed survey data with all calculations
                - pd.DataFrame: Kick-off point and landing point data
                - float: Proposed azimuth at final survey station

        Notes:
            Processing stages:
            1. Initialize reference settings and constants
            2. Create survey header and process KOP/LP points
            3. Process survey calculations and minimum curvature
            4. Transform coordinates (UTM and Lat/Lon)
            5. Calculate tool face angles
            6. Organize and format output data
            7. Add shape points and process drilled depths

        Processing includes:
            - Coordinate transformations
            - Geometric calculations
            - Data validation and cleanup
            - Column formatting and rounding
        """
        # Define columns requiring 2-decimal rounding
        columns_to_round = [
            'MeasuredDepth', 'Inclination', 'Azimuth', 'TVD', 'N Offset', 'E Offset',
            'Vertical Section', 'DeltaZ', 'DeltaY', 'DeltaX', 'DeltaMD'
        ]

        # Initialize reference system settings
        ref_settings = _get_reference_settings(ref_type)
        north_ref = ref_settings['north_ref']
        rad_type = ref_settings['rad_type']
        deg_type = ref_settings['deg_type']

        # Setup survey header and process kickoff points
        header = _setup_survey_header(north_ref, self.conv_angle)
        # self.find_kop(self.df_referenced)
        # self.kop_lp = self._find_kop_and_lp(self.df_referenced, north_ref, rad_type)

        # Combine and clean survey data
        # self.df = pd.concat([self.df_referenced, self.kop_lp], ignore_index=True)
        self.df = self.df_referenced.drop_duplicates(subset='MeasuredDepth', keep='first')
        self.df = self.df.sort_values('MeasuredDepth').reset_index(drop=True)

        # Process survey calculations
        survey_used = _create_survey(self.df, self.start_nev, header)
        proposed_azimuth = survey_used.survey_deg[-1][2]

        # Calculate minimum curvature and coordinate transformations
        min_curve = _process_min_curve(self.df, survey_used, rad_type, self.start_nev)
        utm_vals, latlons = _solve_utm(self.df['MeasuredDepth'], self.start_lat, self.start_lon, min_curve)

        # Process tool face angles and create output structure
        outputs = _create_output_dict(survey_used, min_curve, utm_vals, latlons, deg_type)
        df = pd.DataFrame(outputs)

        # Round specified columns
        df[columns_to_round] = df[columns_to_round].round(2)
        init_md = self.df['MeasuredDepth'].tolist()# + self.kop_lp['MeasuredDepth'].tolist()
        df = df[df['MeasuredDepth'].isin(init_md)]

        # Add shape points and process indices
        df = df.reset_index(drop=True)
        df['shp_pt'] = df.apply(lambda row: Point(row['Easting'], row['Northing']), axis=1)
        df['point_index'] = df.index

        # Organize final column structure
        final_columns = [
            'Feature', 'MeasuredDepth', 'Inclination', 'Azimuth', 'TVD', 'Vertical Section',
            'RatioFactor', 'DogLegSeverity', 'DeltaMD', 'BuildRadius', 'BuildRate', 'TurnRate',
            'DeltaX', 'DeltaY', 'DeltaZ', 'PositionX', 'PositionY', 'DepthActual', 'Easting',
            'Northing', 'Lat', 'Lon', 'shp_pt', 'point_index'
        ]
        df = df.reindex(columns=final_columns)

        return df, proposed_azimuth
    def find_kop2(self, survey_data, dls_threshold=0.035, smoothing_sigma=2):

        # Ensure input data is sorted by MD
        # survey_data['DLS'] = np.concatenate(([0], calculate_dls(survey_data)))

        # Moving average of DLS (for smoothing)
        survey_data['DLS_smooth'] = survey_data['DLS'].rolling(window=3).mean()
        #
        # # Calculate the rate of change (derivative) of DLS
        survey_data['DLS_rate_of_change'] = survey_data['DLS_smooth'].diff()
        #
        # # Find the point where the rate of change of DLS is the highest (possible KOP)
        kop_point = survey_data.iloc[survey_data['DLS_rate_of_change'].idxmax()]
        print(kop_point)
        # foo_data = survey_data.values.tolist()


        # survey_data = survey_data.sort_values('MeasuredDepth')
        #
        # # Calculate derivatives of inclination and azimuth
        # survey_data['dI_dMD'] = np.gradient(survey_data['Inclination'], survey_data['MeasuredDepth'])
        # survey_data['dA_dMD'] = np.gradient(survey_data['Azimuth'], survey_data['MeasuredDepth'])
        #
        # # Smooth derivatives to reduce noise
        # survey_data['dI_dMD_smoothed'] = savgol_filter(survey_data['dI_dMD'], window_length=5, polyorder=2)
        # survey_data['Curvature'] = np.abs(survey_data['dI_dMD_smoothed'])  # Approximation for curvature
        #
        # # Identify regions with significant deviation using clustering
        # features = survey_data[['MeasuredDepth', 'Curvature']].values
        # kmeans = KMeans(n_clusters=2, random_state=0).fit(features)
        # survey_data['Cluster'] = kmeans.labels_
        #
        # # Determine the transition from cluster 0 (vertical) to cluster 1 (deviation)
        # kop_cluster = survey_data[survey_data['Cluster'] == 1]
        # kop_row = kop_cluster.iloc[0]  # First occurrence in the deviation cluster
        # # # Extract KOP details
        # kop_md = kop_row['MeasuredDepth']
        # kop_incl = kop_row['Inclination']
        # kop_azim = kop_row['Azimuth']
        survey_data['Smoothed_Inclination'] = gaussian_filter1d(survey_data['Inclination'], sigma=smoothing_sigma)

        # Calculate rate of change in inclination (radians per foot)
        survey_data['Inclination_Change'] = survey_data['Smoothed_Inclination'].diff() / survey_data['MeasuredDepth'].diff()

        # Calculate Dogleg Severity (approximation for small angles)
        survey_data['DLS'] = survey_data['Inclination_Change'].abs()

        # Detect KOP where DLS exceeds threshold with sustained trend
        for i in range(1, len(survey_data)):
            if survey_data['DLS'].iloc[i] > dls_threshold:
                # Verify sustained change trend
                if survey_data['Inclination_Change'].iloc[i] > 0:
                    print(survey_data['MeasuredDepth'].iloc[i])
                    return survey_data['MeasuredDepth'].iloc[i]
        return None  # KOP not found
    def find_lp(self, survey_data):
        md = survey_data['MeasuredDepth'].values
        inc = survey_data['Inclination'].values
        azi = survey_data['Azimuth'].values
        dls = survey_data['DogLegSeverity'].values


    def find_kop(self, survey_data):
        def is_valid_number(x: float) -> bool:
            """Validate numeric values."""
            return pd.notnull(x) and np.isreal(x) and x > 0

        print(solve_inclination(survey_data))
        md = survey_data['MeasuredDepth'].values
        inc = survey_data['Inclination'].values
        azi = survey_data['Azimuth'].values
        dls = survey_data['DogLegSeverity'].values

        # # Primary method: Find KOP using DLS criteria
        kop_index = np.argmax(dls > 1.5)

        # # Create result DataFrame
        result = pd.DataFrame({
            'MeasuredDepth': [md[kop_index]],
            'Inclination': [inc[kop_index]],
            'Azimuth': [azi[kop_index]],
            'Point': ['KOP']
        })
        # Fallback method if primary method fails validation
        invalid_x = result[~result['MeasuredDepth'].apply(is_valid_number)]
        if not invalid_x.empty:
            DEVIATED_INC_THRESHOLD = 5.0  # Degrees

            # Reinterpolate survey data
            inc_deg = np.degrees(inc)

            # Find KOP using inclination threshold
            kop_candidates = np.where(inc_deg > DEVIATED_INC_THRESHOLD)[0]
            kop_index = kop_candidates[0]
            # Create new result DataFrame
            result = pd.DataFrame({
                'MeasuredDepth': [md[kop_index]],
                'Inclination': [np.radians(inc_deg[kop_index])],
                'Azimuth': [azi[kop_index]],
                'Point': ['KOP']
            })
        return result[['MeasuredDepth', 'Inclination', 'Azimuth', 'Point']]


    def _find_kop_and_lp(
            self,
            df: pd.DataFrame,
            north_ref: str,
            rad_type: str
    ) -> pd.DataFrame:
        """Find kickoff point (KOP) and landing point (LP) in well trajectory data.

        Identifies critical trajectory points using inclination and dogleg severity criteria.
        Uses two different methods depending on data quality:
        1. Primary method using dogleg severity thresholds
        2. Fallback method using inclination thresholds

        Args:
            df: DataFrame containing survey data with columns:
                - MeasuredDepth: Measured depth values
                - Inclination: Inclination angles in radians
                - Azimuth: Azimuth angles in radians
            north_ref: North reference system identifier
            rad_type: Type of radian measurement to use

        Returns:
            DataFrame containing KOP and LP data with columns:
                - MeasuredDepth: MD at KOP and LP
                - Inclination: Inclination angles at KOP and LP
                - Azimuth: Azimuth angles at KOP and LP
                - Point: Point identifier ('KOP' or 'LP')

        Notes:
            - Primary method identifies:
                KOP: First point where DLS > 1.5°/100ft
                LP: First point where Inc > 85° and DLS < 1.0°/100ft
            - Fallback method identifies:
                KOP: First point where Inc > 5°
                LP: First point after KOP where Inc < 5°
            - All angles are processed in radians internally

        Examples:
            >>> kop_lp_data = _find_kop_and_lp(survey_df, 'true', 'azi_true_rad')
            >>> print(f"KOP depth: {kop_lp_data.loc[0, 'MeasuredDepth']:.2f}")
        """

        def is_valid_number(x: float) -> bool:
            """Validate numeric values."""
            return pd.notnull(x) and np.isreal(x) and x > 0

        # Initialize survey header with convergence angle
        header = FastSurveyHeader(
            azi_reference=_return_spelled_north_ref(north_ref.lower()),
            deg=False,
            convergence=math.radians(self.conv_angle)
        )

        # Create survey object with error model
        s = we.survey.Survey(
            md=df['MeasuredDepth'].values,
            inc=df['Inclination'].values,
            azi=df['Azimuth'].values,
            start_nev=self.start_nev,
            deg=False,
            header=header,
            error_model='ISCWSA MWD Rev4'
        )
        if len(s.md) < 30:
            s = s.interpolate_survey(step=100)

        # test_result = pd.DataFrame({
        #     'MeasuredDepth': s.md,
        #     'Inclination': s.inc_rad,
        #     'Azimuth': getattr(s, rad_type), 'DLS': s.dls
        # })
        # self.find_kop(test_result)
        # Interpolate survey at 10ft intervals using numpy
        md = np.arange(s.md[0], s.md[-1], 10)
        inc = np.interp(md, s.md, s.inc_rad)
        azi = np.interp(md, s.md, getattr(s, rad_type))
        dls = np.interp(md, s.md, s.dls)
        # Primary method: Find KOP and LP using DLS criteria
        kop_index = np.argmax(dls > 1.5)
        lp_index = np.argmax((np.degrees(inc) > 85) & (dls < 1.0))

        # Create result DataFrame
        result = pd.DataFrame({
            'MeasuredDepth': [md[kop_index], md[lp_index]],
            'Inclination': [inc[kop_index], inc[lp_index]],
            'Azimuth': [azi[kop_index], azi[lp_index]],
            'Point': ['KOP', 'LP']
        })

        # Fallback method if primary method fails validation
        invalid_x = result[~result['MeasuredDepth'].apply(is_valid_number)]
        if not invalid_x.empty:
            VERTICAL_INC_THRESHOLD = 5.0  # Degrees
            DEVIATED_INC_THRESHOLD = 5.0  # Degrees

            # Reinterpolate survey data
            inc_deg = np.degrees(inc)

            # Find KOP using inclination threshold
            kop_candidates = np.where(inc_deg > DEVIATED_INC_THRESHOLD)[0]
            kop_index = kop_candidates[0]
            print(md[kop_index])
            # Find LP using inclination threshold after KOP
            lp_candidates = np.where(inc_deg[kop_index:] < VERTICAL_INC_THRESHOLD)[0]
            lp_index = kop_index + lp_candidates[0]

            # Create new result DataFrame
            result = pd.DataFrame({
                'MeasuredDepth': [md[kop_index], md[lp_index]],
                'Inclination': [np.radians(inc_deg[kop_index]), np.radians(inc_deg[lp_index])],
                'Azimuth': [azi[kop_index], azi[lp_index]],
                'Point': ['KOP', 'LP']
            })

        return result[['MeasuredDepth', 'Inclination', 'Azimuth', 'Point']]


# def detect_angle_units(survey_data):
#     """
#     Detect whether inclination and azimuth values are in degrees or radians
#
#     Parameters:
#     survey_data: DataFrame with columns ['MD', 'Inclination', 'Azimuth']
#
#     Returns:
#     dict: Contains unit determination and confidence level
#     """
#     print(survey_data)
#     inc_values = survey_data['Inclination'].values
#
#     # Initialize confidence counters
#     radian_evidence = 0
#     degree_evidence = 0
#
#     # Check 1: Values greater than 2π (6.28...) strongly suggest degrees
#     if np.any(inc_values > 2 * np.pi):
#         print('foo')
#         degree_evidence += 3
#
#     # Check 2: Values greater than π (3.14...) in inclination suggest degrees
#     if np.any(inc_values > np.pi):
#         degree_evidence += 2
#
#     # Check 3: Typical inclination range check
#     inc_max = np.max(inc_values)
#     if 0 <= inc_max <= np.pi:
#         radian_evidence += 1
#     elif 0 <= inc_max <= 180:
#         degree_evidence += 1
#
#     # Check 5: Precision/step size check
#     inc_steps = np.diff(np.sort(np.unique(inc_values)))
#     if len(inc_steps) > 0:
#         avg_step = np.mean(inc_steps)
#         if 0.001 <= avg_step <= 0.1:  # Typical radian steps
#             radian_evidence += 1
#         elif 0.1 <= avg_step <= 5:  # Typical degree steps
#             degree_evidence += 1
#
#     # Make determination
#     unit_type = 'radians' if radian_evidence > degree_evidence else 'degrees'
#     confidence = abs(radian_evidence - degree_evidence)
#
#     result = {
#         'unit_type': unit_type,
#         'confidence': confidence,
#         'radian_evidence': radian_evidence,
#         'degree_evidence': degree_evidence
#     }
#     print(result)
#     return result

def solve_inclination(df):
    """
    Solve for inclination given MD, dog leg, RF, and TVD

    Parameters:
    df: DataFrame with columns ['MD', 'DogLeg', 'RF', 'TVD']

    Returns:
    DataFrame with added Inclination column
    """
    df = df.copy()
    inclination = np.zeros(len(df))

    # First point is typically vertical (0 inclination)
    inclination[0] = 0
    for idx, row in df.iterrows():
        delta_md = row['MeasuredDepth'][idx] - df['MeasuredDepth'].iloc[i - 1]
        print(df.iloc[idx])

    for i in range(1, len(df)):
        print(df['MeasuredDepth'].iloc[i])
        delta_md = df['MeasuredDepth'].iloc[i] - df['MeasuredDepth'].iloc[i - 1]
        delta_tvd = df['TVD'].iloc[i] - df['TVD'].iloc[i - 1]
        rf = df['RatioFactor'].iloc[i]
        dog_leg = df['DogLegSeverity'].iloc[i]

        # We know that:
        # delta_tvd = delta_md * ((cos(inc1) + cos(inc2))/2) * RF
        # where inc1 is known (previous inclination)
        # Solve for inc2 using iterative method

        inc1 = inclination[i - 1]

        def objective_function(inc2):
            return (delta_md * ((np.cos(inc1) + np.cos(inc2)) / 2) * rf) - delta_tvd

        # Initial guess for inc2
        inc2_guess = inc1 + dog_leg

        # Newton-Raphson method
        max_iter = 100
        tolerance = 1e-6
        inc2 = inc2_guess

        for _ in range(max_iter):
            f = objective_function(inc2)
            # Derivative of objective function with respect to inc2
            df = -delta_md * rf * np.sin(inc2) / 2

            if abs(df) < 1e-10:  # Avoid division by zero
                break

            inc2_new = inc2 - f / df

            if abs(inc2_new - inc2) < tolerance:
                inc2 = inc2_new
                break

            inc2 = inc2_new

        inclination[i] = inc2

    df['Inclination'] = inclination
    return df