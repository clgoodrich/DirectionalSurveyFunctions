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
        * measured_depth: Measured depth values
        * inclination: inclination angles
        * azimuth: azimuth angles
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
        df: A pandas DataFrame containing a 'measured_depth' column with survey data.
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
        >>> data = pd.DataFrame({'measured_depth': [100, 200], 'Value': [1, 2]})
        >>> result = _check_and_insert_zero_md(data)
        >>> print(result['measured_depth'].tolist())
        [0, 100, 200]
    """
    # Check if first row's measured depth is not zero
    if df['measured_depth'].iloc[0] != 0:
        # Create a copy of the first row to preserve all other values
        first_row = df.iloc[0].copy()
        # Modify the measured depth to zero
        first_row['measured_depth'] = 0
        # Concatenate the new zero md row with the original dataframe
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
        header: 'FastSurveyHeader'
) -> Tuple['we.survey.Survey', pd.DataFrame]:
    """Creates and automatically interpolates a wellbore survey object based on depth ratio.

    This function processes raw survey data, creating a welleng Survey object with
    automatic interpolation based on the total depth of the well. The interpolation
    step is calculated to maintain sufficient survey point density for accurate
    trajectory representation.

    Args:
        df (pd.DataFrame): Survey data containing:
            - measured_depth (float): Measured depth in feet
            - inclination (float): inclination angle in radians
            - azimuth (float): azimuth angle in radians
        start_nev (Tuple[float, float, float]): Starting position coordinates:
            - North offset in feet
            - East offset in feet
            - Vertical depth in feet (tvd)
        header (FastSurveyHeader): Survey header object containing:
            - azi_reference: azimuth reference system
            - convergence: Grid convergence in radians
            - deg: Boolean flag for angle units

    Returns:
        Tuple[we.survey.Survey, pd.DataFrame]: A tuple containing:
            - survey: Processed welleng Survey object
            - df: Interpolated DataFrame matching survey points

    Raises:
        ValueError: If required columns are missing from input DataFrame
        ValueError: If start_nev contains invalid coordinates

    Notes:
        - Interpolation is automatic based on depth_ratio = max_depth/50
        - Uses ISCWSA MWD Rev4 error model for uncertainty calculations
        - All angular inputs must be in radians (deg=False)
        - Interpolation preserves original survey points
        - Step size is fixed at 50 feet for interpolated points

    Example:
        >>> survey_data = pd.DataFrame({
        ...     'measured_depth': [0, 1000, 2000],
        ...     'inclination': [0, 0.2, 0.4],
        ...     'azimuth': [1.5, 1.5, 1.5]
        ... })
        >>> survey, interpolated_df = _create_survey(
        ...     survey_data,
        ...     (0, 0, 0),
        ...     header
        ... )
    """
    # Input validation
    required_columns = ['measured_depth', 'inclination', 'azimuth']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"DataFrame must contain columns: {required_columns}")

    if not all(isinstance(x, (int, float)) for x in start_nev):
        raise ValueError("start_nev must contain numeric values")

    # Create base survey object with error model
    survey = we.survey.Survey(
        md=df['measured_depth'],  # Measured depth array
        inc=df['inclination'],  # inclination array in radians
        azi=df['azimuth'],  # azimuth array in radians
        start_nev=start_nev,  # Starting position tuple
        deg=False,  # Angles are in radians
        header=header,  # Survey header object
        error_model='ISCWSA MWD Rev4'  # Industry standard error model
    )

    # Calculate interpolation requirements
    max_depth = df['measured_depth'].max()
    depth_ratio = max_depth / 50  # One point every 50 feet recommended

    # Interpolate if survey point density is insufficient
    if len(df) < depth_ratio:
        survey = survey.interpolate_survey(step=50)
        df = pd.DataFrame({
            'measured_depth': survey.md,
            'inclination': survey.inc_rad,
            'azimuth': survey.azi_true_rad
        })

    return survey, df

# def _create_survey(
#         df: pd.DataFrame,
#         start_nev: Tuple[float, float, float],
#         header: 'FastSurveyHeader',
# ) -> 'we.survey.Survey':
#     """Create and optionally interpolate a wellbore survey object.
#
#     Initializes a Survey object from measured depth, inclination, and azimuth data
#     with specified starting position and reference parameters.
#
#     Args:
#         df: DataFrame containing survey data with columns:
#             - measured_depth: Measured depth values
#             - inclination: inclination angles
#             - azimuth: azimuth angles
#         start_nev: Tuple of starting position coordinates (north, east, vertical)
#         header: FastSurveyHeader object containing reference system settings
#
#     Returns:
#         we.survey.Survey: Survey object containing well trajectory data and calculated parameters
#
#     Notes:
#         - All angular inputs should be in radians (deg=False)
#         - Uses ISCWSA MWD Rev4 error model for uncertainty calculations
#         - Interpolation creates regular spacing between survey points
#         - Original survey points are preserved in interpolation
#
#     Examples:
#         >>> survey = _create_survey(df, (0,0,0), header, True, 100)
#         >>> print(len(survey.md))  # Will show interpolated point count
#     """
#     # Initialize base survey object
#     survey = we.survey.Survey(
#         md=df['measured_depth'],
#         inc=df['inclination'],
#         azi=df['azimuth'],
#         start_nev=start_nev,
#         deg=False,
#         header=header,
#         error_model='ISCWSA MWD Rev4'
#     )
#     max_depth = df['measured_depth'].max()
#     depth_ratio = max_depth/50
#     if len(df) < depth_ratio:
#         survey = survey.interpolate_survey(step=50)
#         df = pd.DataFrame({
#                     'measured_depth': survey.md,
#                     'inclination': survey.inc_rad,
#                     'azimuth': survey.azi_true_rad})
#     return survey, df

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
            - measured_depth: Array of measured depth values
            - inclination: Array of inclination angles
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
        - azimuth values are extracted dynamically using getattr
        - Position calculations start from provided start_nev coordinates
        - Used primarily for well path position and geometric calculations

    Examples:
        >>> min_curve = _process_min_curve(df, survey, 'azi_true_rad', (0,0,0))
        >>> print(min_curve.dls)  # Get dogleg severity values
    """
    return we.utils.MinCurve(
        md=df['measured_depth'],
        inc=df['inclination'],
        azi=getattr(survey, rad_type),
        unit='feet',
        start_xyz=start_nev
    )


def _create_output_df(
        survey: 'we.survey.Survey',
        min_curve: 'we.utils.MinCurve',
        utm_vals: npt.NDArray[np.float64],
        latlons: npt.NDArray[np.float64],
        deg_type: str
) -> pd.DataFrame:
    """Creates a comprehensive DataFrame containing all computed survey and position data.

    Consolidates wellbore trajectory calculations, spatial coordinates, and derived
    measurements into a standardized DataFrame format for analysis and export.
    Handles coordinate system transformations and unit conversions.

    Args:
        survey (we.survey.Survey): Survey object containing:
            - Basic trajectory calculations
            - Tool orientation data
            - Vertical section values
        min_curve (we.utils.MinCurve): Minimum curvature calculations including:
            - Ratio factors
            - Delta values
            - Dogleg severity
        utm_vals (npt.NDArray[np.float64]): UTM coordinate array of shape (n,2):
            - Column 0: easting values
            - Column 1: northing values
        latlons (npt.NDArray[np.float64]): Geographic coordinate array of shape (n,2):
            - Column 0: Latitude values
            - Column 1: Longitude values
        deg_type (str): azimuth reference type to use from survey object
            Valid values: 'azi_true_deg', 'azi_grid_deg', 'azi_mag_deg'

    Returns:
        pd.DataFrame: DataFrame containing aligned arrays for:
            Survey Measurements:
                - measured_depth: Float, measured depth along wellbore
                - inclination: Float, wellbore inclination in degrees
                - azimuth: Float, wellbore azimuth in degrees
                - tvd: Float, true vertical depth
            Position Data:
                - N/E Offset: Float, surface-referenced offsets
                - easting/northing: Float, UTM coordinates
                - Lat/Lon: Float, geographic coordinates
            Geometric Calculations:
                - dls: Float, wellbore turn severity
                - build_rate: Float, inclination change rate
                - turn_rate: Float, azimuth change rate
            MinCurve Results:
                - ratio_factor: Float, minimum curvature ratio
                - Delta values: Float, section-wise changes
            Tool Orientation:
                - ToolFace: Float, tool orientation angle
            Cartesian Positions:
                - position_x/Y: Float, transformed coordinates
                - depth_actual: Float, vertical component

    Raises:
        ValueError: If coordinate arrays don't match survey length
        ValueError: If deg_type is not a valid azimuth reference

    Notes:
        - All angular measurements are in degrees
        - Position values maintain original input units
        - Cartesian positions use transformed coordinate system
        - Arrays are aligned by measured depth index

    Example:
        >>> survey_df = _create_output_df(
        ...     survey,
        ...     min_curve,
        ...     utm_coordinates,
        ...     latlon_pairs,
        ...     'azi_true_deg'
        ... )
        >>> print(survey_df[['measured_depth', 'tvd', 'lat', 'lon']])
    """
    # Input validation
    if len(utm_vals) != len(survey.md) or len(latlons) != len(survey.md):
        raise ValueError("Coordinate arrays must match survey length")

    # Unpack coordinate arrays for clarity and efficiency
    lats, lons = latlons.T
    x, y, z = min_curve.poss.T
    easting, northing = utm_vals.T

    # Create comprehensive output dataframe with all survey and position data
    return pd.DataFrame({
        # Survey measurements
        'measured_depth': survey.md,
        'inclination': min_curve.inc,
        'azimuth': min_curve.azi,
        'tvd': survey.z,

        # MinCurve calculations
        'ratio_factor': min_curve.rf,
        'delta_z': min_curve.delta_z,
        'delta_y': min_curve.delta_y,
        'delta_x': min_curve.delta_x,
        'delta_md': min_curve.delta_md,

        # Position data
        'N Offset': survey.y,
        'E Offset': survey.x,
        'easting': easting,
        'northing': northing,
        'lat': lats,
        'lon': lons,

        # Tool orientation
        'ToolFace': survey.toolface,
        'vertical_section': survey.vertical_section,

        # Geometric calculations
        'dls': min_curve.dls,
        'build_radius': survey.radius,
        'build_rate': survey.build_rate,
        'turn_rate': survey.turn_rate,

        # Transformed cartesian positions
        'position_x': y,  # Coordinate system transformed
        'position_y': x,  # Coordinate system transformed
        'depth_actual': z
    })

# def _create_output_df(
#         survey: 'we.survey.Survey',
#         min_curve: 'we.utils.MinCurve',
#         utm_vals: npt.NDArray[np.float64],
#         latlons: npt.NDArray[np.float64],
#         deg_type: str
# ) -> DataFrame:
#     """Create a dictionary containing all computed survey and position data.
#
#     Organizes survey calculations, position data, and derived measurements into a
#     standardized dictionary format for analysis and export.
#
#     Args:
#         survey: Survey object containing basic survey calculations
#         min_curve: MinCurve object with minimum curvature calculations
#         utm_vals: Array of UTM coordinates (easting, northing pairs)
#         latlons: Array of latitude/longitude coordinates
#         deg_type: String specifying which degree attribute to use from survey
#
#     Returns:
#         Dictionary containing arrays for:
#             - Survey measurements (md, Inc, Azi, tvd)
#             - Position data (N/E offsets, UTM, Lat/Lon)
#             - Geometric calculations (dls, Build Rate, Turn Rate)
#             - Minimum curvature results (RF, deltas)
#             - Tool orientation (Tool Face)
#             - Cartesian positions (X, Y, Z)
#
#     Notes:
#         - All angular values are in degrees
#         - Position values maintain original input units
#         - Arrays are aligned by measured depth
#         - X/Y/Z positions are transformed from min_curve coordinate system
#
#     Examples:
#         >>> output = _create_output_df(survey, min_curve, utm, latlon, tf, 'azi_true_deg')
#         >>> print(output['tvd'])  # Access true vertical depth array
#     """
#     # Unpack coordinate arrays for clarity
#     lats, lons = latlons.T
#     x, y, z = min_curve.poss.T
#     easting, northing = utm_vals.T
#     # Create comprehensive output dataframe with all survey and position data
#
#     return pd.DataFrame({
#         'measured_depth': survey.md,
#         'inclination': min_curve.inc,
#         'azimuth': min_curve.azi,
#         'tvd': survey.z,
#         'ratio_factor': min_curve.rf,
#         'N Offset': survey.y,
#         'E Offset': survey.x,
#         'ToolFace': survey.toolface,
#         'vertical_section': survey.vertical_section,
#         'easting': easting,
#         'northing': northing,
#         'Lat': lats,
#         'Lon': lons,
#         'delta_z': min_curve.delta_z,
#         'delta_y': min_curve.delta_y,
#         'delta_x': min_curve.delta_x,
#         'delta_md': min_curve.delta_md,
#         'dls': min_curve.dls,
#         'build_radius': survey.radius,
#         'build_rate': survey.build_rate,
#         'turn_rate': survey.turn_rate,
#         'position_x': y,  # Note coordinate system transformation
#         'position_y': x,  # Note coordinate system transformation
#         'depth_actual': z
#     })
#

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
            - azimuth: Well azimuth in degrees
            - inclination: Well inclination in degrees
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
        for col in ['azimuth', 'inclination']:
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
                - measured_depth: Survey measured depths

        Returns:
            pd.DataFrame: Input DataFrame with additional column:
                - feature: String identifier of geological feature/formation

        Notes:
            - Uses self.drilled_depths which must contain:
                - interval: Depth ranges for features
                - feature: Names/labels for geological features
            - intervals are converted to pandas intervalIndex with left-closed bounds
            - Points not matching any interval are labeled as 'Unknown'
            - Each depth point can only belong to one feature interval

        Examples:
            >>> depths_df = pd.DataFrame({
            ...     'interval': [(0, 100), (100, 200)],
            ...     'feature': ['Surface', 'Reservoir']
            ... })
            >>> survey_df = pd.DataFrame({'measured_depth': [50, 150]})
            >>> processed = drilled_depths_process(survey_df)
            >>> print(processed['feature'])
            0    Surface
            1    Reservoir
            :param df:
            :param drilled_depths:
        """

        # Convert intervals to left-closed intervalIndex
        drilled_depths['interval'] = pd.IntervalIndex(
            drilled_depths['interval'],
            closed='left'
        )

        # Create mapping dictionary from intervals to features
        interval_to_feature = dict(zip(
            drilled_depths['interval'],
            drilled_depths['feature']
        ))

        # Assign features based on depth intervals
        df['feature'] = df['measured_depth'].apply(
            lambda x: next((interval_to_feature[interval]
                            for interval in interval_to_feature
                            if x in interval),
                           'Unknown')
        )
        df = df[['feature', 'measured_depth']]
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
                - n: northing coordinates in US survey feet
                - e: easting coordinates in US survey feet

        Notes:
            - Uses an azimuthal Equidistant projection (aeqd)
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
            'measured_depth', 'inclination', 'azimuth', 'tvd', 'N Offset', 'E Offset',
            'vertical_section', 'delta_z', 'delta_y', 'delta_x', 'delta_md'
        ]

        # Initialize reference system settings
        ref_settings = _get_reference_settings(ref_type)
        north_ref = ref_settings['north_ref']
        rad_type = ref_settings['rad_type']
        deg_type = ref_settings['deg_type']

        # Setup survey header and process kickoff points
        header = _setup_survey_header(north_ref, self.conv_angle)

        # Combine and clean survey data
        self.df = self.df_referenced.drop_duplicates(subset='measured_depth', keep='first')
        self.df = self.df.sort_values('measured_depth').reset_index(drop=True)

        # Process survey calculations
        survey_used, self.df = _create_survey(self.df, self.start_nev, header)
        proposed_azimuth = survey_used.survey_deg[-1][2]

        # Calculate minimum curvature and coordinate transformations
        min_curve = _process_min_curve(self.df, survey_used, rad_type, self.start_nev)
        utm_vals, latlons = _solve_utm(self.df['measured_depth'], self.start_lat, self.start_lon, min_curve)

        # Process tool face angles and create output structure
        df = _create_output_df(survey_used, min_curve, utm_vals, latlons, deg_type)

        # Round specified columns
        df[columns_to_round] = df[columns_to_round].round(2)
        init_md = self.df['measured_depth'].tolist()# + self.kop_lp['measured_depth'].tolist()
        df = df[df['measured_depth'].isin(init_md)]

        # Add shape points and process indices
        df = df.reset_index(drop=True)
        df['shp_pt'] = df.apply(lambda row: Point(row['easting'], row['northing']), axis=1)
        df['point_index'] = df.index

        # Organize final column structure
        final_columns = [
            'feature', 'measured_depth', 'inclination', 'azimuth', 'tvd', 'vertical_section',
            'ratio_factor', 'dls', 'delta_md', 'build_radius', 'build_rate', 'turn_rate',
            'delta_x', 'delta_y', 'delta_z', 'position_x', 'position_y', 'depth_actual', 'easting',
            'northing', 'lat', 'lon', 'shp_pt', 'point_index'
        ]
        df = df.reindex(columns=final_columns)

        return df, proposed_azimuth

    def find_kick_off_point(
            self,
            survey_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Identifies the kick-off point (KOP) in a directional wellbore survey using
        both dls and inclination rate of change analysis.

        The function determines the KOP using two methods:
        1. Primary: First point where dls exceeds 1.5°/100ft
        2. Secondary: First point where inclination rate of change becomes consistently positive

        Args:
            survey_data (pd.DataFrame): Survey data containing:
                - measured_depth (float): Measured depth in feet
                - inclination (float): inclination angle in radians
                - azimuth (float): azimuth angle in radians
                - dls (float): Dogleg severity in degrees/100ft
                - tvd (float): True vertical depth in feet

        Returns:
            pd.DataFrame: Single-row DataFrame containing KOP information:
                - md (float): Measured depth at KOP in feet
                - inclination (float): inclination at KOP in radians
                - azimuth (float): azimuth at KOP in radians
                - Point (str): Point identifier ('KOP')

        Raises:
            ValueError: If required columns are missing from survey_data
            ValueError: If no valid KOP is found

        Notes:
            - The function prioritizes dls-based detection (>1.5°/100ft)
            - Falls back to inclination gradient analysis if dls method fails
            - Sorts data by measured depth before analysis
            - Assumes continuous survey data without large gaps

        Example:
            >>> survey = pd.DataFrame({
            ...     'measured_depth': [0, 100, 200, 300],
            ...     'inclination': [0, 0.01, 0.05, 0.1],
            ...     'azimuth': [1.5, 1.5, 1.6, 1.7],
            ...     'dls': [0, 0.5, 1.8, 2.0],
            ...     'tvd': [0, 99.9, 199.5, 298.0]
            ... })
            >>> kop = find_kick_off_point(survey)
            >>> print(f"KOP at {kop['md'].values[0]} ft md")
        """
        # Input validation
        required_columns = ['measured_depth', 'inclination', 'azimuth', 'dls', 'tvd']
        if not all(col in survey_data.columns for col in required_columns):
            raise ValueError(f"Survey data must contain columns: {required_columns}")

        # Ensure data is sorted by Measured Depth
        survey_data = survey_data.sort_values(by="measured_depth").reset_index(drop=True)

        # Primary method: dls threshold
        dls_kop = survey_data[survey_data['dls'] > 1.5]
        if not dls_kop.empty:
            kop_idx = dls_kop.index[0]
        else:
            # Secondary method: inclination rate of change
            survey_data["dInc_dmd"] = np.gradient(
                survey_data["inclination"],
                survey_data["measured_depth"]
            )

            # Find first point of consistent positive inclination change
            inc_changes = survey_data[survey_data["dInc_dmd"] > 0]
            if inc_changes.empty:
                raise ValueError("No valid kick-off point found in survey data")
            kop_idx = inc_changes.index[0]

        # Extract KOP data
        kick_off_point = survey_data.iloc[kop_idx]
        kop_data = pd.DataFrame({
            "md": [kick_off_point["measured_depth"]],
            "inclination": [kick_off_point["inclination"]],
            "azimuth": [kick_off_point["azimuth"]],
            "Point": ["KOP"]
        })

        return kop_data

    def find_landing_point(
            self,
            survey_data: pd.DataFrame,
            target_inclination: float = np.pi / 2,
            tol: float = 0.1
    ) -> pd.DataFrame:
        """Identifies the landing point (LP) in a directional wellbore survey where the well
        path stabilizes at the target inclination.

        The function uses first and second derivatives of inclination with respect to measured
        depth to identify where the wellbore stabilizes at the target angle. Landing is
        determined by finding where rate of inclination change approaches zero and the
        inclination matches the target value within tolerance.

        Args:
            survey_data (pd.DataFrame): Survey data containing:
                - measured_depth (float): Measured depth in feet
                - inclination (float): inclination angle in radians
                - azimuth (float): azimuth angle in radians
                - dls (float): Dogleg severity in degrees/100ft
                - tvd (float): True vertical depth in feet
            target_inclination (float, optional): Target stabilization inclination in radians.
                Defaults to π/2 (horizontal).
            tol (float, optional): Tolerance for inclination matching in radians.
                Defaults to 0.1.

        Returns:
            pd.DataFrame: Single-row DataFrame containing LP information:
                - md (float): Measured depth at LP in feet
                - inclination (float): inclination at LP in radians
                - azimuth (float): azimuth at LP in radians
                - Point (str): Point identifier ('LP')

        Raises:
            ValueError: If required columns are missing from survey_data
            ValueError: If no valid landing point is found

        Notes:
            - Uses second derivative analysis to detect inclination stabilization
            - Prioritizes points matching target inclination within tolerance
            - Falls back to deepest stabilization point if exact match not found
            - Assumes continuous survey data without large gaps

        Example:
            >>> survey = pd.DataFrame({
            ...     'measured_depth': [1000, 2000, 3000, 4000],
            ...     'inclination': [0.2, 1.2, 1.55, 1.57],
            ...     'azimuth': [1.5, 1.5, 1.6, 1.6],
            ...     'dls': [1.2, 1.5, 0.2, 0.1],
            ...     'tvd': [990, 1800, 2400, 2450]
            ... })
            >>> lp = find_landing_point(survey)
            >>> print(f"Landing Point at {lp['md'].values[0]} ft md")
        """
        # Input validation
        required_columns = ['measured_depth', 'inclination', 'azimuth']
        if not all(col in survey_data.columns for col in required_columns):
            raise ValueError(f"Survey data must contain columns: {required_columns}")

        # Ensure data is sorted by Measured Depth
        survey_data = survey_data.sort_values(by="measured_depth").reset_index(drop=True)

        # Calculate first and second derivatives of inclination
        survey_data["dInc_dmd"] = np.gradient(
            survey_data["inclination"],
            survey_data["measured_depth"]
        )
        survey_data["d2Inc_dmd2"] = np.gradient(
            survey_data["dInc_dmd"],
            survey_data["measured_depth"]
        )

        # Find stabilization points (where second derivative approaches zero)
        stabilization_mask = np.isclose(survey_data["d2Inc_dmd2"], 0, atol=1e-6)
        stabilization_points = survey_data[stabilization_mask]

        if stabilization_points.empty:
            raise ValueError("No stabilization points found in survey data")

        # Check for points near target inclination
        target_mask = np.isclose(
            stabilization_points["inclination"],
            target_inclination,
            atol=tol
        )
        horizontal_points = stabilization_points[target_mask]

        # Select landing point
        landing_point = (horizontal_points.iloc[0] if not horizontal_points.empty
                         else stabilization_points.iloc[-1])

        # Format return data
        lp_data = pd.DataFrame({
            "md": [landing_point["measured_depth"]],
            "inclination": [landing_point["inclination"]],
            "azimuth": [landing_point["azimuth"]],
            "Point": ["LP"]
        })

        return lp_data
