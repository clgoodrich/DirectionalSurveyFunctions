import sqlite3
import time

from scipy.spatial import ConvexHull
from rdp import rdp
import copy
from welltrajconvert.wellbore_trajectory import *
from shapely.geometry import Point, LineString, MultiPoint, Polygon
import welleng as we
from pyproj import Proj, Geod, Transformer, transform
import matplotlib.pyplot as plt
import numpy.typing as npt
from numpy.typing import NDArray
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
from typing import Optional, Tuple, Union, TypeVar
import pstats
from io import StringIO
import cProfile
from rtree import index  # R-tree for spatial indexing
from shapely.geometry import Point, LineString
from shapely.strtree import STRtree
from collections import ChainMap
def _reorganize_lst_points_with_angle(
        lst: List[List[float]],
        centroid: List[float]
) -> List[List[float]]:
    """Reorganizes a list of points by calculating angles relative to a centroid.

    Takes a list of 2D points and adds an angle component calculated from the 
    point's position relative to the centroid. The angle is calculated as the 
    arctangent of the vector from centroid to point, converted to degrees and 
    normalized to [0,360].

    Args:
        lst: List of points where each point is [x,y]
        centroid: Reference point [x,y] for angle calculations

    Returns:
        List of points with appended angles where each element is [x,y,angle]
        Angles are in degrees, normalized to [0,360] range

    Notes:
        - Uses atan2 for stable angle calculations in all quadrants
        - Adds 360 and takes modulo 360 to ensure positive angles
        - Preserves original point coordinates
        - Point order is maintained, only angles are appended

    Example:
        >>> points = [[1,1], [2,2]]
        >>> centroid = [0,0]
        >>> result = _reorganize_lst_points_with_angle(points, centroid)
        >>> print(result)
        [[1, 1, 45.0], [2, 2, 45.0]]
    """
    # Calculate angles relative to centroid and append to points
    lst_arrange = [
        list(i) + [
            (math.degrees(
                math.atan2(centroid[1] - i[1], centroid[0] - i[0])
            ) + 360) % 360
        ]
        for i in lst
    ]

    return lst_arrange

def _calculate_well_to_line_clearance_detailed(
        well_trajectory: Union[List[List[float]], npt.NDArray],
        line_points: Union[List[List[float]], npt.NDArray]
) -> List[Dict[str, Any]]:
    """Calculates detailed clearance metrics between well trajectory points and a line segment
    using vectorized operations.

    Computes minimum distances, intersection angles, and closest points on a line segment
    for an entire well trajectory using numpy vectorization for improved performance.

    Args:
        well_trajectory: Array-like of shape (n, 2) or (n, 3) containing well trajectory points.
            Each point should be [x, y] or [x, y, z]
        line_points: Array-like of shape (2, 2) or (2, 3) containing line segment endpoints.
            Should be two points [[x1, y1], [x2, y2]] or [[x1, y1, z1], [x2, y2, z2]]

    Returns:
        List of dictionaries containing detailed results for each trajectory point:
            - point_index: Index of the well trajectory point
            - well_point: Coordinates of the well point
            - distance: Minimum distance to line segment
            - closest_surface_point: Point on line segment closest to well point
            - intersection_angle: Acute angle between well-to-closest vector and line segment
            - original_segment: Original line segment coordinates

    Notes:
        - Uses vectorized numpy operations for improved performance
        - Projects points onto line segment using vector math
        - Clamps projection points to line segment bounds
        - Handles numerical stability with epsilon value
        - Returns acute angles (≤ 90°)
        - Uses L2 norm for distance calculations

    Example:
        >>> trajectory = [[0,0], [1,1], [2,2]]
        >>> line = [[0,2], [2,2]]
        >>> results = _calculate_well_to_line_clearance_detailed(trajectory, line)
        >>> print(f"Distance to first point: {results[0]['distance']:.2f}")
    """
    # Convert inputs to numpy arrays
    well_trajectory = np.array(well_trajectory)
    line_points = np.array(line_points)
    p1, p2 = line_points
    line_vector = p2 - p1
    line_length_squared = np.sum(line_vector ** 2)

    # Vectorized calculation of projection parameters
    well_points_diff = well_trajectory - p1
    t = np.divide(
        np.sum(well_points_diff * line_vector, axis=1),
        line_length_squared,
        where=line_length_squared != 0
    )

    # Clamp projection parameters to segment bounds
    t = np.clip(t, 0, 1)

    # Calculate closest points for all trajectory points
    closest_points = p1 + t[:, np.newaxis] * line_vector

    # Calculate distances using L2 norm
    distances = np.linalg.norm(well_trajectory - closest_points, axis=1)

    # Calculate intersection angles
    well_to_closest = well_trajectory - closest_points
    norm_well_to_closest = np.linalg.norm(well_to_closest, axis=1)
    norm_line_vector = np.linalg.norm(line_vector)

    # Calculate angles with numerical stability
    epsilon = 1e-10
    angle_cos = np.sum(well_to_closest * line_vector, axis=1) / (
            norm_well_to_closest * norm_line_vector + epsilon
    )
    angles = np.degrees(np.arccos(np.clip(angle_cos, -1, 1)))

    # Convert obtuse angles to acute
    angles = np.where(angles > 90, 180 - angles, angles)

    # Format results
    result = [
        {
            "point_index": i,
            "well_point": well_trajectory[i],
            "distance": distances[i],
            "closest_surface_point": closest_points[i],
            "intersection_angle": angles[i],
            "original_segment": line_points
        }
        for i in range(len(well_trajectory))
    ]

    return result

def _optimized_corner_process(
        trajectory: Union[List[List[float]], npt.NDArray]
) -> List[List[float]]:
    """Process a trajectory to identify and sort corner points using convex hull and RDP simplification.

    Takes a trajectory of points and identifies corner vertices through angle analysis,
    simplification, and centroid-based angle calculations for ordering.

    Args:
        trajectory: Array-like of shape (n, 2) containing trajectory points
            Each point should be [x, y] coordinates

    Returns:
        List of corner points with centroid angles where each element is [x, y, angle]
        Angles are in degrees normalized to [0,360] range

    Notes:
        - Uses ConvexHull for initial clockwise point ordering
        - Applies Ramer-Douglas-Peucker (RDP) algorithm for trajectory simplification
        - Epsilon value for RDP varies based on coordinate magnitudes:
            * 0.002 for coordinates between 35-55
            * 200 otherwise
        - Corner detection uses angle difference threshold of π/35 radians
        - Centroid angles calculated relative to positive x-axis
        - Points are ordered clockwise around centroid

    Implementation Details:
        1. Converts input to numpy array
        2. Calculates centroid using Shapely Polygon
        3. Orders points using ConvexHull
        4. Simplifies trajectory using adaptive RDP
        5. Detects corners through vector angle analysis
        6. Calculates centroid-relative angles
        7. Returns corners with angle information

    Example:
        >>> traj = [[0,0], [1,0], [1,1], [0,1]]
        >>> corners = _optimized_corner_process(traj)
        >>> print(f"Found {len(corners)} corners")
    """
    # Convert to numpy array for vector operations
    trajectory = np.array(trajectory)

    # Calculate polygon centroid
    centroid = Polygon(trajectory).centroid.coords[0]

    # Create clockwise point ordering using ConvexHull
    hull = ConvexHull(trajectory)
    trajectory = trajectory[hull.vertices]

    # Apply adaptive RDP simplification
    epsilon = 0.002 if (35 < trajectory[0, 0] < 55 or 35 < trajectory[0, 1] < 55) else 200
    simplified = rdp(np.vstack((trajectory, trajectory[0])), epsilon=epsilon)

    # Calculate sequential angle differences
    vectors = np.diff(simplified, axis=0)
    angles = np.arctan2(vectors[:, 1], vectors[:, 0])
    angle_diffs = np.diff(angles, append=angles[0] - 2 * np.pi)
    angle_diffs = np.abs(
        np.where(angle_diffs > np.pi, angle_diffs - 2 * np.pi, angle_diffs)
    )

    # Identify corners using angle threshold
    corners = simplified[1:][angle_diffs > np.pi / 35]

    # Calculate centroid-relative angles
    centroid_vectors = corners - centroid
    centroid_angles = (
                              np.degrees(
                                  np.arctan2(centroid_vectors[:, 1], centroid_vectors[:, 0])
                              ) + 360
                      ) % 360

    # Combine corner coordinates with angles
    result = np.column_stack((corners, centroid_angles))

    return result.tolist()

T = TypeVar('T', bound=List[Any])
def _remove_dupes_list_of_lists(lst: List[List[T]]) -> List[List[T]]:
    """Removes duplicate sublists while preserving the original order.

    Takes a list of lists and removes any duplicate sublists by converting each
    sublist to a tuple for hashing, maintaining the order of first occurrence.

    Args:
        lst: List of lists where each sublist contains hashable elements
            Example: [[1,2], [3,4], [1,2]] or [['a','b'], ['c','d']]

    Returns:
        A new list with duplicate sublists removed, preserving original order
        of first occurrence

    Notes:
        - Uses a set for O(1) lookup of previously seen items
        - Converts sublists to tuples for hashability
        - Maintains original sublist type in output
        - Memory usage is O(n) where n is number of unique sublists

    Example:
        >>> data = [[1,2], [3,4], [1,2], [5,6]]
        >>> result = _remove_dupes_list_of_lists(data)
        >>> print(result)  # [[1,2], [3,4], [5,6]]

    Raises:
        TypeError: If sublists contain unhashable elements
    """
    # Initialize data structures for tracking duplicates
    dup_free: List[List[T]] = []
    dup_free_set: set = set()

    # Process each sublist while maintaining order
    for x in lst:
        x_tuple = tuple(x)  # Convert to hashable type
        if x_tuple not in dup_free_set:
            dup_free.append(x)  # Keep original list type
            dup_free_set.add(x_tuple)

    return dup_free

def _remove_duplicates_preserve_order(points_list: List[T]) -> List[Tuple]:
    """Removes duplicate points while preserving order, converting results to tuples.

    Efficiently removes duplicate entries from a list of points/coordinates by
    converting to tuples for hashable comparison. Maintains original ordering
    while returning results as tuples.

    Args:
        points_list: List of lists containing coordinate/point data.
            Inner lists should contain comparable elements (typically numbers).

    Returns:
        List of tuples containing unique points in their original order
        of first appearance. All points are converted to tuples in output.

    Notes:
        - Uses set for O(1) lookup efficiency
        - Preserves first occurrence ordering
        - Converts all points to tuples in output
        - Memory usage is O(n) where n is number of unique points

    Examples:
        >>> coords = [[1,2], [3,4], [1,2], [5,6]]
        >>> _remove_duplicates_preserve_order(coords)
        [(1,2), (3,4), (5,6)]

        >>> points = [[0.5,1.0], [0.5,1.0], [2.0,3.0]]
        >>> _remove_duplicates_preserve_order(points)
        [(0.5,1.0), (2.0,3.0)]
    """
    # Initialize tracking set and result list
    seen: Set[tuple] = set()
    result: List[tuple] = []

    # Process each point
    for point in points_list:
        point_tuple = tuple(point)  # Convert to hashable type
        if point_tuple not in seen:
            result.append(point_tuple)  # Store as tuple
            seen.add(point_tuple)

    return result

def _consolidate_columns(
        df: pd.DataFrame,
        num_segments: int,
        dir_val: str
) -> pd.DataFrame:
    """Consolidates segmented distance measurements and related data into single columns.

    Takes a DataFrame with multiple segment columns and consolidates them based on
    minimum distance values, handling missing data appropriately.

    Args:
        df: Input DataFrame containing segmented measurements
            Must have columns formatted as:
            - distance{i}_{dir_val}
            - closest_surface_point{i}_{dir_val}
            - intersection_angle{i}_{dir_val}
            - segments{i}_{dir_val}
            where i ranges from 1 to num_segments
        num_segments: Number of segment columns to process
        dir_val: Direction value suffix for column names

    Returns:
        DataFrame with consolidated columns:
        - distance_{dir_val}
        - closest_surface_point_{dir_val}
        - intersection_angle_{dir_val}
        - segments_{dir_val}

    Notes:
        - Handles missing values by returning NaN for all fields if no valid distances
        - Selects values based on minimum distance when multiple valid segments exist
        - Preserves row order from input DataFrame
        - All consolidated columns include dir_val suffix

    Example:
        >>> df = pd.DataFrame({
        ...     'distance1_up': [1.0, np.nan],
        ...     'distance2_up': [2.0, 3.0]})
        >>> _consolidate_columns(df, 2, 'up')
    """
    # Initialize list to store consolidated results
    consolidated: List[Dict[str, Any]] = []

    # Process each row
    for _, row in df.iterrows():
        # Find valid segment indices (non-NaN distances)
        valid_indices = [i for i in range(1, num_segments + 1)
                         if pd.notna(row[f'distance{i}_{dir_val}'])]

        # Handle case with no valid distances
        if not valid_indices:
            consolidated.append({
                'distance': np.nan,
                'closest_surface_point': np.nan,
                'intersection_angle': np.nan,
                'segments': np.nan,
            })
        else:
            # Find index with minimum distance
            min_distance_index = min(valid_indices,
                                     key=lambda i: row[f'distance{i}_{dir_val}'])

            # Consolidate values from minimum distance segment
            consolidated.append({
                f'distance_{dir_val}': row[f'distance{min_distance_index}_{dir_val}'],
                f'closest_surface_point_{dir_val}': row[
                    f'closest_surface_point{min_distance_index}_{dir_val}'],
                f'intersection_angle_{dir_val}': row[f'intersection_angle{min_distance_index}_{dir_val}'],
                f'segments_{dir_val}': row[f'segments{min_distance_index}_{dir_val}']
            })

    return pd.DataFrame(consolidated)

def _process_row(
        row: pd.Series,
        num_segments: int,
        dir_val: str
) -> pd.Series:
    """Process row data to keep only the segment with angle closest to 90 degrees.

    Analyzes intersection angles across segments and preserves only the data from
    the segment whose angle is closest to 90 degrees, setting all other segments
    to NaN.

    Args:
        row: Pandas Series containing segment data with columns formatted as:
            - intersection_angle{i}_{dir_val}
            - distance{i}_{dir_val}
            - closest_surface_point{i}_{dir_val}
            - segments{i}_{dir_val}
            where i ranges from 1 to num_segments
        num_segments: Number of segments to process
        dir_val: Direction value suffix for column names

    Returns:
        Modified Pandas Series with all segments except the one closest
        to 90 degrees set to NaN

    Notes:
        - Modifies row data in-place
        - Uses absolute difference from 90 degrees for comparison
        - Sets all values to NaN for segments not closest to 90 degrees
        - Preserves original data structure and column names

    Example:
        >>> row = pd.Series({
        ...     'intersection_angle1_up': 85,
        ...     'intersection_angle2_up': 45,
        ...     'distance1_up': 1.0,
        ...     'distance2_up': 2.0
        ... })
        >>> processed = _process_row(row, 2, 'up')
        # Will keep segment 1 data (85 degrees) and set segment 2 to NaN
    """
    # Extract all intersection angles for comparison
    angles = [row[f'intersection_angle{i}_{dir_val}']
              for i in range(1, num_segments + 1)]

    # Find index of angle closest to 90 degrees
    closest_to_90 = min(range(len(angles)),
                        key=lambda i: abs(angles[i] - 90))

    # Set all segments except closest to 90 to NaN

    for i in range(1, num_segments + 1):
        if i != closest_to_90 + 1:  # Add 1 since segment numbering starts at 1
            row[f'distance{i}_{dir_val}'] = np.nan
            row[f'closest_surface_point{i}_{dir_val}'] = np.nan
            row[f'intersection_angle{i}_{dir_val}'] = np.nan
            row[f'segments{i}_{dir_val}'] = np.nan

    return row


def set_non_closest_to_nan(df, closest_to_90, num_segments, dir_val):
    # Create arrays of column names
    cols_prefix = ['distance', 'closest_surface_point', 'intersection_angle', 'segments']
    cols_to_nan = []

    # Generate all column names that need to be set to NaN
    for prefix in cols_prefix:
        for i in range(1, num_segments + 1):
            if i != closest_to_90 + 1:
                cols_to_nan.append(f'{prefix}{i}_{dir_val}')

    # Set all identified columns to NaN in one operation
    df[cols_to_nan] = np.nan

    return df

def process_results_vectorized_pandas(all_results_dicts, well_trajectory, segments, dir_val):
    # Create base DataFrame
    base_df = pd.DataFrame({
        'point_index': range(len(well_trajectory)),
        'well_point': list(well_trajectory)
    })

    # Process each segment's results
    for i, results in enumerate(all_results_dicts):
        seg_num = i + 1

        # Create segment specific DataFrame
        segment_data = pd.DataFrame({
            f'distance{seg_num}_{dir_val}': [round(r['distance'] / 0.3048, 2) for r in results],
            f'closest_surface_point{seg_num}_{dir_val}': [r['closest_surface_point'] for r in results],
            f'intersection_angle{seg_num}_{dir_val}': [r['intersection_angle'] for r in results],
            f'segments{seg_num}_{dir_val}': [segments[i]] * len(results)
        })

        # Combine with base DataFrame
        base_df = pd.concat([base_df, segment_data], axis=1)
    return base_df
    # return base_df.to_dict('records')


def process_clearance_results_pandas(well_trajectory_points, segments, dir_val):
    # Create base DataFrame
    base_df = pd.DataFrame({
        'point_index': range(len(well_trajectory_points)),
        'well_point': list(well_trajectory_points)
    })

    # Process each segment
    for i, segment in enumerate(segments):
        results = _calculate_well_to_line_clearance_detailed(well_trajectory_points, segment)

        # Create segment-specific columns
        segment_data = pd.DataFrame({
            f'distance{i + 1}_{dir_val}': [round(r['distance'] / 0.3048, 2) for r in results],
            f'closest_surface_point{i + 1}_{dir_val}': [r['closest_surface_point'] for r in results],
            f'intersection_angle{i + 1}_{dir_val}': [r['intersection_angle'] for r in results],
            f'segments{i + 1}_{dir_val}': [segment] * len(results)
        })

        # Add to base DataFrame
        base_df = pd.concat([base_df, segment_data], axis=1)

    return base_df.to_dict('records')


def _results_finder(
        segments: List[List[float]],
        dir_val: str,
        well_trajectory: List[Tuple[float, float, int]]
) -> pd.DataFrame:

    """Calculates and consolidates clearance measurements between well trajectory and surface segments.

    Processes multiple surface segments against well trajectory points to find distances,
    intersection angles, and closest points, then consolidates results into a DataFrame.

    Args:
        segments: List of surface line segments to check against well trajectory
        dir_val: Direction identifier used in column naming ('up'/'down'/etc)
        well_trajectory: List of tuples containing (x, y, index) for well points

    Returns:
        DataFrame containing consolidated results with columns:
            - point_index: Original well point index
            - well_point: (x,y) coordinates of well point
            - distance_{dir_val}: Minimum distance to surface
            - closest_surface_point_{dir_val}: Nearest point on surface
            - intersection_angle_{dir_val}: Angle of intersection
            - segments_{dir_val}: Associated surface segment

    Notes:
        - Distances are converted from meters to feet (divided by 0.3048)
        - Empty results generate placeholder DataFrame with same structure
        - Results are processed to keep only points closest to 90° intersection
        - All floating point distances are rounded to 2 decimal places

    Example:
        >>> segments = [[[0,0], [1,1]], [[2,2], [3,3]]]
        >>> well = [(0.5, 0.5, 1), (1.5, 1.5, 2)]
        >>> df = _results_finder(segments, 'up', well)
    """
    # Initialize results storage
    all_results: List[Dict[str, Any]] = []

    # Separate well trajectory components
    well_index: List[int] = [i[2] for i in well_trajectory]
    well_trajectory_points: List[List[float]] = [i[:2] for i in well_trajectory]

    # Process each segment against well trajectory

    for i, segment in enumerate(segments):
        # Calculate clearance details for current segment
        results = _calculate_well_to_line_clearance_detailed(well_trajectory_points, segment)

        # Store results for each well point
        for j, result in enumerate(results):
            # Initialize well point data on first segment
            if i == 0:
                all_results.append({
                    'point_index': well_index[j],
                    'well_point': well_trajectory_points[j]
                })

            # Add segment-specific results
            all_results[j].update({
                f'distance{i + 1}_{dir_val}': round(result['distance'] / 0.3048, 2),
                f'closest_surface_point{i + 1}_{dir_val}': result['closest_surface_point'],
                f'intersection_angle{i + 1}_{dir_val}': result['intersection_angle'],
                f'segments{i + 1}_{dir_val}': segments[0]
            })

    # Create DataFrame and set column order
    df = pd.DataFrame(all_results)
    column_order = ['point_index', 'well_point'] + [
        f'{col}{i}_{dir_val}' for i in range(1, len(segments) + 1)
        for col in ['distance', 'closest_surface_point', 'intersection_angle', 'segments']
    ]

    # Handle empty results case
    if df.empty:
        well_traj_pts = [[None, None] for _ in well_trajectory]
        placeholder_data = {
            'point_index': well_index,
            'well_point': well_trajectory_points,
            f'distance1_{dir_val}': [None] * len(well_trajectory),
            f'closest_surface_point1_{dir_val}': [None] * len(well_trajectory),
            f'intersection_angle1_{dir_val}': [None] * len(well_trajectory),
            f'segments1_{dir_val}': well_traj_pts
        }
        return pd.DataFrame(placeholder_data)

    # Process and consolidate results


    df = df[column_order]
    num_segments = len(segments)
    df = df.apply(lambda row: _process_row(row, num_segments, dir_val), axis=1)

    consolidated_df = _consolidate_columns(df, num_segments, dir_val)

    # Combine with well point information
    return pd.concat([df[['point_index', 'well_point']], consolidated_df], axis=1)





# def _results_finder(
#         segments: List[List[float]],
#         dir_val: str,
#         well_trajectory: List[Tuple[float, float, int]]
# ) -> pd.DataFrame:
#     print(segments)
#     print(dir_val)
#     print(well_trajectory.tolist())
#     """Calculates and consolidates clearance measurements between well trajectory and surface segments.
#
#     Processes multiple surface segments against well trajectory points to find distances,
#     intersection angles, and closest points, then consolidates results into a DataFrame.
#
#     Args:
#         segments: List of surface line segments to check against well trajectory
#         dir_val: Direction identifier used in column naming ('up'/'down'/etc)
#         well_trajectory: List of tuples containing (x, y, index) for well points
#
#     Returns:
#         DataFrame containing consolidated results with columns:
#             - point_index: Original well point index
#             - well_point: (x,y) coordinates of well point
#             - distance_{dir_val}: Minimum distance to surface
#             - closest_surface_point_{dir_val}: Nearest point on surface
#             - intersection_angle_{dir_val}: Angle of intersection
#             - segments_{dir_val}: Associated surface segment
#
#     Notes:
#         - Distances are converted from meters to feet (divided by 0.3048)
#         - Empty results generate placeholder DataFrame with same structure
#         - Results are processed to keep only points closest to 90° intersection
#         - All floating point distances are rounded to 2 decimal places
#
#     Example:
#         >>> segments = [[[0,0], [1,1]], [[2,2], [3,3]]]
#         >>> well = [(0.5, 0.5, 1), (1.5, 1.5, 2)]
#         >>> df = _results_finder(segments, 'up', well)
#     """
#     # Initialize results storage
#     all_results: List[Dict[str, Any]] = []
#
#     # Separate well trajectory components
#     well_index: List[int] = [i[2] for i in well_trajectory]
#     well_trajectory_points: List[List[float]] = [i[:2] for i in well_trajectory]
#
#     # Process each segment against well trajectory
#     for i, segment in enumerate(segments):
#         # Calculate clearance details for current segment
#         results = _calculate_well_to_line_clearance_detailed(well_trajectory_points, segment)
#
#         # Store results for each well point
#         for j, result in enumerate(results):
#             # Initialize well point data on first segment
#             if i == 0:
#                 all_results.append({
#                     'point_index': well_index[j],
#                     'well_point': well_trajectory_points[j]
#                 })
#
#             # Add segment-specific results
#             all_results[j].update({
#                 f'distance{i + 1}_{dir_val}': round(result['distance'] / 0.3048, 2),
#                 f'closest_surface_point{i + 1}_{dir_val}': result['closest_surface_point'],
#                 f'intersection_angle{i + 1}_{dir_val}': result['intersection_angle'],
#                 f'segments{i + 1}_{dir_val}': segments[0]
#             })
#
#     # Create DataFrame and set column order
#     df = pd.DataFrame(all_results)
#     column_order = ['point_index', 'well_point'] + [
#         f'{col}{i}_{dir_val}' for i in range(1, len(segments) + 1)
#         for col in ['distance', 'closest_surface_point', 'intersection_angle', 'segments']
#     ]
#
#     # Handle empty results case
#     if df.empty:
#         well_traj_pts = [[None, None] for _ in well_trajectory]
#         placeholder_data = {
#             'point_index': well_index,
#             'well_point': well_trajectory_points,
#             f'distance1_{dir_val}': [None] * len(well_trajectory),
#             f'closest_surface_point1_{dir_val}': [None] * len(well_trajectory),
#             f'intersection_angle1_{dir_val}': [None] * len(well_trajectory),
#             f'segments1_{dir_val}': well_traj_pts
#         }
#         return pd.DataFrame(placeholder_data)
#
#     # Process and consolidate results
#     df = df[column_order]
#     num_segments = len(segments)
#     df = df.apply(lambda row: _process_row(row, num_segments, dir_val), axis=1)
#     consolidated_df = _consolidate_columns(df, num_segments, dir_val)
#
#     # Combine with well point information
#     return pd.concat([df[['point_index', 'well_point']], consolidated_df], axis=1)
#


def _regular_corner_class(
        corners: List[List[float]],
        data_lengths: List[List[float]]
) -> List[List[List[float]]]:
    """Classifies and organizes polygon corner points into directional sides.

    Processes corner points and associated data points to group them into four sides
    (west, north, east, south) based on their angular positions relative to the polygon.

    Args:
        corners: List of corner points with format [x, y, angle]
        data_lengths: List of polygon points with format [x, y, angle]

    Returns:
        List containing four lists representing the sides in order:
        [west_side, north_side, east_side, south_side]
        Each side list contains points in proper geometric order

    Notes:
        - Angles are expected in degrees (0-360)
        - West side handles angle wrap-around at 0/360 degrees
        - Points are deduplicated while preserving order
        - Empty lists are returned for sides with missing corner points
    """

    def find_corner_point(
            corners: List[List[float]],
            min_angle: float,
            max_angle: float
    ) -> Optional[List[float]]:
        """Finds first corner point within specified angle range.

        Args:
            corners: List of corner points [x, y, angle]
            min_angle: Minimum angle in degrees (exclusive)
            max_angle: Maximum angle in degrees (inclusive)

        Returns:
            Corner point if found, None otherwise
        """
        return next((i for i in corners if min_angle < i[-1] <= max_angle), None)

    def find_side_points(
            data_lengths: List[List[float]],
            start_angle: float,
            end_angle: float,
            reverse: bool = False
    ) -> List[List[float]]:
        """Extracts points between start and end angles.

        Args:
            data_lengths: List of points [x, y, angle]
            start_angle: Starting angle in degrees (inclusive)
            end_angle: Ending angle in degrees (inclusive)
            reverse: If True, reverses point order

        Returns:
            Deduplicated list of points in specified order
        """
        points = [i for i in data_lengths if start_angle <= i[-1] <= end_angle]
        return _remove_duplicates_preserve_order(points[::-1] if reverse else points)

    def get_side(
            data_lengths: List[List[float]],
            corners: List[List[float]],
            start_angle: float,
            end_angle: float,
            reverse: bool = False
    ) -> List[List[float]]:
        """Extracts points forming one side of the polygon.

        Args:
            data_lengths: All polygon points
            corners: Corner points only
            start_angle: Starting angle for side
            end_angle: Ending angle for side
            reverse: If True, reverses point order

        Returns:
            List of points forming the requested side
        """
        start_point = find_corner_point(corners, start_angle, end_angle)
        end_point = find_corner_point(corners, start_angle - 90, start_angle)

        if start_point is None or end_point is None:
            return []

        start_idx = data_lengths.index(start_point)
        end_idx = data_lengths.index(end_point)

        return find_side_points(data_lengths,
                                data_lengths[end_idx][-1],
                                data_lengths[start_idx][-1],
                                reverse)

    # Process cardinal sides
    south_side = get_side(data_lengths, corners, 90, 180, reverse=True)
    east_side = get_side(data_lengths, corners, 180, 270, reverse=True)
    north_side = get_side(data_lengths, corners, 270, 360, reverse=True)

    # Handle west side angle wrap-around
    nw_point = find_corner_point(corners, 270, 360)
    sw_point = find_corner_point(corners, 0, 90)

    # Construct west side handling 0/360 degree boundary
    if nw_point is not None and sw_point is not None:
        nw_idx = data_lengths.index(nw_point)
        sw_idx = data_lengths.index(sw_point)
        west_side = [sw_point] + [
            i for i in data_lengths
            if (i[-1] > data_lengths[nw_idx][-1] or i[-1] < data_lengths[sw_idx][-1])
               and i not in (east_side + south_side + north_side)
        ] + [nw_point]
        west_side = _remove_duplicates_preserve_order(west_side)
    else:
        west_side = []

    return [west_side, north_side, east_side, south_side]

def _corner_generator_process(
        data_lengths: List[List[float]]
) -> Tuple[List[List[float]], List[List[List[float]]]]:
    """Processes polygon points to identify and classify corners.

    Takes a list of polygon points, identifies corners, calculates angles relative
    to centroid, and organizes points into directional sides.

    Args:
        data_lengths: List of polygon points, each containing [x, y] coordinates

    Returns:
        Tuple containing:
            - List of identified corner points with format [x, y, angle]
            - List of four lists representing polygon sides (west, north, east, south)
              where each side contains its constituent points

    Notes:
        - Uses Shapely Polygon centroid for angle calculations
        - Angles are measured clockwise from 0-360 degrees
        - Duplicate points are removed while preserving order
        - Corner optimization is performed before angle calculations

    Example:
        >>> points = [[0,0], [1,0], [1,1], [0,1]]
        >>> corners, sides = _corner_generator_process(points)
        >>> print(len(corners))  # Number of detected corners
        4
    """
    # Optimize corner point detection
    corner_arrange = _optimized_corner_process(data_lengths)

    # Calculate polygon centroid for angle references
    centroid = Polygon(data_lengths).centroid
    centroid_point = [centroid.x, centroid.y]

    # Calculate angles relative to centroid
    corner_arrange = _reorganize_lst_points_with_angle(
        [i[:2] for i in corner_arrange],
        centroid_point
    )

    # Sort and deduplicate corner points
    corners = sorted(corner_arrange, key=lambda r: r[-1])
    corners = _remove_dupes_list_of_lists(corners)

    # Process all points with angles
    data_lengths = _reorganize_lst_points_with_angle(data_lengths, centroid_point)
    data_lengths = sorted(data_lengths, key=lambda r: r[-1])

    # Ensure consistent list format
    corners = [list(i) for i in corners]

    # Classify points into directional sides
    all_data = _regular_corner_class(corners, data_lengths)

    return corners, all_data

def _id_sides(polygon: List[List[float]]) -> Tuple[List[List[List[float]]], ...]:
    """Identifies and segments the sides of a polygon into directional components.

    Takes a polygon defined by points and returns segmented lists of points organized
    by cardinal direction (right/east, left/west, up/north, down/south). Points are
    sorted and paired into segments for each side.

    Args:
        polygon: List of [x,y] coordinates defining the polygon vertices in order

    Returns:
        Tuple containing four lists of segments, in order:
            - right_lst_segments: List of point pairs for eastern side
            - left_lst_segments: List of point pairs for western side
            - up_lst_segments: List of point pairs for northern side
            - down_lst_segments: List of point pairs for southern side
        Each segment is a pair of [x,y] coordinates defining start and end points

    Notes:
        - Uses _corner_generator_process() to identify corners and classify sides
        - Points are sorted based on appropriate coordinate for each direction:
          * East/West sides sort by y-coordinate
          * North/South sides sort by x-coordinate
        - Segments are created as sequential pairs of sorted points
    """
    # Process corners and generate initial side classifications
    corners, sides_generated = _corner_generator_process(polygon)

    # Remove angle information from classified points
    sides_generated = [[j[:-1] for j in i] for i in sides_generated]

    # Extract directional sides
    left_lst, up_lst, right_lst, down_lst = (
        sides_generated[0],  # West
        sides_generated[1],  # North
        sides_generated[2],  # East
        sides_generated[3]  # South
    )

    # Sort points appropriately for each direction
    left_lst = sorted(left_lst, key=lambda x: x[1])  # Sort west points by y
    up_lst = sorted(up_lst, key=lambda x: x[0])  # Sort north points by x
    right_lst = sorted(right_lst, key=lambda x: x[1], reverse=True)  # Sort east points by y descending
    down_lst = sorted(down_lst, key=lambda x: x[0], reverse=True)  # Sort south points by x descending

    # Generate segments as sequential point pairs
    right_lst_segments = [[right_lst[i], right_lst[i + 1]] for i in range(len(right_lst) - 1)]
    left_lst_segments = [[left_lst[i], left_lst[i + 1]] for i in range(len(left_lst) - 1)]
    down_lst_segments = [[down_lst[i], down_lst[i + 1]] for i in range(len(down_lst) - 1)]
    up_lst_segments = [[up_lst[i], up_lst[i + 1]] for i in range(len(up_lst) - 1)]

    return right_lst_segments, left_lst_segments, up_lst_segments, down_lst_segments


class ClearanceProcess:
    """Processes clearance data for well surveys and plats.

    This class handles the processing of well survey data in relation to plat boundaries
    and adjacent plats, calculating concentrations and clearance metrics.

    Attributes:
        df (pd.DataFrame): Survey data with shape points
        plats (pd.DataFrame): Plat boundary data
        adjacent_plats (pd.DataFrame): Data for adjacent plat boundaries
        all_polygons_concs (np.ndarray): Unique concentration values
        whole_df (pd.DataFrame): Complete processed dataset
        clearance_data (pd.DataFrame): Filtered clearance results
        used_conc (List[Union[str, float]]): List of used concentration values

    Notes:
        - Expects input DataFrames to have specific column structure
        - 'shp_pt' column must contain shapely Point objects
        - Plat DataFrames must include geometry and centroid columns
    """

    def __init__(
            self,
            df_used: pd.DataFrame,
            df_plat: pd.DataFrame,
            adjacent_plats: pd.DataFrame
    ) -> None:
        """Initialize the ClearanceProcess with survey and plat data.

        Args:
            df_used: DataFrame containing survey points and associated data
            df_plat: DataFrame containing plat boundary information
            adjacent_plats: DataFrame containing adjacent plat information

        Notes:
            - Automatically calculates concentrations upon initialization
            - Creates empty whole_df for later processing
            - Triggers _main_clearance processing during initialization

        Raises:
            ValueError: If required columns are missing from input DataFrames
            TypeError: If geometry objects are not properly formatted
        """
        # Store input DataFrames
        self.df = df_used
        self.plats = df_plat
        self.adjacent_plats = adjacent_plats

        # Calculate concentrations for each survey point
        self.df['Conc'] = self.df['shp_pt'].apply(self._find_conc)

        # Extract unique concentration values
        self.all_polygons_concs = df_used['Conc'].unique()

        # Initialize empty DataFrame for complete dataset
        self.whole_df = pd.DataFrame()

        # Process clearance data
        # analyzeTimeNoArgs(self._main_clearance)
        self.clearance_data = self._main_clearance()

        # Extract used concentration values
        self.used_conc = self.clearance_data['Conc'].unique().tolist()

    def _main_clearance(self) -> pd.DataFrame:
        """Processes clearance calculations for well trajectories against plat boundaries.

        Calculates distances from well trajectory points to the boundaries of their
        containing plats in all cardinal directions (FNL, FSL, FEL, FWL).

        Returns:
            pd.DataFrame: Original survey data merged with calculated boundary distances
                Contains columns:
                - All original survey columns
                - point_index: Index of trajectory point
                - FNL: Distance to north line
                - FSL: Distance to south line
                - FEL: Distance to east line
                - FWL: Distance to west line

        Notes:
            - Processes each concentration (Conc) group separately
            - Segments plat boundaries into directional components
            - Calculates minimum distances to each boundary
            - Merges directional results into comprehensive dataset
            - Handles missing plat geometries with error reporting

        Raises:
            IndexError: When plat geometry is missing for a concentration
        """
        # Process each unique concentration
        # time2 = time.perf_counter()
        # sum_time += (time2 - time1)
        # print(time2 - time1)
        # print('sum_time', sum_time)
        for i in range(len(self.all_polygons_concs)):
            # Extract trajectory points for current concentration
            well_traj = self.df[self.df['Conc'] == self.all_polygons_concs[i]]
            well_trajectory = well_traj[['Easting', 'Northing', 'point_index']].values

            # Get polygon geometry for current concentration
            try:
                used_poly = list(
                    self.plats[self.plats['Conc'] == self.all_polygons_concs[i]]
                    ['geometry'].values.tolist()[0].exterior.coords
                )
            except IndexError:
                print('index error on guisurvey tab')  # Consider logging instead
                continue

            # Generate directional boundary segments
            right_lst_segments, left_lst_segments, up_lst_segments, down_lst_segments = _id_sides(used_poly)
            # Calculate distances to each boundary
            # analyzeTime2(_results_finder, [left_lst_segments, 'West', well_trajectory])
            #

            left_df = _results_finder(left_lst_segments, 'West', well_trajectory)
            right_df = _results_finder(right_lst_segments, 'East', well_trajectory)
            down_df = _results_finder(down_lst_segments, 'South', well_trajectory)
            up_df = _results_finder(up_lst_segments, 'North', well_trajectory)

            # Merge north-south and east-west results
            up_down = pd.merge(up_df, down_df, on='point_index')
            left_right = pd.merge(left_df, right_df, on='point_index')

            # Clean up intermediate columns
            up_down = up_down.drop(columns=['well_point_y'])
            left_right = left_right.drop(columns=['well_point_y'])

            # Merge all directional results
            all_data = pd.merge(up_down, left_right, on='point_index')
            all_data = all_data.drop(columns=['well_point_x_y'])
            all_data = all_data.rename(columns={'well_point_x_x': 'well_point'})

            # Accumulate results
            self.whole_df = pd.concat([all_data, self.whole_df]).reset_index(drop=True)

        # Final processing and column renaming
        self.whole_df = self.whole_df.sort_values(by=self.whole_df.columns[0])
        self.whole_df = self.whole_df.rename(
            columns={
                'distance_East': 'FEL',
                'distance_West': 'FWL',
                'distance_North': 'FNL',
                'distance_South': 'FSL'
            }
        )

        # Extract relevant columns and merge with original data
        edited_df = self.whole_df[['point_index', 'FNL', 'FSL', 'FEL', "FWL"]]
        result = pd.merge(self.df, edited_df, on='point_index')

        return result


    def _find_conc(self, point: Point) -> Optional[Any]:
        """Finds the concentration value for a point by checking containing plat polygons.

        Iterates through adjacent plats to find which polygon contains the given point,
        returning that plat's concentration value. Used for assigning survey points to
        their containing plat regions.

        Args:
            point: A Shapely Point object representing the location to check

        Returns:
            The concentration value from the containing plat polygon, or None if the
            point is not contained within any plat

        Notes:
            - Performs sequential scan of all adjacent plats
            - Returns first matching concentration found
            - Returns None if point falls outside all plat boundaries
            - Depends on adjacent_plats DataFrame having 'geometry' and 'Conc' columns
        """
        # Iterate through adjacent plats to find containing polygon
        for idx, row in self.adjacent_plats.iterrows():
            if row['geometry'].contains(point):
                return row['Conc']

        # No containing polygon found
        return None
