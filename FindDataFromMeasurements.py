from typing import List, Tuple, Dict, Callable
from pyproj import Proj, CRS, Transformer, transform, Geod
from shapely import affinity
from shapely.geometry import Point, LineString
import math
import matplotlib.pylab as plt
import utm
import numpy as np


class SectionSolver:
    def __init__(self):
        # Known point (Top of Hole) - Example coordinates in UTM
        self.known_point = None
        self.ew_offset = None  # Distance from east boundary in feet
        self.ns_offset = None  # Distance from south boundary in feet
        self.ns_dir = None
        self.ew_dir = None
        # Boundary measurements in feet (clockwise from NE corner)
        self.east_segments = None  # North to South
        self.south_segments = None  # West to East
        self.west_segments = None  # South to North
        self.north_segments = None  # East to West

        # Conversion factor from feet to meters
        self.ft_to_m = 0.3048

    def set_known_point(self, coords: Tuple[float, float], ew_offset: float, ns_offset: float, ns_dir:str, ew_dir:str):
        """Set the known point coordinates and its offsets from boundaries"""
        self.known_point = coords
        # Convert offsets from feet to meters
        self.ew_offset = ew_offset * self.ft_to_m
        self.ns_offset = ns_offset * self.ft_to_m
        self.ns_dir = ns_dir
        self.ew_dir = ew_dir

    def set_boundary_segments(self, east: List[float], south: List[float],
                              west: List[float], north: List[float]):
        """Set the boundary segment measurements for each side, converted to meters"""
        self.east_segments = [seg * self.ft_to_m for seg in east]
        self.south_segments = [seg * self.ft_to_m for seg in south]
        self.west_segments = [seg * self.ft_to_m for seg in west]
        self.north_segments = [seg * self.ft_to_m for seg in north]

    def calculate_section_corners(self) -> dict:
        if self.ns_dir == 'FNL' and self.ew_dir == 'FWL':
            return self.calculate_section_corners_NW()
        elif self.ns_dir == 'FSL' and self.ew_dir == 'FEL':
            return self.calculate_section_corners_SE()
        elif self.ns_dir == 'FSL' and self.ew_dir == 'FWL':
            return self.calculate_section_corners_SW()
        elif self.ns_dir == 'FNL' and self.ew_dir == 'FEL':
            return self.calculate_section_corners_NE()

    def calculate_section_corners_SE(self) -> dict:
        """Calculate coordinates for all section corners"""
        # Calculate the SE corner using known point and offsets
        se_corner_easting = self.known_point[0] + self.ew_offset
        se_corner_northing = self.known_point[1] - self.ns_offset

        # Initialize corners dictionary
        corners = {
            'SE': (se_corner_easting, se_corner_northing)
        }

        # Calculate NE corner (moving north along east boundary segments)
        ne_corner_easting = se_corner_easting
        ne_corner_northing = se_corner_northing + sum(self.east_segments)
        corners['NE'] = (ne_corner_easting, ne_corner_northing)

        # Calculate NW corner (moving west along north boundary segments)
        nw_corner_easting = ne_corner_easting - sum(self.north_segments)
        nw_corner_northing = ne_corner_northing
        corners['NW'] = (nw_corner_easting, nw_corner_northing)

        # Calculate SW corner (moving south along west boundary segments)
        sw_corner_easting = nw_corner_easting
        sw_corner_northing = nw_corner_northing - sum(self.west_segments)
        corners['SW'] = (sw_corner_easting, sw_corner_northing)

        return corners
    def calculate_section_corners_NW(self) -> dict:
        """Calculate coordinates for all section corners starting from north and west."""
        # Calculate the NW corner using known point and offsets
        nw_corner_easting = self.known_point[0] - self.ew_offset
        nw_corner_northing = self.known_point[1] + self.ns_offset

        # Initialize corners dictionary
        corners = {
            'NW': (nw_corner_easting, nw_corner_northing)
        }

        # Calculate NE corner (moving east along north boundary segments)
        ne_corner_easting = nw_corner_easting + sum(self.north_segments)
        ne_corner_northing = nw_corner_northing
        corners['NE'] = (ne_corner_easting, ne_corner_northing)

        # Calculate SE corner (moving south along east boundary segments)
        se_corner_easting = ne_corner_easting
        se_corner_northing = ne_corner_northing - sum(self.east_segments)
        corners['SE'] = (se_corner_easting, se_corner_northing)

        # Calculate SW corner (moving west along south boundary segments)
        sw_corner_easting = se_corner_easting - sum(self.south_segments)
        sw_corner_northing = se_corner_northing
        corners['SW'] = (sw_corner_easting, sw_corner_northing)

        return corners

    def calculate_section_corners_SW(self) -> dict:
        """Calculate coordinates for all section corners, using west and south boundaries."""
        # Calculate the SW corner using the known point and offsets
        sw_corner_easting = self.known_point[0] - self.ew_offset
        sw_corner_northing = self.known_point[1] - self.ns_offset

        # Initialize corners dictionary
        corners = {
            'SW': (sw_corner_easting, sw_corner_northing)
        }

        # Calculate NW corner (moving north along west boundary segments)
        nw_corner_easting = sw_corner_easting
        nw_corner_northing = sw_corner_northing + sum(self.west_segments)
        corners['NW'] = (nw_corner_easting, nw_corner_northing)

        # Calculate NE corner (moving east along north boundary segments)
        ne_corner_easting = nw_corner_easting + sum(self.north_segments)
        ne_corner_northing = nw_corner_northing
        corners['NE'] = (ne_corner_easting, ne_corner_northing)

        # Calculate SE corner (moving south along east boundary segments)
        se_corner_easting = ne_corner_easting
        se_corner_northing = ne_corner_northing - sum(self.east_segments)
        corners['SE'] = (se_corner_easting, se_corner_northing)

        return corners

    def calculate_section_corners_NE(self) -> dict:
        """Calculate coordinates for all section corners"""
        # Calculate the NE corner using known point and offsets
        ne_corner_easting = self.known_point[0] + self.ew_offset
        ne_corner_northing = self.known_point[1] + self.ns_offset

        # Initialize corners dictionary
        corners = {
            'NE': (ne_corner_easting, ne_corner_northing)
        }

        # Calculate SE corner (moving south along east boundary segments)
        se_corner_easting = ne_corner_easting
        se_corner_northing = ne_corner_northing - sum(self.east_segments)
        corners['SE'] = (se_corner_easting, se_corner_northing)

        # Calculate SW corner (moving west along south boundary segments)
        sw_corner_easting = se_corner_easting - sum(self.south_segments)
        sw_corner_northing = se_corner_northing
        corners['SW'] = (sw_corner_easting, sw_corner_northing)

        # Calculate NW corner (moving north along west boundary segments)
        nw_corner_easting = sw_corner_easting
        nw_corner_northing = sw_corner_northing + sum(self.west_segments)
        corners['NW'] = (nw_corner_easting, nw_corner_northing)

        return corners

    def calculate_segment_points(self, corners: dict) -> dict:
        if self.ns_dir == 'FNL' and self.ew_dir == 'FWL':
            return self.calculate_segment_points_NW(corners)
        elif self.ns_dir == 'FSL' and self.ew_dir == 'FEL':
            return self.calculate_segment_points_SE(corners)
        elif self.ns_dir == 'FSL' and self.ew_dir == 'FWL':
            return self.calculate_segment_points_SW(corners)
        elif self.ns_dir == 'FNL' and self.ew_dir == 'FEL':
            return self.calculate_segment_points_NE(corners)
    def calculate_segment_points_SE(self, corners: dict) -> dict:
        """Calculate coordinates for all intermediate segment points"""
        segments = {'east': [], 'south': [], 'west': [], 'north': []}

        # Calculate east boundary segment points (moving north from SE corner)
        segments['east'].append(corners['SE'])  # Include SE corner
        for i, dist in enumerate(self.east_segments, 1):
            northing = corners['SE'][1] + sum(self.east_segments[:i])
            segments['east'].append((corners['SE'][0], northing))

        # Calculate south boundary segment points (moving east from SW corner)
        segments['south'].append(corners['SW'])  # Include SW corner
        for i, dist in enumerate(self.south_segments, 1):
            easting = corners['SW'][0] + sum(self.south_segments[:i])
            segments['south'].append((easting, corners['SW'][1]))

        # Calculate west boundary segment points (moving south from NW corner)
        segments['west'].append(corners['NW'])  # Include NW corner
        for i, dist in enumerate(self.west_segments, 1):
            northing = corners['NW'][1] - sum(self.west_segments[:i])
            segments['west'].append((corners['NW'][0], northing))

        # Calculate north boundary segment points (moving west from NE corner)
        segments['north'].append(corners['NE'])  # Include NE corner
        for i, dist in enumerate(self.north_segments, 1):
            easting = corners['NE'][0] - sum(self.north_segments[:i])
            segments['north'].append((easting, corners['NE'][1]))

        return segments  # class SectionSolver:

    def calculate_segment_points_NW(self, corners: dict) -> dict:
        """Calculate coordinates for all intermediate segment points based on north and west boundaries"""
        segments = {'north': [], 'east': [], 'south': [], 'west': []}

        # Calculate north boundary segment points (moving east from NW corner)
        segments['north'].append(corners['NW'])  # Include NW corner
        for i, dist in enumerate(self.north_segments, 1):
            easting = corners['NW'][0] + sum(self.north_segments[:i])
            segments['north'].append((easting, corners['NW'][1]))

        # Calculate east boundary segment points (moving south from NE corner)
        segments['east'].append(corners['NE'])  # Include NE corner
        for i, dist in enumerate(self.east_segments, 1):
            northing = corners['NE'][1] - sum(self.east_segments[:i])
            segments['east'].append((corners['NE'][0], northing))

        # Calculate south boundary segment points (moving west from SE corner)
        segments['south'].append(corners['SE'])  # Include SE corner
        for i, dist in enumerate(self.south_segments, 1):
            easting = corners['SE'][0] - sum(self.south_segments[:i])
            segments['south'].append((easting, corners['SE'][1]))

        # Calculate west boundary segment points (moving north from SW corner)
        segments['west'].append(corners['SW'])  # Include SW corner
        for i, dist in enumerate(self.west_segments, 1):
            northing = corners['SW'][1] + sum(self.west_segments[:i])
            segments['west'].append((corners['SW'][0], northing))

        return segments

    def calculate_segment_points_SW(self, corners: dict) -> dict:
        """Calculate coordinates for all intermediate segment points."""
        segments = {'west': [], 'north': [], 'east': [], 'south': []}

        # Calculate west boundary segment points (moving north from SW corner)
        segments['west'].append(corners['SW'])  # Include SW corner
        for i, dist in enumerate(self.west_segments, 1):
            northing = corners['SW'][1] + sum(self.west_segments[:i])
            segments['west'].append((corners['SW'][0], northing))

        # Calculate north boundary segment points (moving east from NW corner)
        segments['north'].append(corners['NW'])  # Include NW corner
        for i, dist in enumerate(self.north_segments, 1):
            easting = corners['NW'][0] + sum(self.north_segments[:i])
            segments['north'].append((easting, corners['NW'][1]))

        # Calculate east boundary segment points (moving south from NE corner)
        segments['east'].append(corners['NE'])  # Include NE corner
        for i, dist in enumerate(self.east_segments, 1):
            northing = corners['NE'][1] - sum(self.east_segments[:i])
            segments['east'].append((corners['NE'][0], northing))

        # Calculate south boundary segment points (moving west from SE corner)
        segments['south'].append(corners['SE'])  # Include SE corner
        for i, dist in enumerate(self.south_segments, 1):
            easting = corners['SE'][0] - sum(self.south_segments[:i])
            segments['south'].append((easting, corners['SE'][1]))

        return segments

    def calculate_segment_points_NE(self, corners: dict) -> dict:
        """Calculate coordinates for all intermediate segment points"""
        segments = {'east': [], 'south': [], 'west': [], 'north': []}

        # Calculate east boundary segment points (moving south from NE corner)
        segments['east'].append(corners['NE'])  # Include NE corner
        for i, dist in enumerate(self.east_segments, 1):
            northing = corners['NE'][1] - sum(self.east_segments[:i])
            segments['east'].append((corners['NE'][0], northing))

        # Calculate south boundary segment points (moving west from SE corner)
        segments['south'].append(corners['SE'])  # Include SE corner
        for i, dist in enumerate(self.south_segments, 1):
            easting = corners['SE'][0] - sum(self.south_segments[:i])
            segments['south'].append((easting, corners['SE'][1]))

        # Calculate west boundary segment points (moving north from SW corner)
        segments['west'].append(corners['SW'])  # Include SW corner
        for i, dist in enumerate(self.west_segments, 1):
            northing = corners['SW'][1] + sum(self.west_segments[:i])
            segments['west'].append((corners['SW'][0], northing))

        # Calculate north boundary segment points (moving east from NW corner)
        segments['north'].append(corners['NW'])  # Include NW corner
        for i, dist in enumerate(self.north_segments, 1):
            easting = corners['NW'][0] + sum(self.north_segments[:i])
            segments['north'].append((easting, corners['NW'][1]))

        return segments
# class SectionSolver:
#     def __init__(self):
#         # Known point (Top of Hole) - Example coordinates in UTM
#         self.known_point = None
#         self.east_offset = None  # Distance from east boundary in feet
#         self.south_offset = None  # Distance from south boundary in feet
#
#
#         # Boundary measurements in feet (clockwise from NE corner)
#         self.east_segments = None  # North to South
#         self.south_segments = None  # West to East
#         self.west_segments = None  # South to North
#         self.north_segments = None  # East to West
#
#         # Conversion factor from feet to meters
#         self.ft_to_m = 0.3048
#
#     def set_known_point(self, coords: Tuple[float, float], east_offset: float, south_offset: float):
#         """Set the known point coordinates and its offsets from boundaries"""
#         self.known_point = coords
#         # Convert offsets from feet to meters
#         self.east_offset = east_offset * self.ft_to_m
#         self.south_offset = south_offset * self.ft_to_m
#
#     def set_boundary_segments(self, east: List[float], south: List[float],
#                               west: List[float], north: List[float]):
#         """Set the boundary segment measurements for each side, converted to meters"""
#         self.east_segments = [seg * self.ft_to_m for seg in east]
#         self.south_segments = [seg * self.ft_to_m for seg in south]
#         self.west_segments = [seg * self.ft_to_m for seg in west]
#         self.north_segments = [seg * self.ft_to_m for seg in north]
#
#     def validate_measurements(self) -> bool:
#         """Validate that measurements are consistent"""
#         if not all([self.east_segments, self.south_segments, self.west_segments, self.north_segments]):
#             return False
#
#         # Check if total lengths of opposite sides are similar (within tolerance)
#         tolerance = 0.3048  # 1 foot tolerance in meters
#
#         east_total = sum(self.east_segments)
#         west_total = sum(self.west_segments)
#         north_total = sum(self.north_segments)
#         south_total = sum(self.south_segments)
#
#         return (abs(east_total - west_total) <= tolerance and
#                 abs(north_total - south_total) <= tolerance)
#
#     def calculate_section_corners(self) -> dict:
#         """Calculate coordinates for all section corners"""
#         # Calculate the SE corner using known point and offsets
#         se_corner_easting = self.known_point[0] + self.east_offset
#         se_corner_northing = self.known_point[1] - self.south_offset
#
#         # Initialize corners dictionary
#         corners = {
#             'SE': (se_corner_easting, se_corner_northing)
#         }
#
#         # Calculate NE corner (moving north along east boundary segments)
#         ne_corner_easting = se_corner_easting
#         ne_corner_northing = se_corner_northing + sum(self.east_segments)
#         corners['NE'] = (ne_corner_easting, ne_corner_northing)
#
#         # Calculate NW corner (moving west along north boundary segments)
#         nw_corner_easting = ne_corner_easting - sum(self.north_segments)
#         nw_corner_northing = ne_corner_northing
#         corners['NW'] = (nw_corner_easting, nw_corner_northing)
#
#         # Calculate SW corner (moving south along west boundary segments)
#         sw_corner_easting = nw_corner_easting
#         sw_corner_northing = nw_corner_northing - sum(self.west_segments)
#         corners['SW'] = (sw_corner_easting, sw_corner_northing)
#
#         return corners
#
#     def calculate_segment_points(self, corners: dict) -> dict:
#         """Calculate coordinates for all intermediate segment points"""
#         segments = {'east': [], 'south': [], 'west': [], 'north': []}
#
#         # Calculate east boundary segment points (moving north from SE corner)
#         segments['east'].append(corners['SE'])  # Include SE corner
#         for i, dist in enumerate(self.east_segments, 1):
#             northing = corners['SE'][1] + sum(self.east_segments[:i])
#             segments['east'].append((corners['SE'][0], northing))
#
#         # Calculate south boundary segment points (moving east from SW corner)
#         segments['south'].append(corners['SW'])  # Include SW corner
#         for i, dist in enumerate(self.south_segments, 1):
#             easting = corners['SW'][0] + sum(self.south_segments[:i])
#             segments['south'].append((easting, corners['SW'][1]))
#
#         # Calculate west boundary segment points (moving south from NW corner)
#         segments['west'].append(corners['NW'])  # Include NW corner
#         for i, dist in enumerate(self.west_segments, 1):
#             northing = corners['NW'][1] - sum(self.west_segments[:i])
#             segments['west'].append((corners['NW'][0], northing))
#
#         # Calculate north boundary segment points (moving west from NE corner)
#         segments['north'].append(corners['NE'])  # Include NE corner
#         for i, dist in enumerate(self.north_segments, 1):
#             easting = corners['NE'][0] - sum(self.north_segments[:i])
#             segments['north'].append((easting, corners['NE'][1]))
#
#         return segments# class SectionSolver:

def convertDataToGrid():
    all_data_processed = [
        ["West-Up2", 1341.12, 0, 52, 41, 3],
        ["West-Up1", 1322.88, 0, 4, 37, 4],
        ["West-Down1", 1325.20, 0, 14, 58, 1],
        ["West-Down2", 1320.05, 0, 11, 36, 3],
        ["East-Up2", 1322.94, 0, 8, 20, 3],
        ["East-Up1", 1322.58, 0, 25, 14, 3],
        ["East-Down1", 1321.37, 0, 6, 7, 3],
        ["East-Down2", 1323.47, 0, 7, 14, 1],
        ["North-Left2", 1347.56, 89, 1, 1, 1],
        ["North-Left1", 1317.26, 89, 1, 1, 1],
        ["North-Right1", 1317.26, 89, 1, 1, 1],
        ["North-Right2", 1324.49, 89, 41, 33, 4],
        ["South-Left2", 1336.58, 89, 23, 20, 1],
        ["South-Left1", 1322.66, 89, 18, 8, 1],
        ["South-Right1", 1323.64, 89, 27, 57, 1],
        ["South-Right2", 1328.48, 89, 24, 5, 1]
    ]

    utm_test = [[40.186001, -109.821559],
    [40.185937, -109.816738],
    [40.185875, -109.812025],
    [40.185812, -109.807312],
    [40.185792, -109.802573],
    [40.182161, -109.802585],
    [40.178532, -109.802621],
    [40.174906, -109.802630],
    [40.171274, -109.802621],
    [40.171313, -109.807374],
    [40.171347, -109.812109],
    [40.171391, -109.816840],
    [40.171431, -109.821621],
    [40.175053, -109.821605],
    [40.178690, -109.821626],
    [40.182321, -109.821632]]
    utm_real = [utm.from_latlon(i[0], i[1], 12, 'T') for i in utm_test]



    # all_data_processed = [['West-Up2', 0, 0, 0, 0, 0, 'T', '1703S04WU'],
    #                            ['West-Up1', 2629.41, 0, 18, 52, 4, 'T', '1703S04WU'],
    #                            ['West-Down1', 2631.58, 0, 11, 50, 4, 'T', '1703S04WU'],
    #                            ['West-Down2', 0, 0, 0, 0, 0, 'T', '1703S04WU'],
    #                            ['East-Up2', 0, 0, 0, 0, 0, 'T', '1703S04WU'],
    #                            ['East-Up1', 2624.22, 0, 18, 1, 4, 'T', '1703S04WU'],
    #                            ['East-Down1', 2620.81, 0, 13, 36, 4, 'T', '1703S04WU'],
    #                            ['East-Down2', 0, 0, 0, 0, 0, 'T', '1703S04WU'],
    #                            ['North-Left2', 0, 0, 0, 0, 0, 'T', '1703S04WU'],
    #                            ['North-Left1', 2670.48, 89, 17, 51, 3, 'T', '1703S04WU'],
    #                            ['North-Right1', 2670.52, 89, 17, 50, 3, 'T', '1703S04WU'],
    #                            ['North-Right2', 0, 0, 0, 0, 0, 'T', '1703S04WU'],
    #                            ['South-Left2', 0, 0, 0, 0, 0, 'T', '1703S04WU'],
    #                            ['South-Left1', 2674.6, 89, 9, 56, 3, 'T', '1703S04WU'],
    #                            ['South-Right1', 2667.27, 89, 5, 11, 3, 'T', '1703S04WU'],
    #                            ['South-Right2', 0, 0, 0, 0, 0, 'T', '1703S04WU']]
    # local_data_to_utm = utm.from_latlon(40.223541, -110.351490)[:2]
    # tsr_data = [1505, 'FNL', 135, 'FEL', None, None, local_data_to_utm[0], local_data_to_utm[1]]
    # utm_real = [[40.213063, -110.370073], [40.227500, -110.370153], [40.227675, -110.351033], [40.213282, -110.350955]]

    local_data_to_utm = utm.from_latlon(40.180351, -109.814491)[:2]
    tsr_data = [2024, 'FNL', 1995, 'FWL', None, None, local_data_to_utm[0], local_data_to_utm[1]]
    print(tsr_data)
    east_segments = [i[1] for i in all_data_processed if 'east' in i[0].lower()]
    west_segments = [i[1] for i in all_data_processed if 'west' in i[0].lower()]
    north_segments = [i[1] for i in all_data_processed if 'north' in i[0].lower()]
    south_segments = [i[1] for i in all_data_processed if 'south' in i[0].lower()]

    solver = SectionSolver()
    solver.set_known_point(coords = tsr_data[6:], ew_offset=tsr_data[2], ns_offset=tsr_data[0], ns_dir='FNL', ew_dir='FEL')
    solver.set_boundary_segments(east_segments, south_segments, west_segments, north_segments)

    # Calculate section corners and segment points
    corners = solver.calculate_section_corners()
    segments = solver.calculate_segment_points(corners)
    print(segments)


    all_points = [points for side, points in segments.items()]
    all_points = all_points[0] + all_points[1] + all_points[2] + all_points[3]
    point_used = tsr_data[-2:]


    ns_data = segments['south'] if tsr_data[1] == 'FSL' else segments['north']
    ew_data = segments['east'] if tsr_data[3] == 'FEL' else segments['west']

    # out_pt = findSurfaceCoordinates2(ew_data, ns_data, tsr_data)
    # print(out_pt)
    # print('prev')
    # diff = [out_pt[0] - basis_point[0], out_pt[1] - basis_point[1]]
    # for side, points in segments.items():
    #     points = [[i[0] + diff[0], i[1] + diff[1]] for i in points]


    fig, ax = plt.subplots()
    colors = ['cyan', 'red', 'yellow', 'blue']

    counter = 0
    x_utm, y_utm = [k[0] for k in utm_real], [k[1] for k in utm_real]


    # for i in range(len(x_utm)):
    #     plt.plot(x_utm, y_utm, c='black', linewidth=2)
    #     # plt.scatter(x_utm,y_utm, c='black', s = 5)
    #     counter += 1

    plt.scatter([local_data_to_utm[0]], local_data_to_utm[1], c='black')
    # for corner, coords in corners.items():
    #     plt.scatter([coords[0]], [coords[1]], c='black')
    for side, points in segments.items():
        x, y = [k[0] for k in points], [k[1] for k in points]
        plt.plot(x, y, c='red', linewidth=1)
        # plt.scatter(x,y, c='red')
        counter += 1

    plt.show()



convertDataToGrid()
