import math
import sqlite3
from shapely import wkt
# import ModuleAgnostic as ma
import pandas as pd
import numpy as np
import welleng as we
from welleng.architecture import String
import ast
from typing import Optional, Dict, Any, Union, Literal, NoReturn, List


def calculateSoloMapsBurstLoadDF(section: Dict[str, Union[float, int]]) -> Dict[str, Union[float, int]]:
    """Calculates Maximum Anticipated Surface Pressure (MAPS), burst load, and burst design
    factor for a single casing section.

    Computes critical pressure values and safety factors for burst design considering mud
    weights, fracture pressures, and depth-based calculations. Updates the section
    dictionary with calculated values.

    Args:
        section: Dictionary containing casing section data with keys:
            mud_weight (float): Mud weight in ppg
            backup_mud (float): Backup mud weight in ppg
            tvd (float): True vertical depth in feet
            frac_init_pressure (float): Formation fracture initiation pressure in psi
            int_gradient (float): Internal pressure gradient
            burst_strength (float): Casing burst strength rating in psi

    Returns:
        Dict[str, Union[float, int]]: Updated section dictionary with added keys:
            maps (float): Maximum Anticipated Surface Pressure in psi
            burst_load (float): Calculated burst load in psi
            burst_df (float): Burst design factor (dimensionless)

    Notes:
        - Uses conversion factor 0.05194806 for ppg to psi/ft
        - Burst load calculation considers:
            * Differential pressure from mud weights
            * Formation fracture pressures
            * Internal/external pressure gradients
        - Design factor calculations handle zero burst load case
        - Pressure containment hierarchy:
            1. Differential mud weight pressures
            2. Fracture initiation pressures
            3. MAPS with internal gradient effects

    Example:
        >>> section = {
        ...     'mud_weight': 15.0,
        ...     'backup_mud': 8.6,
        ...     'tvd': 10000,
        ...     'frac_init_pressure': 8000,
        ...     'int_gradient': 0.1,
        ...     'burst_strength': 10000
        ... }
        >>> updated = calculateSoloMapsBurstLoadDF(section)
        >>> print(f"MAPS: {updated['maps']:.0f} psi")
        MAPS: 7792 psi
    """
    # Calculate Maximum Anticipated Surface Pressure
    maps = section['mud_weight'] * section['tvd'] * 0.05194806

    # Calculate burst load considering multiple pressure scenarios
    burst_load = max(
        # Differential pressure from mud weights
        (0.05194806 * (section['mud_weight'] - section['backup_mud']) * section['tvd']),
        # Minimum of fracture pressure and MAPS effects
        min(
            (section['frac_init_pressure'] - (0.05194806 * section['backup_mud'] * section['tvd'])),
            maps - section['int_gradient'] * (section['tvd'] - section['tvd']) -
            (0.05194806 * section['backup_mud'] * section['tvd'])
        )
    )

    # Calculate burst design factor, handling zero burst load case
    burst_df = float(section['burst_strength']) / float(burst_load) if float(burst_load) > 0 else float('inf')

    # Update section dictionary with calculated values
    section.update({'maps': maps, 'burst_load': burst_load, 'burst_df': burst_df})
    return section
# def calculateSoloMapsBurstLoadDF(section):
#     maps = section['mud_weight'] * section['tvd'] * 0.05194806
#     burst_load = max((0.05194806 * (section['mud_weight'] - section['backup_mud']) * section['tvd']), min((section['frac_init_pressure'] - (0.05194806 * section['backup_mud'] * section['tvd'])),maps - section['int_gradient'] * (section['tvd'] - section['tvd']) - (0.05194806 * section['backup_mud'] * section['tvd'])))
#     burst_df = float(section['burst_strength']) / float(burst_load) if float(burst_load) > 0 else float('inf')
#     section.update({'maps': maps, 'burst_load': burst_load, 'burst_df': burst_df})
#     return section

def calculate_maps(sec1: Dict[str, float], sec2: Dict[str, float]) -> float:
    """Calculates Maximum Anticipated Pressure at Shoe (MAPS) between two adjacent
    casing sections considering bottomhole pressure and pressure gradients.

    Computes MAPS by calculating the hydrostatic pressure at the deeper section's
    total vertical depth and accounting for the pressure differential between
    sections due to internal gradient effects.

    Args:
        sec1: Dictionary containing upper casing section data with keys:
            tvd (float): True vertical depth of upper section in feet
        sec2: Dictionary containing lower casing section data with keys:
            mud_weight (float): Mud weight in ppg
            tvd (float): True vertical depth of lower section in feet
            int_gradient (float): Internal pressure gradient in psi/ft

    Returns:
        float: Maximum Anticipated Pressure at Shoe in psi

    Notes:
        - Uses conversion factor 0.05194806 for ppg to psi/ft
        - Calculation steps:
            1. Calculate bottomhole pressure at lower section
            2. Account for pressure differential between sections
            3. Consider internal gradient effects
        - Assumes:
            * Vertical wellbore sections
            * Static mud conditions
            * No temperature effects
            * No formation pressure effects

    Example:
        >>> sec1 = {'tvd': 5000}
        >>> sec2 = {
        ...     'mud_weight': 15.0,
        ...     'tvd': 10000,
        ...     'int_gradient': 0.1
        ... }
        >>> maps = calculate_maps(sec1, sec2)
        >>> print(f"MAPS: {maps:.0f} psi")
        MAPS: 7292 psi
    """
    # Calculate bottomhole pressure at lower section
    next_bhp = sec2['mud_weight'] * sec2['tvd'] * 0.05194806

    # Calculate MAPS considering depth differential and internal gradient
    maps = next_bhp - (sec2['tvd'] - sec1['tvd']) * sec2['int_gradient']

    return maps

# def calculate_maps(sec1, sec2):
#     next_bhp = sec2['mud_weight'] * sec2['tvd'] * 0.05194806
#     maps = next_bhp - (sec2['tvd'] - sec1['tvd']) * sec2['int_gradient']
#     return maps

def calculate_burst_load(sec1: Dict[str, float], sec2: Dict[str, float], maps: float) -> float:
    """Calculates maximum burst load between casing sections considering differential
    pressures, fracture pressures, and maximum anticipated surface pressures.

    Computes the highest pressure differential that could be experienced by the casing
    by evaluating multiple pressure scenarios and selecting the most severe case.

    Args:
        sec1: Dictionary containing upper casing section data with keys:
            mud_weight (float): Primary mud weight in ppg
            backup_mud (float): Backup/kill mud weight in ppg
            tvd (float): True vertical depth in feet
            frac_init_pressure (float): Formation fracture initiation pressure in psi
            int_gradient (float): Internal pressure gradient in psi/ft
        sec2: Dictionary containing lower section data with keys:
            int_gradient (float): Internal pressure gradient in psi/ft
        maps: float: Maximum Anticipated Surface Pressure in psi

    Returns:
        float: Maximum calculated burst load in psi

    Notes:
        - Uses conversion factor 0.05194806 for ppg to psi/ft
        - Calculation considers three scenarios:
            1. Differential pressure from mud weights
            2. Fracture pressure minus internal gradients
            3. MAPS minus internal gradients
        - Takes maximum value between:
            * Mud weight differential pressure
            * Minimum of fracture and MAPS scenarios
        - Assumes:
            * Static wellbore conditions
            * No temperature effects
            * Vertical wellbore sections

    Example:
        >>> sec1 = {
        ...     'mud_weight': 15.0,
        ...     'backup_mud': 8.6,
        ...     'tvd': 10000,
        ...     'frac_init_pressure': 8000,
        ...     'int_gradient': 0.1
        ... }
        >>> sec2 = {'int_gradient': 0.1}
        >>> maps = 7500
        >>> burst_load = calculate_burst_load(sec1, sec2, maps)
        >>> print(f"Burst Load: {burst_load:.0f} psi")
        Burst Load: 3325 psi
    """
    # Calculate differential pressure from mud weights
    part1 = 0.05194806 * (sec1['mud_weight'] - sec1['backup_mud']) * sec1['tvd']

    # Calculate minimum pressure considering fracture initiation
    minPart1 = sec1['frac_init_pressure'] - (sec1['tvd'] - sec1['tvd']) * sec2['int_gradient'] - \
               (0.05194806 * sec1['backup_mud'] * sec1['tvd'])

    # Calculate minimum pressure considering MAPS
    minPart2 = maps - sec1['int_gradient'] * (sec1['tvd'] - sec1['tvd']) - \
               (0.05194806 * sec1['backup_mud'] * sec1['tvd'])

    # Determine maximum burst load from all scenarios
    max_all = max(part1, min(minPart1, minPart2))

    return max_all
# def calculate_burst_load(sec1, sec2, maps):
#     part1 = 0.05194806 * (sec1['mud_weight'] - sec1['backup_mud']) * sec1['tvd']
#     minPart1 = sec1['frac_init_pressure'] - (sec1['tvd'] - sec1['tvd']) * sec2['int_gradient'] - (0.05194806 * sec1['backup_mud'] * sec1['tvd'])
#     minPart2 = maps - sec1['int_gradient'] * (sec1['tvd'] - sec1['tvd']) - (0.05194806 * sec1['backup_mud'] * sec1['tvd'])
#     max_all = max(part1, min(minPart1, minPart2))
#     return max_all


class WellBoreExpanded(String):
    """A specialized wellbore class that extends the String class with additional depth parameters.

    This class enhances the base String class by adding support for measured depth (MD),
    true vertical depth (TVD), and top of liner (TOL) measurements specific to wellbore operations.

    Args:
        name (str): Identifier for the wellbore collection.
        top (float): Shallowest measured depth (meters) at collection top.
        bottom (float): Deepest measured depth (meters) at collection bottom.
        max_md_depth (float): Maximum measured depth of wellbore (meters).
        max_tvd_depth (float): Maximum true vertical depth of wellbore (meters).
        tol (float, optional): Top of liner depth (meters). Defaults to 0.0.
        *args: Variable length argument list passed to parent class.
        method (Literal["bottom_up", "top_down"], optional): Section addition strategy.
            Defaults to "bottom_up".
        **kwargs: Arbitrary keyword arguments passed to parent class.

    Attributes:
        max_md_depth (float): Maximum measured depth stored as float.
        max_tvd_depth (float): Maximum true vertical depth stored as float.
        tol (float): Top of liner depth stored as float.
        relationships (Dict[str, Any]): Storage for wellbore relationship mappings.

    Raises:
        ValueError: If depth values are invalid or inconsistent.

    Example:
        >>> wellbore = WellBoreExpanded(
        ...     name="WELL-A",
        ...     top=0.0,
        ...     bottom=1000.0,
        ...     max_md_depth=1200.0,
        ...     max_tvd_depth=1150.0,
        ...     tol=950.0
        ... )
    """

    def __init__(
            self,
            name: str,
            top: float,
            bottom: float,
            max_md_depth: float,
            max_tvd_depth: float,
            tol: float = 0.0,
            *args: Any,
            method: Literal["bottom_up", "top_down"] = "bottom_up",
            **kwargs: Any
    ) -> None:
        # Initialize parent class with basic parameters
        super().__init__(name, top, bottom, method=method, *args, **kwargs)

        # Convert and store depth parameters as floats
        self.max_md_depth: float = float(max_md_depth)
        self.max_tvd_depth: float = float(max_tvd_depth)
        self.tol: float = float(tol)  # Top of liner depth

        # Validate all depth parameters for consistency
        self._validate_initial_parameters()

        # Initialize storage for wellbore relationships
        self.relationships: Dict[str, Any] = {}
    # def __init__(self, name, top, bottom, max_md_depth, max_tvd_depth, tol=0.0, *args, method="bottom_up", **kwargs):
    #     """
    #     Extends the `String` class to include additional parameters: max_md_depth, max_tvd_depth, and tol (top of liner).
    #
    #     Parameters
    #     ----------
    #     name : str
    #         The name of the collection.
    #     top : float
    #         The shallowest measured depth at the top of the collection of items in meters.
    #     bottom : float
    #         The deepest measured depth at the bottom of the collection of items in meters.
    #     max_md_depth : float
    #         The maximum measured depth for the wellbore.
    #     max_tvd_depth : float
    #         The maximum true vertical depth for the wellbore.
    #     tol : float, optional
    #         Top of liner depth, defaults to 0.0.
    #     method : str, optional
    #         Method to add sections ('top_down' or 'bottom_up'), by default "bottom_up".
    #     """
    #     super().__init__(name, top, bottom, method=method, *args, **kwargs)
    #     # Assign additional parameters
    #     self.max_md_depth = float(max_md_depth)
    #     self.max_tvd_depth = float(max_tvd_depth)
    #     self.tol = float(tol)  # Top of liner
    #
    #     # Validate the new parameters
    #     self._validate_initial_parameters()
    #
    #     # Initialize relationships dictionary
    #     self.relationships = {}
    def _validate_initial_parameters(self) -> NoReturn:
        """Validates the wellbore initialization parameters for type and value constraints.

        Performs validation checks on max_md_depth, max_tvd_depth, and top of liner (tol)
        parameters to ensure they meet the required criteria for a valid wellbore configuration.

        Args:
            self: Instance of WellBoreExpanded class.

        Raises:
            ValueError: If any of the following conditions are met:
                - max_md_depth is not a number or is not positive
                - max_tvd_depth is not a number or is not positive
                - tol is not a number or is negative

        Notes:
            - max_md_depth must be greater than 0 as a wellbore cannot have zero or negative depth
            - max_tvd_depth must be greater than 0 for valid vertical depth measurements
            - tol (top of liner) must be non-negative as it represents a physical depth

        Example:
            >>> wellbore = WellBoreExpanded(name="WELL-A", top=0, bottom=1000,
            ...                            max_md_depth=-100, max_tvd_depth=900, tol=0)
            ValueError: max_md_depth must be a positive number.
        """
        # Validate measured depth is a positive number
        if not isinstance(self.max_md_depth, (int, float)) or self.max_md_depth <= 0:
            raise ValueError("max_md_depth must be a positive number.")

        # Validate true vertical depth is a positive number
        if not isinstance(self.max_tvd_depth, (int, float)) or self.max_tvd_depth <= 0:
            raise ValueError("max_tvd_depth must be a positive number.")

        # Validate top of liner is non-negative
        if not isinstance(self.tol, (int, float)) or self.tol < 0:
            raise ValueError("tol (top of liner) must be a non-negative number.")
    # def _validate_initial_parameters(self):
    #     """
    #     Validates the newly added initialization parameters.
    #     """
    #     if not isinstance(self.max_md_depth, (int, float)) or self.max_md_depth <= 0:
    #         raise ValueError("max_md_depth must be a positive number.")
    #
    #     if not isinstance(self.max_tvd_depth, (int, float)) or self.max_tvd_depth <= 0:
    #         raise ValueError("max_tvd_depth must be a positive number.")
    #
    #     if not isinstance(self.tol, (int, float)) or self.tol < 0:
    #         raise ValueError("tol (top of liner) must be a non-negative number.")
    def add_section_with_properties(self, **kwargs: Dict[str, Any]) -> NoReturn:
        """Adds a new wellbore section with comprehensive properties and validates required parameters.

        This method extends the base section addition functionality by requiring specific
        wellbore-related properties and ensuring parameter completeness before section creation.

        Args:
            **kwargs: Dictionary containing section properties including:
                id (str): Unique identifier for the section
                tvd (float): True vertical depth
                od (float): Outer diameter
                bottom (float): Bottom depth of section
                casing_type (str): Type of casing material
                weight (float): Weight per unit length
                grade (str): Material grade specification
                connection (str): Connection type
                coeff_friction_sliding (float): Coefficient of friction for sliding
                hole_size (float): Size of the hole
                washout (float): Washout factor
                int_gradient (float): Internal pressure gradient
                mud_weight (float): Drilling mud weight
                backup_mud (float): Backup mud weight
                cement_cu_ft (float): Cement volume in cubic feet
                frac_gradient (float): Fracture gradient
                body_yield (float): Body yield strength
                burst_strength (float): Burst pressure rating
                wall_thickness (float): Wall thickness measurement
                csg_internal_diameter (float): Casing internal diameter
                collapse_pressure (float): Collapse pressure rating
                tension_strength (float): Tension strength rating

        Raises:
            ValueError: If any required parameters are missing from kwargs

        Notes:
            - The method determines section addition strategy based on self.method
            - Sections can be added either top-down or bottom-up
            - All parameters are required and must be provided

        Example:
            >>> wellbore.add_section_with_properties(
            ...     id="SEC1",
            ...     tvd=1000.0,
            ...     od=9.625,
            ...     bottom=1200.0,
            ...     # ... other required parameters ...
            ... )
        """
        # Define required parameters for section creation
        required_params: list[str] = [
            'id', 'tvd', 'od', 'bottom', 'casing_type', 'weight', 'grade',
            'connection', 'coeff_friction_sliding', 'hole_size', 'washout',
            'int_gradient', 'mud_weight', 'backup_mud', 'cement_cu_ft',
            'frac_gradient', 'body_yield', 'burst_strength',
            'wall_thickness', 'csg_internal_diameter', 'collapse_pressure', 'tension_strength'
        ]

        # Validate presence of all required parameters
        missing_params: list[str] = [param for param in required_params if param not in kwargs]
        if missing_params:
            raise ValueError(f"Missing required parameters for section: {missing_params}")

        # Add section based on specified method
        if self.method == "top_down":
            self.add_section_top_down_new(**kwargs)
        elif self.method == "bottom_up":
            self.add_section_bottom_up_new(**kwargs)

    # def add_section_with_properties(self, **kwargs):
    #     """
    #     Adds a new section with additional properties and ensures unique 'id'.
    #
    #     Parameters
    #     ----------
    #     **kwargs :
    #         Arbitrary keyword arguments representing section properties:
    #             - Required: 'id', 'tvd', 'od', 'bottom', 'casing_type', 'weight', 'grade',
    #                         'connection', 'coeff_friction_sliding', 'hole_size', 'washout',
    #                         'int_gradient', 'mud_weight', 'backup_mud', 'cement_cu_ft',
    #                         'frac_gradient', 'body_yield', 'burst_strength',
    #                         'wall_thickness', 'csg_internal_diameter', 'collapse_pressure', 'tension_strength'
    #     """
    #     required_params = [
    #         'id', 'tvd', 'od', 'bottom', 'casing_type', 'weight', 'grade',
    #         'connection', 'coeff_friction_sliding', 'hole_size', 'washout',
    #         'int_gradient', 'mud_weight', 'backup_mud', 'cement_cu_ft',
    #         'frac_gradient', 'body_yield', 'burst_strength',
    #         'wall_thickness', 'csg_internal_diameter', 'collapse_pressure', 'tension_strength'
    #     ]
    #
    #     # Ensure all required parameters are present
    #     missing_params = [param for param in required_params if param not in kwargs]
    #     if missing_params:
    #         raise ValueError(f"Missing required parameters for section: {missing_params}")
    #     if self.method == "top_down":
    #         self.add_section_top_down_new(**kwargs)
    #         # super().add_section(**kwargs)
    #     elif self.method == "bottom_up":
    #         self.add_section_bottom_up_new(**kwargs)

    def add_section_top_down_new(self, **kwargs: Dict[str, Any]) -> NoReturn:
        """Adds a new wellbore section using a top-down approach.

        Builds sections sequentially from top to bottom until the bottommost section's
        bottom depth matches the defined string bottom depth. Maintains ordered section
        indexing and tracks completion status.

        Args:
            **kwargs: Dictionary containing section properties including all required
                parameters as defined in add_section_with_properties()

        Notes:
            - For first section (empty self.sections), starts at wellbore top
            - For subsequent sections, starts at previous section's bottom
            - Automatically sorts sections based on top depth
            - Re-indexes sections to maintain sequential numbering
            - Updates completion status when bottom depth matches target

        Example:
            >>> wellbore.add_section_top_down_new(
            ...     id="SEC1",
            ...     bottom=1000.0,
            ...     tvd=950.0,
            ...     # ... other required parameters ...
            ... )
        """
        # Initialize section position and top depth
        if bool(self.sections) is False:
            temp = 0  # First section index
            top = self.top  # Use wellbore top
        else:
            temp = len(self.sections)  # Next available index
            top = self.sections[temp - 1]['bottom']  # Use previous section bottom

        # Create new section entry and set its top depth
        self.sections[temp] = {}
        self.sections[temp]['top'] = top

        # Add all provided section properties
        for k, v in kwargs.items():
            self.sections[temp][k] = v

        # Sort sections by top depth to maintain proper order
        self.sections = {
            k: v
            for k, v in sorted(
                self.sections.items(), key=lambda item: item[1]['top']
            )
        }

        # Re-index sections to ensure sequential numbering
        temp = {}
        for i, (k, v) in enumerate(self.sections.items()):
            temp[i] = v

        # Check if wellbore is complete (bottom depth matches target)
        if temp[len(temp) - 1]['bottom'] == self.bottom:
            self.complete = True

        # Update sections with re-indexed dictionary
        self.sections = temp
    # def add_section_top_down_new(
    #     self, **kwargs
    # ):
    #     """
    #     Sections built from the top down until the bottom of the bottom section
    #     is equal to the defined string bottom.
    #     """
    #     if bool(self.sections) is False:
    #         temp = 0
    #         top = self.top
    #     else:
    #         temp = len(self.sections)
    #         top = self.sections[temp - 1]['bottom']
    #
    #     self.sections[temp] = {}
    #     self.sections[temp]['top'] = top
    #
    #     # add the section to the sections dict
    #     for k, v in kwargs.items():
    #         self.sections[temp][k] = v
    #
    #     # sort the dict on depth of tops
    #     self.sections = {
    #         k: v
    #         for k, v in sorted(
    #             self.sections.items(), key=lambda item: item[1]['top']
    #         )
    #     }
    #
    #     # re-index the keys
    #     temp = {}
    #     for i, (k, v) in enumerate(self.sections.items()):
    #         temp[i] = v
    #
    #     if temp[len(temp) - 1]['bottom'] == self.bottom:
    #         self.complete = True
    #
    #     self.sections = temp
    def add_section_bottom_up_new(self, **kwargs: Dict[str, Any]) -> NoReturn:
        """Adds a new wellbore section using a bottom-up approach with measured depth support.

        Creates and positions sections starting from the bottom of the wellbore, working upward.
        Supports both measured depth (MD) and true vertical depth (TVD) specifications, with
        section positioning determined by either explicit length or top depth values.

        Args:
            **kwargs: Dictionary containing section properties including:
                length (float, optional): Length of the section
                top (float, optional): Explicit top depth of section
                md_top (float, optional): Measured depth at section top
                md_bottom (float, optional): Measured depth at section bottom
                All other required parameters as defined in add_section_with_properties()

        Notes:
            - For first section (empty self.sections), starts at wellbore bottom
            - For subsequent sections, starts at previous section's md_top or top
            - Section top is determined by:
                1. Length calculation (bottom - length)
                2. Explicit top value if provided
                3. Wellbore top as default
            - Automatically sorts sections based on md_bottom or bottom depth
            - Maintains reverse order sorting for bottom-up configuration
            - Updates completion status when topmost section reaches wellbore top

        Example:
            >>> wellbore.add_section_bottom_up_new(
            ...     length=500.0,
            ...     md_top=500.0,
            ...     md_bottom=1000.0,
            ...     # ... other required parameters ...
            ... )
        """
        # Initialize section position and bottom depth
        if not self.sections:
            temp = 0  # First section index
            bottom = self.bottom  # Use wellbore bottom
        else:
            temp = len(self.sections)  # Next available index
            bottom = self.sections[0].get('md_top', self.sections[0]['top'])  # Use previous section top

        # Determine section top based on available parameters
        if 'length' in kwargs:
            top = bottom - kwargs['length']
            top = max(top, self.top)  # Ensure top doesn't exceed wellbore top
        elif 'top' in kwargs:
            top = kwargs['top']  # Use explicitly provided top
        else:
            top = self.top  # Default to wellbore top

        # Create new section entry with top and bottom depths
        self.sections[temp] = {'top': top, 'bottom': bottom}

        # Add all provided section properties
        for k, v in kwargs.items():
            self.sections[temp][k] = v

        # Sort sections by measured depth or regular depth in reverse order
        self.sections = {
            k: v
            for k, v in sorted(
                self.sections.items(),
                key=lambda item: item[1].get('md_bottom', item[1]['bottom']),
                reverse=True  # Maintain bottom-up order
            )
        }

        # Re-index sections to ensure sequential numbering
        temp = {}
        for i, (k, v) in enumerate(self.sections.items()):
            temp[i] = v

        # Update completion status if top section reaches wellbore top
        if temp[0]['top'] == self.top:
            self.complete = True

        # Update sections with re-indexed dictionary
        self.sections = temp
    # def add_section_bottom_up_new(self, **kwargs):
    #     """
    #     Overridden method for bottom-up sections, supporting md_top and md_bottom.
    #     """
    #     if not self.sections:
    #         temp = 0
    #         bottom = self.bottom
    #     else:
    #         temp = len(self.sections)
    #         bottom = self.sections[0].get('md_top', self.sections[0]['top'])
    #
    #     # Calculate top using length or a provided 'top' value, defaulting if neither is provided
    #     if 'length' in kwargs:
    #         top = bottom - kwargs['length']
    #         top = max(top, self.top)
    #     elif 'top' in kwargs:
    #         top = kwargs['top']
    #     else:
    #         top = self.top
    #
    #     self.sections[temp] = {'top': top, 'bottom': bottom}
    #
    #     # Add provided attributes to the section
    #     for k, v in kwargs.items():
    #         self.sections[temp][k] = v
    #
    #     # Sort sections by md_bottom if available, otherwise by bottom
    #     self.sections = {
    #         k: v
    #         for k, v in sorted(
    #             self.sections.items(), key=lambda item: item[1].get('md_bottom', item[1]['bottom']),
    #             reverse=True  # Sort in reverse for bottom-up alignment
    #         )
    #     }
    #
    #     # Re-index sections
    #     temp = {}
    #     for i, (k, v) in enumerate(self.sections.items()):
    #         temp[i] = v
    #
    #
    #     # Set complete if the top section reaches the well top
    #     if temp[0]['top'] == self.top:
    #         self.complete = True
    #
    #     self.sections = temp
    # def calcParametersContained(self) -> NoReturn:
    #     """Processes and calculates mechanical properties for all wellbore sections.
    #
    #     Performs section-specific calculations and inter-section mechanical analysis:
    #     1. Individual section properties via CasingDataCalc
    #     2. Maximum Allowable Pressure (MAPS) between adjacent sections
    #     3. Burst load calculations and design factors
    #     4. Special handling for single sections
    #
    #     The method updates each section dictionary with calculated results including:
    #     - Section-specific mechanical properties
    #     - Inter-section MAPS values
    #     - Burst loads and design factors
    #     - Solo section calculations when applicable
    #
    #     Args:
    #         None - Uses internal wellbore section data
    #
    #     Updates:
    #         self.sections: Dictionary containing section data, updated with:
    #             maps (float): Maximum allowable pressure
    #             burst_load (float): Calculated burst load
    #             burst_df (float): Burst design factor
    #             Additional properties from CasingDataCalc
    #
    #     Notes:
    #         - Universal parameters (max depths, tolerances) are passed to calculations
    #         - Burst design factor calculations handle zero-load cases
    #         - Different calculation paths for single vs multiple sections
    #         - Sections are processed sequentially from top to bottom
    #
    #     Example:
    #         >>> wellbore.calcParametersContained()
    #         >>> print(wellbore.sections[0]['burst_df'])
    #         1.25
    #     """
    #     # Initialize universal parameters for calculations
    #     secs_num: int = 0
    #     univ_params: Dict[str, float] = {
    #         'max_md_depth': self.max_md_depth,
    #         'max_tvd_depth': self.max_tvd_depth,
    #         'tol': self.tol
    #     }
    #     variables = []
    #     # Process individual section calculations
    #     for i in range(len(self.sections)):
    #         # Calculate section-specific properties
    #         calc_output = CasingDataCalc(self.sections[i], univ_params)
    #         calc_data = calc_output.get_results()
    #         self.sections[i].update(calc_data)
    #         variables.append(self.sections[i]['casing_type'])
    #         secs_num += 1
    #
    #     # Handle multi-section calculations
    #     if secs_num > 1:
    #         # Process inter-section properties for all but last section
    #         for i in range(len(self.sections) - 1):
    #             # Calculate mechanical properties between adjacent sections
    #             maps: float = calculate_maps(self.sections[i], self.sections[i + 1])
    #             burst_load: float = calculate_burst_load(
    #                 self.sections[i],
    #                 self.sections[i + 1],
    #                 maps
    #             )
    #
    #             # Calculate burst design factor with infinity handling
    #             burst_df: float = (
    #                 float(self.sections[i]['burst_strength']) / float(burst_load)
    #                 if float(burst_load) > 0
    #                 else float('inf')
    #             )
    #
    #             # Update section with calculated properties
    #             self.sections[i].update({
    #                 'maps': maps,
    #                 'burst_load': burst_load,
    #                 'burst_df': burst_df
    #             })
    #             counter = i + 1
    #
    #         # Process final section separately
    #         solo_data: Dict[str, float] = calculateSoloMapsBurstLoadDF(
    #             self.sections[counter]
    #         )
    #         self.sections[counter].update(solo_data)
    #
    #     else:
    #         # Handle single section case
    #         solo_data: Dict[str, float] = calculateSoloMapsBurstLoadDF(
    #             self.sections[0]
    #         )
    #         self.sections[0].update(solo_data)
    #     self.create_variables(variables)
    def calcParametersContained(self) -> NoReturn:
        """Processes and calculates mechanical properties for all wellbore sections.

        Performs comprehensive wellbore section analysis including mechanical properties,
        pressure calculations, and design factor evaluations. Handles both single and
        multi-section wellbores with appropriate calculations for each case.

        Key Calculations:
        1. Section-specific mechanical properties (via CasingDataCalc):
           - Body yield strength
           - Burst strength
           - Collapse strength
           - Tension strength
           - Internal diameter calculations
           - Wall thickness validations
        2. Inter-section analysis (for multiple sections):
           - Maximum Allowable Pressure (MAPS)
           - Burst load calculations
           - Design factor computations
        3. Single section analysis:
           - Standalone MAPS calculations
           - Individual burst load analysis
           - Solo design factor computation

        Args:
            None - Uses internal wellbore section data and properties

        Updates:
            self.sections: Dictionary containing section data, updated with:
                maps (float): Maximum allowable pressure between sections
                burst_load (float): Calculated burst load for section
                burst_df (float): Burst design factor (strength/load ratio)
                Additional mechanical properties from CasingDataCalc

        Attributes Used:
            self.max_md_depth (float): Maximum measured depth
            self.max_tvd_depth (float): Maximum true vertical depth
            self.tol (float): Tolerance value for calculations

        Notes:
            - Processes sections sequentially from top to bottom
            - Handles zero-load cases with infinity design factors
            - Creates reference variables for each casing type
            - Updates all sections with calculated mechanical properties

        Example:
            >>> wellbore.calcParametersContained()
            >>> print(wellbore.sections[0]['burst_df'])
            1.25
        """
        # Initialize parameters and tracking variables
        secs_num: int = 0
        univ_params: Dict[str, float] = {
            'max_md_depth': self.max_md_depth,
            'max_tvd_depth': self.max_tvd_depth,
            'tol': self.tol
        }
        variables: List[str] = []  # Track casing types for variable creation

        # Process individual section calculations
        for i in range(len(self.sections)):
            calc_output = CasingDataCalc(self.sections[i], univ_params)
            calc_data = calc_output.get_results()
            self.sections[i].update(calc_data)
            variables.append(self.sections[i]['casing_type'])
            secs_num += 1

        # Handle multi-section calculations
        if secs_num > 1:
            # Process inter-section properties except for last section
            for i in range(len(self.sections) - 1):
                maps: float = calculate_maps(self.sections[i], self.sections[i + 1])
                burst_load: float = calculate_burst_load(
                    self.sections[i],
                    self.sections[i + 1],
                    maps
                )

                # Calculate burst design factor with infinity handling for zero loads
                burst_df: float = (
                    float(self.sections[i]['burst_strength']) / float(burst_load)
                    if float(burst_load) > 0
                    else float('inf')
                )

                # Update section with calculated properties
                self.sections[i].update({
                    'maps': maps,
                    'burst_load': burst_load,
                    'burst_df': burst_df
                })
                counter = i + 1

            # Process final section with solo calculations
            solo_data: Dict[str, float] = calculateSoloMapsBurstLoadDF(
                self.sections[counter]
            )
            self.sections[counter].update(solo_data)

        else:
            # Handle single section calculations
            solo_data: Dict[str, float] = calculateSoloMapsBurstLoadDF(
                self.sections[0]
            )
            self.sections[0].update(solo_data)

        # Create reference variables for all casing types
        self.create_variables(variables)

    def create_variables(self, variables: List[str]) -> NoReturn:
        """Reorganizes section indexing to use casing types as dictionary keys.

        Converts the numerical index-based section dictionary to one that uses
        casing types as keys. This allows for more intuitive access to section
        data using casing type identifiers rather than numerical indices.

        Args:
            variables: List[str]
                List of casing type identifiers corresponding to each section.
                Must match the number of sections in self.sections.

        Updates:
            self.sections: Dict
                Modifies the sections dictionary to use casing types as keys
                instead of numerical indices.

        Notes:
            - Preserves all section data during reorganization
            - Order of sections is maintained through the variables list
            - Original numerical indices are replaced with casing type strings

        Example:
            >>> wellbore.sections = {0: {'casing_type': 'surface', ...},
                                   1: {'casing_type': 'intermediate', ...}}
            >>> variables = ['surface', 'intermediate']
            >>> wellbore.create_variables(variables)
            >>> print(wellbore.sections.keys())
            dict_keys(['surface', 'intermediate'])
        """
        # Reorganize sections using casing types as keys
        for i in range(len(variables)):
            self.sections[variables[i]] = self.sections.pop(i)
    # def create_variables(self, variables):
    #     for i in range(len(variables)):
    #         self.sections[variables[i]] = self.sections.pop(i)

    # def calcParametersContained(self):
    #     """
    #     Processes each section of the wellbore, performs calculations, and appends results to each section.
    #     """
    #     # univ_params = [self.max_md_depth, self.max_tvd_depth, self.tol]
    #     secs_num = 0
    #     univ_params = {'max_md_depth': self.max_md_depth, 'max_tvd_depth': self.max_tvd_depth, 'tol': self.tol}
    #
    #     # Iterate through each section and perform calculations
    #     for i in range(len(self.sections)):
    #         calc_output = CasingDataCalc(self.sections[i], univ_params)
    #         calc_data = calc_output.get_results()
    #         self.sections[i].update(calc_data)
    #         secs_num += 1
    #
    #     # Additional calculations if more than one section exists
    #     if secs_num > 1:
    #         for i in range(len(self.sections) - 1):
    #             maps = calculate_maps(self.sections[i], self.sections[i + 1])
    #             burst_load = calculate_burst_load(self.sections[i], self.sections[i + 1], maps)
    #             burst_df = float(self.sections[i]['burst_strength']) / float(burst_load) if float(burst_load) > 0 else float('inf')
    #             self.sections[i].update({'maps': maps, 'burst_load': burst_load, 'burst_df': burst_df})
    #             counter = i + 1
    #
    #         # Update the last section
    #         solo_data = calculateSoloMapsBurstLoadDF(self.sections[counter])
    #         self.sections[counter].update(solo_data)
    #     else:
    #         # Update the sole section
    #         solo_data = calculateSoloMapsBurstLoadDF(self.sections[0])
    #         self.sections[0].update(solo_data)


class CasingDataCalc:
    """Calculates and stores mechanical and geometric properties for wellbore casing sections.

    This class handles comprehensive calculations for casing sections including pressure
    calculations, mechanical loads, design factors, and geometric properties. It processes
    raw casing data and universal parameters to generate derived properties needed for
    wellbore analysis.

    Attributes:
        tension_df: Optional[float] = None
            Tension design factor
        tension_buoyed: Optional[float] = None
            Buoyed tension force
        tension_air: Optional[float] = None
            Air weight tension force
        neutral_point: Optional[float] = None
            Neutral point depth
        collapse_df: Optional[float] = None
            Collapse design factor
        masp: Optional[float] = None
            Maximum allowable surface pressure
        collapse_load: Optional[float] = None
            Collapse load on casing
        toc: Optional[float] = None
            Top of cement
        cmt_height: Optional[float] = None
            Cement column height
        annular_volume: Optional[float] = None
            Annular volume
        frac_init_pressure: Optional[float] = None
            Initial fracture pressure
        results: Dict[str, Any]
            Calculated results storage

    Args:
        casing: Dict[str, Any]
            Dictionary containing casing properties including:
                frac_gradient: float - Formation fracture gradient
                tvd: float - True vertical depth
                washout: float - Hole washout factor
                int_gradient: float - Internal pressure gradient
                mud_weight: float - Drilling mud weight
                backup_mud: float - Backup mud weight
                cement_cu_ft: float - Cement volume in cubic feet
                hole_size: float - Hole diameter
                bottom: float - Bottom depth
                top: float - Top depth
                weight: float - Casing weight per foot
                od: float - Outer diameter
                grade: str - Casing grade
                connection: str - Connection type
                body_yield: float - Body yield strength
                burst_strength: float - Burst pressure rating
                wall_thickness: float - Wall thickness
                csg_internal_diameter: float - Internal diameter
                collapse_pressure: float - Collapse pressure rating
                tension_strength: float - Tension strength

        univ_params: Dict[str, float]
            Universal parameters including:
                tol: float - Calculation tolerance
                max_md_depth: float - Maximum measured depth
                max_tvd_depth: float - Maximum true vertical depth

    Methods:
        calculateData(): NoReturn
            Performs all casing calculations and stores results

    Notes:
        - All calculations are performed during initialization
        - Results are stored in self.results dictionary
        - None values indicate uncalculated properties
        - All dimensions should be in consistent units
    """

    def __init__(self, casing: Dict[str, Any], univ_params: Dict[str, float]) -> None:
        # Initialize calculation results
        self.tension_df: Optional[float] = None
        self.tension_buoyed: Optional[float] = None
        self.tension_air: Optional[float] = None
        self.neutral_point: Optional[float] = None
        self.collapse_df: Optional[float] = None
        self.masp: Optional[float] = None
        self.collapse_load: Optional[float] = None
        self.toc: Optional[float] = None
        self.cmt_height: Optional[float] = None
        self.annular_volume: Optional[float] = None
        self.frac_init_pressure: Optional[float] = None
        self.results: Dict[str, Any] = {}

        # Store casing properties
        self.frac_gradient: float = casing['frac_gradient']
        self.tvd: float = casing['tvd']
        self.washout: float = casing['washout']
        self.int_gradient: float = casing['int_gradient']
        self.mud_weight: float = casing['mud_weight']
        self.backup_mud: float = casing['backup_mud']
        self.cement_cu_ft: float = casing['cement_cu_ft']
        self.hole_size: float = casing['hole_size']
        self.set_depth: float = casing['bottom']
        self.casing_top: float = casing['top']
        self.csg_weight: float = casing['weight']
        self.csg_size: float = casing['od']
        self.csg_grade: str = casing['grade']

        # Store mechanical properties
        self.csg_collar: str = casing['connection']
        self.body_yield: float = casing['body_yield']
        self.burst_strength: float = casing['burst_strength']
        self.wall_thickness: float = casing['wall_thickness']
        self.csg_internal_diameter: float = casing['csg_internal_diameter']
        self.collapse_strength: float = casing['collapse_pressure']
        self.tension_strength: float = casing['tension_strength']

        # Store universal parameters
        self.tol: float = univ_params['tol']
        self.max_md_depth: float = univ_params['max_md_depth']
        self.max_tvd_depth: float = univ_params['max_tvd_depth']

        # Perform calculations
        self.calculateData()

    def calculateData(self) -> NoReturn:
        """Performs all casing calculations and stores results in the results dictionary.

        Executes a series of calculations to determine casing mechanical properties,
        pressure ratings, and geometric characteristics. All calculations are performed
        in sequence with dependencies handled appropriately.

        Calculations performed:
        1. Formation fracture initial pressure
        2. Annular cement volume
        3. Cement column height
        4. Top of cement (TOC)
        5. Maximum allowable surface pressure (MASP)
        6. Collapse loading conditions
        7. Collapse design factor
        8. Neutral point location
        9. Air weight tension
        10. Buoyed tension
        11. Tension design factor

        Updates:
            All calculated values are stored in both individual attributes
            and consolidated in self.results dictionary.

        Notes:
            - All calculations assume consistent unit systems
            - Results dictionary provides a single access point for all calculations
            - Individual properties remain accessible through class attributes
            - Calculations follow API recommended practices

        Example:
            >>> casing_calc.calculateData()
            >>> print(casing_calc.results['masp'])
            3500.0
        """
        # Calculate formation pressure characteristics
        self.frac_init_pressure = self.frac_gradient * self.tvd

        # Calculate cement properties
        self.annular_volume = self.calculate_cement_volume()
        self.cmt_height = self.calculate_cement_height()
        self.toc = self.calculate_toc()

        # Calculate pressure ratings
        self.masp = self.calculate_masp()
        self.collapse_load = self.calculate_collapse_load()
        self.collapse_df = self.calculate_collapse_df()

        # Calculate mechanical loading conditions
        self.neutral_point = self.calculate_neutral_point()
        self.tension_air = self.calculate_tension_air()
        self.tension_buoyed = self.calculate_tension_buoyed()
        self.tension_df = self.calculate_tension_df()

        # Consolidate results in dictionary
        self.results = {
            'cement_cu_ft': self.cement_cu_ft,  # Cement volume
            'cement_height': self.cmt_height,  # Height of cement column
            'toc': self.toc,  # Top of cement
            'masp': self.masp,  # Maximum allowable surface pressure
            'collapse_strength': self.collapse_strength,  # Rated collapse strength
            'collapse_load': self.collapse_load,  # Applied collapse load
            'collapse_df': self.collapse_df,  # Collapse design factor
            'burst_strength': self.burst_strength,  # Rated burst strength
            'neutral_point': self.neutral_point,  # Neutral point depth
            'tension_strength': self.tension_strength,  # Rated tension strength
            'tension_df': self.tension_df,  # Tension design factor
            'tension_air': self.tension_air,  # Air weight tension
            'tension_buoyed': self.tension_buoyed,  # Buoyed tension
            'frac_init_pressure': self.frac_init_pressure  # Initial fracture pressure
        }

    # def calculateData(self):
    #     self.frac_init_pressure = self.frac_gradient * self.tvd
    #     self.annular_volume = self.calculate_cement_volume()
    #     self.cmt_height = self.calculate_cement_height()
    #     self.toc = self.calculate_toc()
    #     self.masp = self.calculate_masp()
    #     self.collapse_load = self.calculate_collapse_load()
    #     self.collapse_df = self.calculate_collapse_df()
    #     self.neutral_point = self.calculate_neutral_point()
    #     self.tension_air = self.calculate_tension_air()
    #     self.tension_buoyed = self.calculate_tension_buoyed()
    #     self.tension_df = self.calculate_tension_df()
    #     self.results = {'cement_cu_ft': self.cement_cu_ft,
    #                     'cement_height': self.cmt_height,
    #                     'toc': self.toc,
    #                     'masp': self.masp,
    #                     'collapse_strength': self.collapse_strength,
    #                     'collapse_load': self.collapse_load,
    #                     'collapse_df': self.collapse_df,
    #                     'burst_strength': self.burst_strength,
    #                     'neutral_point': self.neutral_point,
    #                     'tension_strength': self.tension_strength,
    #                     'tension_df': self.tension_df,
    #                     'tension_air': self.tension_air,
    #                     'tension_buoyed': self.tension_buoyed,
    #                     'frac_init_pressure': self.frac_init_pressure}
    def get_results(self) -> Dict[str, Any]:
        """Returns the complete set of casing calculation results.

        Provides access to all calculated parameters including mechanical properties,
        pressure ratings, and geometric characteristics of the casing design.

        Returns:
            Dict[str, Any]: A dictionary containing all calculated results including:
                cement_cu_ft (float): Volume of cement in cubic feet
                cement_height (float): Total height of cement column
                toc (float): Top of cement depth
                masp (float): Maximum allowable surface pressure
                collapse_strength (float): Rated collapse strength
                collapse_load (float): Applied collapse load
                collapse_df (float): Collapse design factor
                burst_strength (float): Rated burst strength
                neutral_point (float): Neutral point depth
                tension_strength (float): Rated tension strength
                tension_df (float): Tension design factor
                tension_air (float): Air weight tension
                tension_buoyed (float): Buoyed tension
                frac_init_pressure (float): Initial fracture pressure

        Example:
            >>> results = casing_calc.get_results()
            >>> print(f"MASP: {results['masp']} psi")
            MASP: 3500.0 psi
        """
        return self.results
    # def get_results(self):
    #     """
    #     Returns the calculated results.
    #
    #     Returns
    #     -------
    #     dict
    #         A dictionary of calculated parameters.
    #     """
    #     return self.results
    def calculate_cement_volume(self) -> float:
        """Calculates the annular volume between casing and wellbore per foot.

        Computes the cross-sectional area difference between the open hole and
        the casing outer diameter, accounting for unit conversion from inches
        to feet. The calculation uses the standard geometric formula for
        circular area difference.

        Returns:
            float: Annular volume in cubic feet per linear foot of wellbore.

        Notes:
            - Input dimensions (hole_size, csg_size) are expected in inches
            - Output volume is in cubic feet per foot
            - Does not account for washout or hole irregularities
            - Assumes perfectly concentric casing placement

        Example:
            >>> wellbore.hole_size = 12.25  # inches
            >>> wellbore.csg_size = 9.625   # inches
            >>> volume = wellbore.calculate_cement_volume()
            >>> print(f"Annular volume: {volume:.3f} ft³/ft")
            Annular volume: 0.384 ft³/ft
        """
        # Calculate open hole area in square feet
        hole_area = math.pi * (self.hole_size / 12) ** 2 / 4

        # Calculate casing outer diameter area in square feet
        casing_area = math.pi * (self.csg_size / 12) ** 2 / 4

        # Return annular volume in cubic feet per foot
        return hole_area - casing_area
    # def calculate_cement_volume(self):
    #     # Convert inches to feet and calculate area in square feet
    #     hole_area = math.pi * (self.hole_size / 12) ** 2 / 4
    #     casing_area = math.pi * (self.csg_size / 12) ** 2 / 4
    #     return hole_area - casing_area  # result in cubic feet per foot
    def calculate_cement_height(self) -> float:
        """Calculates the vertical height of cement column in the annular space.

        Computes the height of cement required based on the supplied cement volume,
        taking into account the hole size, casing size, and wellbore washout factor.
        Uses the standard volumetric calculation considering an enlarged wellbore
        due to washout effects.

        The calculation follows these steps:
        1. Adjusts hole size for washout factor
        2. Calculates annular volume per foot
        3. Determines total height needed for given cement volume

        Returns:
            float: Height of cement column in feet

        Notes:
            - Input hole_size and casing_size must be in inches
            - Washout factor is applied as percentage (10 = 10% enlargement)
            - Uses 183.35 as conversion factor for circular geometry
            - Returns 0 if casing size is 0 or negative
            - Does not account for deviation or dogleg severity

        Example:
            >>> wellbore.hole_size = 12.25  # inches
            >>> wellbore.csg_size = 9.625   # inches
            >>> wellbore.washout = 10       # 10% washout
            >>> wellbore.cement_cu_ft = 500  # cubic feet
            >>> height = wellbore.calculate_cement_height()
            >>> print(f"Cement height: {height:.1f} ft")
            Cement height: 1250.5 ft
        """
        if self.csg_size > 0:
            # Calculate effective hole size accounting for washout
            effective_hole_size = self.hole_size * (1 + self.washout / 100)

            # Calculate annular volume per foot using industry standard conversion
            annular_volume_per_foot = (effective_hole_size ** 2 - self.csg_size ** 2) / 183.35

            # Calculate total height needed for given cement volume
            return (1 / annular_volume_per_foot) * self.cement_cu_ft
        else:
            return 0
    # def calculate_cement_height(self):
    #     """
    #     Calculate the cement height.
    #     :param hole_size: Hole size in inches
    #     :param casing_size: Casing size in inches
    #     :param cement_volume: Cement volume in cubic feet
    #     :param washout: Hole washout as a percentage (e.g., 10 for 10%)
    #     :return: Cement height in feet
    #     """
    #     if self.csg_size > 0:
    #         effective_hole_size = self.hole_size * (1 + self.washout / 100)
    #         annular_volume_per_foot = (effective_hole_size ** 2 - self.csg_size ** 2) / 183.35
    #         return (1 / annular_volume_per_foot) * self.cement_cu_ft
    #     else:
    #         return 0
    def calculate_toc(self) -> float:
        """Calculates the Top of Cement (TOC) depth in the wellbore.

        Determines the depth to the top of the cement column by subtracting
        the calculated cement height from the set depth (bottom of cement).
        Includes validation to prevent negative TOC values.

        Returns:
            float: Top of cement depth in feet measured from surface.
                Returns 0 if calculated TOC would be above surface.

        Notes:
            - Assumes vertical wellbore for basic calculations
            - Does not account for wellbore deviation
            - Does not consider cement channeling
            - Assumes complete cement displacement
            - Set depth should be properly initialized before calling

        Example:
            >>> wellbore.set_depth = 5000  # feet
            >>> wellbore.calculate_cement_height()  # returns 2000
            >>> toc = wellbore.calculate_toc()
            >>> print(f"Top of cement: {toc:.1f} ft")
            Top of cement: 3000.0 ft
        """
        # Calculate cement column height
        cement_height = self.calculate_cement_height()

        # Calculate TOC by subtracting cement height from set depth
        output = self.set_depth - cement_height

        # Prevent negative TOC values
        if output < 0:
            return 0

        return output
    # def calculate_toc(self):
    #     """
    #     Calculate the Top of Cement (ToC).
    #
    #     :return: Top of Cement in feet
    #     """
    #     cement_height = self.calculate_cement_height()
    #     output = self.set_depth - cement_height
    #     if output < 0:
    #         return 0
    #     return output
    def calculate_masp(self) -> float:
        """Calculates the Maximum Anticipated Surface Pressure (MASP) for casing design.

        Determines the highest expected surface pressure by comparing pressure
        differentials from both pore pressure and mud hydrostatic conditions.
        The calculation considers internal pressure gradient and prevents
        negative MASP values.

        The calculation follows these steps:
        1. Calculate mud hydrostatic pressure using mud weight and TVD
        2. Calculate internal pressure using internal gradient
        3. Calculate formation pore pressure
        4. Determine maximum differential pressure

        Returns:
            float: Maximum Anticipated Surface Pressure in psi

        Notes:
            - Uses 0.465 psi/ft as default pore pressure gradient
            - Uses 0.05194806 as mud pressure conversion factor
            - Assumes vertical well depth (TVD)
            - Does not account for temperature effects
            - Returns 0 if all differentials are negative

        Example:
            >>> wellbore.tvd = 10000         # feet
            >>> wellbore.mud_weight = 9.5    # ppg
            >>> wellbore.int_gradient = 0.1  # psi/ft
            >>> masp = wellbore.calculate_masp()
            >>> print(f"MASP: {masp:.0f} psi")
            MASP: 3650 psi
        """
        # Default pore pressure gradient - consider making this configurable
        pore_pressure_gradient = 0.465  # psi/ft

        # Calculate pressure components
        mud_hydrostatic_pressure = 0.05194806 * self.tvd * self.mud_weight
        internal_pressure = self.int_gradient * self.tvd
        pore_pressure = pore_pressure_gradient * self.tvd

        # Calculate differential pressures
        masp_from_pore = pore_pressure - internal_pressure
        masp_from_mud = mud_hydrostatic_pressure - internal_pressure

        # Return maximum non-negative pressure
        return max(masp_from_pore, masp_from_mud, 0)

    # def calculate_masp(self):
    #     """
    #     Calculate the Maximum Anticipated Surface Pressure (MASP) considering pore pressure.
    #
    #     Formula: MASP = max(Pore_Pressure - Internal_Pressure, Mud_Hydrostatic_Pressure - Internal_Pressure)
    #
    #     :param pore_pressure_gradient: Pore pressure gradient in psi/ft
    #     :return: MASP in psi
    #     """
    #     # You might want to add pore_pressure_gradient as an attribute of the class or pass it as a parameter
    #     pore_pressure_gradient = 0.465  # Example value, adjust as needed
    #
    #     mud_hydrostatic_pressure = 0.05194806 * self.tvd * self.mud_weight
    #     internal_pressure = self.int_gradient * self.tvd
    #     pore_pressure = pore_pressure_gradient * self.tvd
    #
    #     masp_from_pore = pore_pressure - internal_pressure
    #     masp_from_mud = mud_hydrostatic_pressure - internal_pressure
    #     return max(masp_from_pore, masp_from_mud, 0)
    def calculate_collapse_load(self) -> float:
        """Calculates the differential collapse pressure load at casing bottom.

        Computes the worst-case collapse loading scenario by calculating the
        external pressure from mud weight and assuming an empty casing
        (zero internal pressure). Uses standard pressure gradient calculations
        with industry-standard conversion factors.

        Returns:
            float: Collapse load in psi at the bottom of the casing string

        Notes:
            - Uses 0.052 as pressure conversion factor (psi/ft per ppg)
            - Assumes worst-case empty casing condition
            - Does not account for:
                * Temperature effects on fluid density
                * Cement pressure during setting
                * Formation pressure variations
                * Dynamic well conditions
            - Based on static mud column calculations

        Example:
            >>> wellbore.set_depth = 10000    # feet
            >>> wellbore.mud_weight = 15.0    # ppg
            >>> collapse = wellbore.calculate_collapse_load()
            >>> print(f"Collapse load: {collapse:.0f} psi")
            Collapse load: 7800 psi
        """
        # Calculate external pressure from mud column
        external_pressure = self.set_depth * self.mud_weight * 0.052

        # Assume empty casing for worst-case scenario
        internal_pressure = 0

        # Return differential pressure
        return external_pressure - internal_pressure

    # def calculate_collapse_load(self):
    #     """
    #     Calculate the collapse load at the bottom of the casing.
    #
    #     :return: Collapse load in psi
    #     """
    #     external_pressure = self.set_depth * self.mud_weight * 0.052
    #     internal_pressure = 0  # Assuming empty casing for worst-case scenario
    #     return external_pressure - internal_pressure
    def calculate_collapse_df(self) -> float:
        """Calculates the Collapse Design Factor (DF) for casing design.

        Computes the ratio between the casing's rated collapse strength (from manufacturer data)
        and the calculated collapse load. This factor indicates the safety margin against
        collapse failure under worst-case loading conditions.

        Returns:
            float: Collapse Design Factor (dimensionless)
                - Values > 1 indicate adequate design safety
                - Returns float('inf') if collapse_load is 0
                - Typical minimum acceptable values are 1.125-1.25

        Notes:
            - Collapse strength is obtained from manufacturer specifications
            - Collapse load considers:
                * External mud pressure
                * Empty casing (worst case)
                * Mud weight to PSI conversion (0.052 * depth)
            - Does not include:
                * Temperature effects
                * Corrosion allowance
                * Wear factors

        Example:
            >>> casing.collapse_strength = 4500  # psi
            >>> casing.collapse_load = 3600     # psi
            >>> df = casing.calculate_collapse_df()
            >>> print(f"Collapse DF: {df:.2f}")
            Collapse DF: 1.25
        """
        # Calculate design factor as ratio of strength to load
        return self.collapse_strength / self.collapse_load if self.collapse_load != 0 else float('inf')
    # def calculate_collapse_df(self):
    #     """
    #     Calculate the Collapse Design Factor.
    #
    #     :return: Collapse Design Factor (dimensionless)       """
    #
    #     return self.collapse_strength / self.collapse_load if self.collapse_load != 0 else float('inf')
    def calculate_burst_pressure(self) -> float:
        """Calculates the maximum burst pressure the casing can withstand based on
        material properties and geometry.

        Uses Barlow's formula to determine the internal pressure at which the casing
        will fail under burst conditions. The calculation considers the relationship
        between burst strength, wall thickness, and casing diameter.

        Returns:
            float: Maximum allowable burst pressure in psi

        Notes:
            - Uses Barlow's formula: P = (2 * S * t) / D where:
                * S = Burst strength (psi)
                * t = Wall thickness (inches)
                * D = Outside diameter (inches)
            - Does not account for:
                * Temperature effects on material properties
                * Connection strength derating
                * Material grade variations
                * Corrosion allowance
                * Safety factors

        Example:
            >>> casing.burst_strength = 10000  # psi
            >>> casing.wall_thickness = 0.375  # inches
            >>> casing.csg_size = 7.0         # inches OD
            >>> p_burst = casing.calculate_burst_pressure()
            >>> print(f"Burst Pressure: {p_burst:.0f} psi")
            Burst Pressure: 1071 psi
        """
        # Calculate burst pressure using Barlow's formula
        return (2 * self.burst_strength * self.wall_thickness) / self.csg_size

    # def calculate_burst_pressure(self):
    #     """
    #     Calculate Burst Pressure.
    #
    #     Args:
    #     - self.burst_strength (float): Burst strength in psi.
    #     - self.wall_thickness (float): Wall thickness in inches.
    #     - self.csg_size (float): Outside diameter in inches.
    #
    #     Returns:
    #     - P_burst (float): Burst pressure in psi.
    #     """
    #
    #     return (2 * self.burst_strength * self.wall_thickness) / self.csg_size

    def calculate_neutral_point(self) -> float:
        """Calculates the neutral point depth in the casing string where axial loads transition
        from tension to compression.

        Determines the depth where the buoyed weight of the casing string equals the
        tension load, resulting in a zero net axial force. Uses standard mud weight
        to equivalent density conversion factor of 65.4 ppg.

        Returns:
            float: Neutral point depth in feet measured from surface
                - Above this point, casing is in tension
                - Below this point, casing is in compression

        Notes:
            - Uses 65.4 ppg as fresh water equivalent density
            - Assumes:
                * Vertical wellbore
                * Static conditions
                * Uniform mud weight
                * No external loads
            - Does not account for:
                * Wellbore deviation
                * Temperature effects
                * Dynamic loads
                * External forces

        Example:
            >>> casing.tvd = 10000      # feet
            >>> casing.mud_weight = 9.5  # ppg
            >>> np = casing.calculate_neutral_point()
            >>> print(f"Neutral point: {np:.0f} ft")
            Neutral point: 8547 ft
        """
        # Calculate neutral point using buoyancy effects
        return self.tvd * (1 - self.mud_weight / 65.4)
    # def calculate_neutral_point(self):
    #     return self.tvd * (1 - self.mud_weight / 65.4)
    def calculate_tension_air(self) -> float:
        """Calculates the total tensile load on the casing string in air conditions.

        Computes the axial tension load on the casing string without considering
        buoyancy effects. Accounts for different scenarios based on whether the
        setting depth matches the maximum measured depth.

        Returns:
            float: Tension load in kips (1000 lbs)

        Notes:
            - Calculation factors:
                * Casing weight per foot (lbs/ft)
                * Setting depth (ft)
                * Tolerance adjustment at max depth
            - Different calculations for:
                * Normal setting depth scenarios
                * Maximum measured depth scenarios with tolerance
            - Does not account for:
                * Bending stresses
                * Connection weights
                * Tool weights
                * Dynamic loads

        Example:
            >>> casing.csg_weight = 40      # lbs/ft
            >>> casing.set_depth = 10000    # feet
            >>> tension = casing.calculate_tension_air()
            >>> print(f"Air tension: {tension:.1f} kips")
            Air tension: 400.0 kips
        """
        # Calculate total weight based on setting depth scenario
        if self.set_depth == self.max_md_depth:
            # Adjust for tolerance at maximum depth
            total_weight = self.csg_weight * abs(self.set_depth - self.tol)
        else:
            # Standard calculation for normal setting depths
            total_weight = self.csg_weight * self.set_depth

        # Convert from pounds to kips (1000 lbs)
        return total_weight / 1000

    # def calculate_tension_air(self):
    #     """
    #     Calculate the tension in air.
    #
    #     :param casing_weight: Weight of casing (lbs/ft)
    #     :param setting_depth: Setting depth of the casing (ft)
    #     :return: Tension in air (kips)
    #     """
    #     if self.set_depth == self.max_md_depth:
    #         total_weight = self.csg_weight * abs(self.set_depth - self.tol)
    #     else:
    #         total_weight = self.csg_weight * self.set_depth
    #     return total_weight / 1000  # Convert to kips
    def calculate_tension_buoyed(self) -> float:
        """Calculates the effective tension on the casing string considering buoyancy effects.

        Computes the actual tensile load on the casing while submerged in drilling fluid,
        accounting for buoyancy reduction using Archimedes' principle. Handles different
        calculation scenarios based on setting depth relative to maximum measured depth.

        Returns:
            float: Buoyed tension in kips (1000 lbs)

        Notes:
            - Calculations include:
                * Cross-sectional steel area (pi/4 * (OD² - ID²))
                * Hydrostatic pressure gradient (0.05194806 psi/ft per ppg)
                * Buoyancy force reduction
                * Depth-based weight calculations
            - Different scenarios for:
                * Normal setting depth
                * Maximum measured depth with tolerance adjustment
            - Constants used:
                * 0.05194806: Hydrostatic pressure conversion factor
                * 1000: Conversion from lbs to kips

        Example:
            >>> casing.csg_size = 7.0          # inches OD
            >>> casing.csg_internal_diameter = 6.094  # inches ID
            >>> casing.mud_weight = 15.0       # ppg
            >>> casing.set_depth = 10000       # feet
            >>> tension = casing.calculate_tension_buoyed()
            >>> print(f"Buoyed tension: {tension:.1f} kips")
            Buoyed tension: 285.3 kips
        """
        # Calculate cross-sectional area of steel
        result1 = math.pi / 4 * (self.csg_size ** 2 - self.csg_internal_diameter ** 2)

        # Handle maximum depth scenario with tolerance adjustment
        if self.set_depth == self.max_md_depth:
            # Calculate buoyancy effect at maximum depth
            result2 = 0.05194806 * self.mud_weight * abs(self.tvd - self.max_tvd_depth)
            # Calculate weight with tolerance adjustment
            result3 = self.csg_weight * abs(self.set_depth - self.tol)
        else:
            # Standard calculations for normal depths
            result2 = 0.05194806 * self.mud_weight * self.tvd
            result3 = self.csg_weight * self.set_depth

        # Calculate final buoyed tension and convert to kips
        output = (result3 - result2 * result1) / 1000
        return output

    # def calculate_tension_buoyed(self):
    #     """
    #     Calculate the buoyed tension.
    #
    #     :param casing_weight: Weight of casing (lbs/ft)
    #     :param mud_weight: Mud weight (ppg)
    #     :param setting_depth: Setting depth of the casing (ft)
    #     :return: Buoyed tension (kips)
    #     """
    #
    #     result1 = math.pi / 4 * (self.csg_size ** 2 - self.csg_internal_diameter ** 2)
    #     if self.set_depth == self.max_md_depth:
    #         result2 = 0.05194806 * self.mud_weight * abs(self.tvd - self.max_tvd_depth)
    #         result3 = self.csg_weight * abs(self.set_depth - self.tol)
    #
    #     else:
    #         result2 = 0.05194806 * self.mud_weight * self.tvd
    #         result3 = self.csg_weight * self.set_depth
    #
    #     output = (result3 - result2 * result1) / 1000
    #     return output

    def calculate_tension_df(self) -> float:
        """Calculates the Tension Design Factor (DF) for casing design safety evaluation.

        Computes the ratio between the casing's rated tensile strength and the effective
        buoyed tension load. This factor indicates the safety margin against tensile
        failure under operating conditions.

        Returns:
            float: Tension Design Factor (dimensionless)
                - Values > 1 indicate adequate design safety
                - Returns float('inf') if tension_buoyed is 0
                - Industry standard minimum is typically 1.6-1.8

        Notes:
            - Uses:
                * Tension strength from manufacturer specifications
                * Calculated buoyed tension load
            - Safety considerations:
                * Higher DFs needed for critical wells
                * Additional safety margin for:
                    - Dynamic loads
                    - Shock loads
                    - Temperature effects
                    - Connection efficiency
            - Does not account for:
                * Bending stresses
                * Fatigue effects
                * Corrosion allowance

        Example:
            >>> casing.tension_strength = 800  # kips
            >>> casing.tension_buoyed = 400   # kips
            >>> df = casing.calculate_tension_df()
            >>> print(f"Tension DF: {df:.2f}")
            Tension DF: 2.00
        """
        # Avoid division by zero and return infinity if no buoyed tension
        if self.tension_buoyed == 0:
            return float('inf')

        # Calculate design factor as ratio of strength to load
        return self.tension_strength / self.tension_buoyed
    # def calculate_tension_df(self):
    #     """
    #     Calculate the tension design factor.
    #
    #     :param tension_buoyed: Buoyed tension (kips)
    #     :param tension_strength: Tension strength of the casing (kips)
    #     :return: Tension design factor (dimensionless)
    #     """
    #     if self.tension_buoyed == 0:
    #         return float('inf')  # To avoid division by zero
    #     return self.tension_strength / self.tension_buoyed


pd.set_option('display.max_columns', None)  # Show all columns when displaying DataFrames
pd.options.mode.chained_assignment = None  # Suppress chained assignment warnings


conn = sqlite3.connect('sample_casing.db')
wb_df = pd.read_sql('SELECT * FROM wb_info', conn)
hole_df = pd.read_sql('SELECT * FROM hole_parameters', conn)
casing_df = pd.read_sql('SELECT * FROM casing', conn)
string_df = pd.read_sql('SELECT * FROM string_parameters', conn)

wb_df['casing_depths'] = wb_df['casing_depths'].apply(lambda row: ast.literal_eval(row))


wellbore = WellBoreExpanded(
    name='Wellbore (Planned)', top=wb_df['conductor_casing_bottom'].iloc[0], bottom=wb_df['max_depth_md'].iloc[0], method='top_down', tol=wb_df['top_of_liner'].iloc[0],
    max_md_depth=wb_df['max_depth_md'].iloc[0], max_tvd_depth=wb_df['max_depth_tvd'].iloc[0])
# wellbore = WellBoreExpanded(
#     name='Wellbore (Planned)', top=conductor_casing_bottom, bottom=max_depth_md, method='top_down', tol=top_of_liner,
#     max_md_depth=max_depth_md, max_tvd_depth=max_depth_tvd)

# casing_data = [['Surf', '12.25', '9.625', 2481, '36.0', 'J-55', 'LTC', 'CEMENT', '286', '2.6', '12.0', '166', '1.13', '16.0', '11.0', 'CEMENT'],
# ['I1', '8.75', '7.0', 9346, '29.0', 'P-110', 'BTC', 'CEMENT', '504', '2.1', '12.0', '156', '1.16', '15.8', '11.0', 'CEMENT'],
# ['Prod', '6.125', '5.0', 20512, '18.0', 'P-110', 'BTC', 'CEMENT', '613', '1.38', '14.5', '0', '0.0', '0.0', '15.0', 9246, 'CEMENT']]
# data_labels = ['label', 'hole_size','csg_size', 'set_depth', 'csg_weight', 'csg_grade', 'csg_collar', 'lead_qty', 'lead_yield', 'tail_qty', 'tail_yield']
# casing_data = [['surface', 12.25, 9.625, 2481, 36.0, 'J-55', 'LTC', 286, 2.6, 166, 1.13],
#                 ['intermediate', 8.75, 7.0, 9346, 29.0, 'P-110', 'BTC', 504, 2.1, 156, 1.16],
#                 ['production', 6.125, 5.0, 20512, 18.0, 'P-110', 'BTC', 613, 1.38, 0, 0.0]]

# casing_df = pd.DataFrame(data = casing_data, columns = data_labels)
#
# conn = sqlite3.connect('sample_casing.db')
#
# conn.execute('DROP TABLE IF EXISTS casing')
# casing_df.to_sql('casing', conn, index=False)
# #

# labels = ['label', 'buoyed', 'mw', 'tvd', 'hole_washout', 'internal_gradient', 'backup_mud', 'internal_mud']
# hole_parameters = [['surface', 'y', 11.0, 2481.0, 10, 0.12, 0.0, 0.0],
#                     ['intermediate', 'y', 11.0, 9308.15, 4, 0.22, 0.0, 0.0],
#                     ['production', 'y', 15.0, 10247.499, 4, 0.22, 0.0, 0.0]]
# hole_df = pd.DataFrame(data = hole_parameters, columns = labels)
# conn.execute('DROP TABLE IF EXISTS hole_parameters')
# hole_df.to_sql('hole_parameters', conn, index=False)



# wb_labels = ['conductor_casing_bottom', 'casing_depths', 'top_of_liner', 'max_depth_md', 'max_depth_tvd', 'frac_gradient']
# wb_info = [[100, '[2450, 9350, 9246, 20512]', 9246, 20512, 9208.15, 1]]
# wb_df = pd.DataFrame(data = wb_info, columns = wb_labels)
# conn.execute('DROP TABLE IF EXISTS wb_info')
# wb_df.to_sql('wb_info', conn, index=False)

# df_string_data = [['surface', 9.625, 36.0, 'J-55', '2020', '3520', 'LTC', '453', '564', '0.352', '8.921', '8.765', None],
#  ['intermediate', 7.0, 29.0, 'P-110', '8530', '11220', 'BTC', '955', '929', '0.408', '6.184', '6.059', None],
#  ['production', 5.0, 18.0, 'P-110', '13470', '13620', 'BTC', '606', '580', '0.362', '4.276', '4.151', None]]
# string_labels = ['label', 'CasingDiameter', 'NominalWeight', 'Grade', 'Collapse',
#        'InternalYieldPressure', 'JointType', 'JointStrength', 'BodyYield',
#        'Wall', 'ID', 'DriftDiameterAPI', 'DriftDiameterSD']
# string_labels = [i.lower() for i in string_labels]
# string_df = pd.DataFrame(data = df_string_data, columns = string_labels)
# conn.execute('DROP TABLE IF EXISTS string_parameters')
# string_df.to_sql('string_parameters', conn, index=False)

# print(string_df)
# print(wb_df)
# print(casing_df)
# print(hole_df)
# df = pd.read_sql('SELECT * FROM DX', conn)
# df = df.drop_duplicates()
# conn.execute('DROP TABLE IF EXISTS DX')
# df.to_sql('DX', conn, index=False)
#
# df = pd.read_sql('SELECT * FROM PlatData', conn)
# df = df.drop_duplicates()
# conn.execute('DROP TABLE IF EXISTS PlatData')
# df.to_sql('PlatData', conn, index=False)
print(string_df.columns)
print(casing_df.columns)
print(hole_df.columns)
for idx, row in casing_df.iterrows():
    current_casing = row['label']
    current_hole_parameters = hole_df[hole_df['label']==current_casing]
    current_string_parameters = string_df[string_df['label']==current_casing]
    wellbore.add_section_with_properties(
        id=idx,
        casing_type=current_casing,
        coeff_friction_sliding=0.39,
        frac_gradient=float(wb_df['frac_gradient'].iloc[0]),
        od=float(row['csg_size']),
        bottom=float(row['set_depth']),
        weight=float(row['csg_weight']),
        grade=row['csg_grade'],
        connection=row['csg_collar'],
        hole_size=float(row['hole_size']),
        cement_cu_ft=(float(row['lead_qty']) * float(row['lead_yield'])) + (
                float(row['tail_qty']) * float(row['tail_yield'])),
        tvd=float(current_hole_parameters['tvd'].iloc[0]),
        washout=float(current_hole_parameters['hole_washout'].iloc[0]),
        int_gradient=float(current_hole_parameters['internal_gradient'].iloc[0]),
        mud_weight=float(current_hole_parameters['mw'].iloc[0]),
        backup_mud=float(current_hole_parameters['backup_mud'].iloc[0]),
        body_yield=float(string_df.loc[string_df.index[-1], 'bodyyield']),
        burst_strength=float(string_df.loc[string_df.index[-1], 'internalyieldpressure']),
        wall_thickness=float(string_df.loc[string_df.index[-1], 'wall']),
        csg_internal_diameter=float(string_df.loc[string_df.index[-1], 'id']),
        collapse_pressure=float(string_df.loc[string_df.index[-1], 'collapse']),
        tension_strength=float(string_df.loc[string_df.index[-1], 'jointstrength'])
    )
wellbore.calcParametersContained()
# for i in range(len(wellbore.sections)):
#     print(wellbore.sections[i])
    # data_dict = wellbore.sections[i]
    # print(data_dict)
    # data_dict = {k: v for k, v in data_dict.items() if k not in excluded_keys}
    # data_dict = {key: data_dict[key] for key in desired_order if key in data_dict}
# for i, val in enumerate(self.casing_data):
#     data_table = getattr(self.ui, f"string{i + 1}_table_model_1")
#     side_bar = getattr(self.ui, f"string{i + 1}_parameter_data_model")
#     data_output, side_bar_data = getCasingData(data_table, side_bar)
#     for ids, row in data_output.iterrows():
#         if ids == 0:
#             casing_type = getattr(self.ui, f"string_{i + 1}_label").text().lower()
#         else:
#             casing_type = getattr(self.ui, f"string_{i + 1}_label").text().lower()
#             casing_type = f"{casing_type}_{ids + 1}"
#
#         string_df = self.get_strength_df(self.casing_strength_df, row['Csg Collar'], row['Csg Grade'],
#                                          float(row['Csg Wt.']), float(row['Csg Size']))
#
#         wellbore.add_section_with_properties(
#             id=i,
#             tvd=float(side_bar_data[side_bar_data['1'] == 'TVD']['2'].iloc[0]),
#             od=float(row['Csg Size']),
#             bottom=float(row['Set Depth']),
#             casing_type=casing_type,
#             weight=float(row['Csg Wt.']),
#             grade=row['Csg Grade'],
#             connection=row['Csg Collar'],
#             coeff_friction_sliding=0.39,
#             hole_size=float(row['Hole Size']),
#             washout=float(side_bar_data[side_bar_data['1'] == 'Hole Washout']['2'].iloc[0]),
#             int_gradient=float(side_bar_data[side_bar_data['1'] == 'Internal Grad.']['2'].iloc[0]),
#             mud_weight=float(side_bar_data[side_bar_data['1'] == 'MW']['2'].iloc[0]),
#             backup_mud=float(side_bar_data[side_bar_data['1'] == 'Backup Mud']['2'].iloc[0]),
#             cement_cu_ft=(float(row['Lead Qty']) * float(row['Lead Yield'])) + (
#                     float(row['Tail Qty']) * float(row['Tail Yield'])),
#             frac_gradient=float(self.ui.frac_grad_line.text()),
#             body_yield=float(string_df.loc[string_df.index[-1], 'BodyYield']),
#             burst_strength=float(string_df.loc[string_df.index[-1], 'InternalYieldPressure']),
#             wall_thickness=float(string_df.loc[string_df.index[-1], 'Wall']),
#             csg_internal_diameter=float(string_df.loc[string_df.index[-1], 'I.D.']),
#             collapse_pressure=float(string_df.loc[string_df.index[-1], 'Collapse']),
#             tension_strength=float(string_df.loc[string_df.index[-1], 'JointStrength'])
#         )