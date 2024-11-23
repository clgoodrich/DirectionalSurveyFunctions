import math
import sqlite3
from shapely import wkt
# import ModuleAgnostic as ma
import pandas as pd
import numpy as np
import welleng as we
from welleng.architecture import String
import ast



class WellBoreExpanded(String):
    def __init__(self, name, top, bottom, max_md_depth, max_tvd_depth, tol=0.0, *args, method="bottom_up", **kwargs):
        """
        Extends the `String` class to include additional parameters: max_md_depth, max_tvd_depth, and tol (top of liner).

        Parameters
        ----------
        name : str
            The name of the collection.
        top : float
            The shallowest measured depth at the top of the collection of items in meters.
        bottom : float
            The deepest measured depth at the bottom of the collection of items in meters.
        max_md_depth : float
            The maximum measured depth for the wellbore.
        max_tvd_depth : float
            The maximum true vertical depth for the wellbore.
        tol : float, optional
            Top of liner depth, defaults to 0.0.
        method : str, optional
            Method to add sections ('top_down' or 'bottom_up'), by default "bottom_up".
        """
        super().__init__(name, top, bottom, method=method, *args, **kwargs)
        # Assign additional parameters
        self.max_md_depth = float(max_md_depth)
        self.max_tvd_depth = float(max_tvd_depth)
        self.tol = float(tol)  # Top of liner

        # Validate the new parameters
        self._validate_initial_parameters()

        # Initialize relationships dictionary
        self.relationships = {}

    def _validate_initial_parameters(self):
        """
        Validates the newly added initialization parameters.
        """
        if not isinstance(self.max_md_depth, (int, float)) or self.max_md_depth <= 0:
            raise ValueError("max_md_depth must be a positive number.")

        if not isinstance(self.max_tvd_depth, (int, float)) or self.max_tvd_depth <= 0:
            raise ValueError("max_tvd_depth must be a positive number.")

        if not isinstance(self.tol, (int, float)) or self.tol < 0:
            raise ValueError("tol (top of liner) must be a non-negative number.")

    def add_section_with_properties(self, **kwargs):
        """
        Adds a new section with additional properties and ensures unique 'id'.

        Parameters
        ----------
        **kwargs :
            Arbitrary keyword arguments representing section properties:
                - Required: 'id', 'tvd', 'od', 'bottom', 'casing_type', 'weight', 'grade',
                            'connection', 'coeff_friction_sliding', 'hole_size', 'washout',
                            'int_gradient', 'mud_weight', 'backup_mud', 'cement_cu_ft',
                            'frac_gradient', 'body_yield', 'burst_strength',
                            'wall_thickness', 'csg_internal_diameter', 'collapse_pressure', 'tension_strength'
        """
        required_params = [
            'id', 'tvd', 'od', 'bottom', 'casing_type', 'weight', 'grade',
            'connection', 'coeff_friction_sliding', 'hole_size', 'washout',
            'int_gradient', 'mud_weight', 'backup_mud', 'cement_cu_ft',
            'frac_gradient', 'body_yield', 'burst_strength',
            'wall_thickness', 'csg_internal_diameter', 'collapse_pressure', 'tension_strength'
        ]

        # Ensure all required parameters are present
        missing_params = [param for param in required_params if param not in kwargs]
        if missing_params:
            raise ValueError(f"Missing required parameters for section: {missing_params}")
        if self.method == "top_down":
            self.add_section_top_down_new(**kwargs)
            # super().add_section(**kwargs)
        elif self.method == "bottom_up":
            self.add_section_bottom_up_new(**kwargs)


    def add_section_top_down_new(
        self, **kwargs
    ):
        """
        Sections built from the top down until the bottom of the bottom section
        is equal to the defined string bottom.
        """
        if bool(self.sections) is False:
            temp = 0
            top = self.top
        else:
            temp = len(self.sections)
            top = self.sections[temp - 1]['bottom']

        self.sections[temp] = {}
        self.sections[temp]['top'] = top

        # add the section to the sections dict
        for k, v in kwargs.items():
            self.sections[temp][k] = v

        # sort the dict on depth of tops
        self.sections = {
            k: v
            for k, v in sorted(
                self.sections.items(), key=lambda item: item[1]['top']
            )
        }

        # re-index the keys
        temp = {}
        for i, (k, v) in enumerate(self.sections.items()):
            temp[i] = v

        # # check inputs
        # for k, v in temp.items():
        #     if k == 0:
        #         assert v['top'] == self.top
        #     else:
        #         assert v['top'] == temp[k - 1]['bottom']
        #     assert v['bottom'] <= self.bottom

        if temp[len(temp) - 1]['bottom'] == self.bottom:
            self.complete = True

        self.sections = temp

    def add_section_bottom_up_new(self, **kwargs):
        """
        Overridden method for bottom-up sections, supporting md_top and md_bottom.
        """
        if not self.sections:
            temp = 0
            bottom = self.bottom
        else:
            temp = len(self.sections)
            bottom = self.sections[0].get('md_top', self.sections[0]['top'])

        # Calculate top using length or a provided 'top' value, defaulting if neither is provided
        if 'length' in kwargs:
            top = bottom - kwargs['length']
            top = max(top, self.top)
        elif 'top' in kwargs:
            top = kwargs['top']
        else:
            top = self.top

        self.sections[temp] = {'top': top, 'bottom': bottom}

        # Add provided attributes to the section
        for k, v in kwargs.items():
            self.sections[temp][k] = v

        # Sort sections by md_bottom if available, otherwise by bottom
        self.sections = {
            k: v
            for k, v in sorted(
                self.sections.items(), key=lambda item: item[1].get('md_bottom', item[1]['bottom']),
                reverse=True  # Sort in reverse for bottom-up alignment
            )
        }

        # Re-index sections
        temp = {}
        for i, (k, v) in enumerate(self.sections.items()):
            temp[i] = v

        # Check section alignment for bottom-up order
        number_of_sections = len(temp)
        # for k, v in temp.items():
        #     if k == number_of_sections - 1:
        #         assert v['bottom'] == self.bottom
        #     else:
        #         assert v['bottom'] == temp[k + 1]['top']
        #     assert v['top'] >= self.top

        # Set complete if the top section reaches the well top
        if temp[0]['top'] == self.top:
            self.complete = True

        self.sections = temp

    # def calcParametersContained(self):
    #     secs_num = 0
    #     """
    #     Processes each section of the wellbore and performs calculations.
    #     """
    #     # Universal parameters (if needed)
    #     univ_params = {'max_md_depth': self.max_md_depth, 'max_tvd_depth': self.max_tvd_depth, 'tol': self.tol}
    #     # Iterate through each section and perform calculations
    #     for i in range(len(self.sections)):
    #         calc_output = CasingDataCalc(self.sections[i], univ_params)
    #         calc_data = calc_output.get_results()
    #         self.sections[i].update(calc_data)
    #         secs_num += 1
    #     if secs_num > 1:
    #         for i in range(len(self.sections) - 1):
    #             maps = self.calculate_maps(self.sections[i], self.sections[i + 1])
    #             burst_load = self.calculate_burst_load(self.sections[i], self.sections[i + 1], maps)
    #             burst_df = float(self.sections[i]['burst_strength']) / float(burst_load) if float(burst_load) > 0 else float('inf')
    #             self.sections[i].update({'maps':maps, 'burst_load': burst_load, 'burst_df':burst_df})
    #             counter = i + 1
    #         self.sections[counter] = self.calculateSoloMapsBurstLoadDF(self.sections[counter])
    #     else:
    #         self.sections[0] = self.calculateSoloMapsBurstLoadDF(self.sections[0])
    #     # new_wellbore = WellBoreExpanded(
    #     #     name='Wellbore (Planned)', top=self.top, bottom=self.bottom, method='top_down', tol=self.tol, max_md_depth=self.max_md_depth, max_tvd_depth=self.max_tvd_depth)
    #     return self.sections
    def calcParametersContained(self):
        """
        Processes each section of the wellbore, performs calculations, and appends results to each section.
        """
        # univ_params = [self.max_md_depth, self.max_tvd_depth, self.tol]
        secs_num = 0
        univ_params = {'max_md_depth': self.max_md_depth, 'max_tvd_depth': self.max_tvd_depth, 'tol': self.tol}

        # Iterate through each section and perform calculations
        for i in range(len(self.sections)):
            calc_output = CasingDataCalc(self.sections[i], univ_params)
            calc_data = calc_output.get_results()
            self.sections[i].update(calc_data)
            secs_num += 1

        # Additional calculations if more than one section exists
        if secs_num > 1:
            for i in range(len(self.sections) - 1):
                maps = self.calculate_maps(self.sections[i], self.sections[i + 1])
                burst_load = self.calculate_burst_load(self.sections[i], self.sections[i + 1], maps)
                burst_df = float(self.sections[i]['burst_strength']) / float(burst_load) if float(burst_load) > 0 else float('inf')
                self.sections[i].update({'maps': maps, 'burst_load': burst_load, 'burst_df': burst_df})
                counter = i + 1

            # Update the last section
            solo_data = self.calculateSoloMapsBurstLoadDF(self.sections[counter])
            self.sections[counter].update(solo_data)
        else:
            # Update the sole section
            solo_data = self.calculateSoloMapsBurstLoadDF(self.sections[0])
            self.sections[0].update(solo_data)

    # def calcParametersContained(self, wellbore):
    #     for i, val in enumerate(wellbore.sections):
    #         dothething()
    #         pass
    # self.frac_gradient = casing['frac_gradient']
    # self.tvd = casing['tvd']
    # self.washout = casing['washout']
    # self.int_gradient = casing['int_gradient']
    # self.mud_weight = casing['mud_weight']
    # self.backup_mud = casing['backup_mud']
    # self.cement_cu_ft = casing['cement_cu_ft']
    # self.hole_size = casing['hole_size']
    # self.set_depth = casing['bottom']
    # self.casing_top = casing['top']
    # self.csg_weight = casing['weight']
    # self.csg_size = casing['od']
    # self.csg_grade = casing['grade']
    # self.csg_collar = casing['connection']
    # self.tol = casing['tol']
    # self.max_depth = casing['max_depth']
    # self.tvd_tol = casing['tvd_tol']
    #
    # self.specified_minimum_yield_strength = (self.burst_strength * self.wall_thickness) / (2 * self.csg_size)
    # self.annular_volume = self.calculate_cement_volume()
    # self.cmt_height = self.calculate_cement_height()
    # self.toc = self.calculate_toc()
    # self.masp = self.calculate_masp()
    # self.collapse_strength = self.calculate_collapse_strength()
    # self.collapse_load = self.calculate_collapse_load()
    # self.collapse_df = self.calculate_collapse_df()
    # self.tension_strength = self.calculate_tension_strength()
    # self.neutral_point = self.calculate_neutral_point()
    # self.tension_air = self.calculate_tension_air()
    # self.tension_buoyed = self.calculate_tension_buoyed()
    # self.tension_df = self.calculate_tension_df(self.tension_strength, self.tension_buoyed)
    # pass
    def calculateSoloMapsBurstLoadDF(self, section):
        maps = section['mud_weight'] * section['tvd'] * 0.05194806
        burst_load = max((0.05194806 * (section['mud_weight'] - section['backup_mud']) * section['tvd']), min((section['frac_init_pressure'] - (0.05194806 * section['backup_mud'] * section['tvd'])),maps - section['int_gradient'] * (section['tvd'] - section['tvd']) - (0.05194806 * section['backup_mud'] * section['tvd'])))
        burst_df = float(section['burst_strength']) / float(burst_load) if float(burst_load) > 0 else float('inf')
        section.update({'maps': maps, 'burst_load': burst_load, 'burst_df': burst_df})

        return section

    def calculate_maps(self, sec1, sec2):
        next_bhp = sec2['mud_weight'] * sec2['tvd'] * 0.05194806
        maps = next_bhp - (sec2['tvd'] - sec1['tvd']) * sec2['int_gradient']
        return maps

    def calculate_burst_load(self, sec1, sec2, maps):
        part1 = 0.05194806 * (sec1['mud_weight'] - sec1['backup_mud']) * sec1['tvd']
        minPart1 = sec1['frac_init_pressure'] - (sec1['tvd'] - sec1['tvd']) * sec2['int_gradient'] - (0.05194806 * sec1['backup_mud'] * sec1['tvd'])
        minPart2 = maps - sec1['int_gradient'] * (sec1['tvd'] - sec1['tvd']) - (0.05194806 * sec1['backup_mud'] * sec1['tvd'])
        max_all = max(part1, min(minPart1, minPart2))
        return max_all

        # part1 = calcBurstLoadPart1(backup_mud, mud_weight, tvd_depth)

        # frac_pressure = fractureInitiationPressure(frac_gradient, tvd_depth)

    #     part1 = calcBurstLoadPart1(backup_mud, mud_weight, tvd_depth)
    #     part2 = calcBurstLoadPart2(frac_pressure, tvd_depth, next_internal_gradient, backup_mud, shoe_press, internal_gradient)
    #     max_all = max(part1, part2)
    #     return max_all

    def add_relationship(self, section_id_1, relation, section_id_2):
        """
        Defines a relationship between two sections.

        Parameters
        ----------
        section_id_1 : int or str
            ID of the first section.
        relation : str
            Type of relationship (e.g., 'supports', 'is_supported_by').
        section_id_2 : int or str
            ID of the second section.
        """
        if section_id_1 not in self.get_all_section_ids():
            raise ValueError(f"Section ID {section_id_1} does not exist.")
        if section_id_2 not in self.get_all_section_ids():
            raise ValueError(f"Section ID {section_id_2} does not exist.")

        if section_id_1 not in self.relationships:
            self.relationships[section_id_1] = {}
        self.relationships[section_id_1][relation] = section_id_2

    def get_all_section_ids(self):
        """
        Retrieves all section IDs in the string.

        Returns
        -------
        set
            A set of all section IDs.
        """
        return {props['id'] for props in self.sections.values()}

    def get_section_by_id(self, section_id):
        """
        Retrieves a section's properties by its ID.

        Parameters
        ----------
        section_id : int or str
            The unique identifier of the section.

        Returns
        -------
        dict
            The properties of the section.
        """
        for props in self.sections.values():
            if props.get('id') == section_id:
                return props
        raise ValueError(f"Section with ID {section_id} not found.")

    def get_relationships(self, section_id):
        """
        Retrieves all relationships for a given section.

        Parameters
        ----------
        section_id : int or str
            The unique identifier of the section.

        Returns
        -------
        dict
            A dictionary of relationships.
        """
        return self.relationships.get(section_id, {})

    def __repr__(self):
        return f"ExtendedString(name={self.name}, sections={self.sections}, relationships={self.relationships})"


class CasingDataCalc:
    def __init__(self, casing, univ_params):
        self.tension_df = None
        self.tension_buoyed = None
        self.tension_air = None
        self.neutral_point = None
        self.collapse_df = None
        self.masp = None
        self.collapse_load = None
        self.toc = None
        self.cmt_height = None
        self.annular_volume = None
        self.frac_init_pressure = None
        self.results = {}
        self.frac_gradient = casing['frac_gradient']
        self.tvd = casing['tvd']
        self.washout = casing['washout']
        self.int_gradient = casing['int_gradient']
        self.mud_weight = casing['mud_weight']
        self.backup_mud = casing['backup_mud']
        self.cement_cu_ft = casing['cement_cu_ft']
        self.hole_size = casing['hole_size']
        self.set_depth = casing['bottom']
        self.casing_top = casing['top']
        self.csg_weight = casing['weight']
        self.csg_size = casing['od']
        self.csg_grade = casing['grade']
        self.csg_collar = casing['connection']
        self.body_yield = casing['body_yield']
        self.burst_strength = casing['burst_strength']
        self.wall_thickness = casing['wall_thickness']
        self.csg_internal_diameter = casing['csg_internal_diameter']
        self.collapse_strength = casing['collapse_pressure']
        self.tension_strength = casing['tension_strength']
        self.tol = univ_params['tol']
        self.max_md_depth = univ_params['max_md_depth']
        self.max_tvd_depth = univ_params['max_tvd_depth']
        self.calculateData()

    def calculateData(self):
        self.frac_init_pressure = self.frac_gradient * self.tvd
        self.annular_volume = self.calculate_cement_volume()
        self.cmt_height = self.calculate_cement_height()
        self.toc = self.calculate_toc()
        self.masp = self.calculate_masp()
        self.collapse_load = self.calculate_collapse_load()
        self.collapse_df = self.calculate_collapse_df()
        self.neutral_point = self.calculate_neutral_point()
        self.tension_air = self.calculate_tension_air()
        self.tension_buoyed = self.calculate_tension_buoyed()
        self.tension_df = self.calculate_tension_df()
        self.results = {'cement_cu_ft': self.cement_cu_ft,
                        'cement_height': self.cmt_height,
                        'toc': self.toc,
                        'masp': self.masp,
                        'collapse_strength': self.collapse_strength,
                        'collapse_load': self.collapse_load,
                        'collapse_df': self.collapse_df,
                        'burst_strength': self.burst_strength,
                        'neutral_point': self.neutral_point,
                        'tension_strength': self.tension_strength,
                        'tension_df': self.tension_df,
                        'tension_air': self.tension_air,
                        'tension_buoyed': self.tension_buoyed,
                        'frac_init_pressure': self.frac_init_pressure}

    def get_results(self):
        """
        Returns the calculated results.

        Returns
        -------
        dict
            A dictionary of calculated parameters.
        """
        return self.results
        # labels = ['cement_cu_ft',
        #           'cement_height',
        #           'toc',
        #           'masp',
        #           'collapse_strength',
        #           'collapse_load',
        #           'collapse_df',
        #           'burst_strength',
        #           'neutral_point',
        #           'tension_strength',
        #           'tension_df',
        #           'tension_air',
        #           'tension_buoyed']
        # values = [self.cement_cu_ft, self.cmt_height, self.toc, self.masp, self.collapse_strength, self.collapse_load, self.collapse_df, self.burst_strength, self.neutral_point, self.tension_strength, self.tension_df, self.tension_air, self.tension_buoyed]
        # result_dict = dict(zip(labels, values))
        # sr_val = pd.Series(data=values, index=labels)
        #
        #
        # self.frac_init_pressure = self.frac_gradient * self.tvd
        # self.strength_queried_df = self.get_strength_df(casing_strength_df)

    # class Casing:
    #     def __init__(self, casing, casing_strength_df):
    #         self.frac_gradient = casing['frac_gradient']
    #         self.tvd = casing['tvd']
    #         self.washout = casing['washout']
    #         self.int_gradient = casing['int_gradient']
    #         self.mud_weight = casing['mud_weight']
    #         self.backup_mud = casing['backup_mud']
    #         self.cement_cu_ft = casing['cement_cu_ft']
    #         self.hole_size = casing['hole_size']
    #         self.set_depth = casing['bottom']
    #         self.casing_top = casing['top']
    #         self.csg_weight = casing['weight']
    #         self.csg_size = casing['od']
    #         self.csg_grade = casing['grade']
    #         self.csg_collar = casing['connection']
    #         self.tol = casing['tol']
    #         self.max_depth = casing['max_depth']
    #         self.tvd_tol = casing['tvd_tol']
    #         self.frac_init_pressure = self.frac_gradient * self.tvd
    #         self.strength_queried_df = self.get_strength_df(casing_strength_df)
    #         self.body_yield = float(self.strength_queried_df.loc[self.strength_queried_df.index[-1], 'BodyYield'])
    #         self.burst_strength = float(self.strength_queried_df.loc[self.strength_queried_df.index[-1], 'InternalYieldPressure']) #minimum_internal_yield_pressure
    #         self.wall_thickness = float(self.strength_queried_df.loc[self.strength_queried_df.index[-1], 'Wall'])  # t in inches
    #         self.specified_burst_strength = float(self.strength_queried_df.loc[self.strength_queried_df.index[-1], 'JointStrength'])
    #         self.specified_minimum_yield_strength = (self.burst_strength * self.wall_thickness) / (2 * self.csg_size)
    #         self.csg_internal_diameter = float(self.strength_queried_df.loc[self.strength_queried_df.index[-1], 'I.D.'])
    #         self.collapse_pressure = float(self.strength_queried_df.loc[self.strength_queried_df.index[-1], 'Collapse'])
    #         self.annular_volume = self.calculate_cement_volume()
    #         self.cmt_height = self.calculate_cement_height()
    #         self.toc = self.calculate_toc()
    #         self.masp = self.calculate_masp()
    #
    #         self.collapse_strength = self.calculate_collapse_strength()
    #         self.collapse_load = self.calculate_collapse_load()
    #         self.collapse_df = self.calculate_collapse_df()
    #         # self.burst_strength = self.calculate_burst_strength()
    #         self.maps = self.calculate_maps()
    #         self.burst_load = self.calculate_burst_load()
    #         self.burst_load = self.burst_strength / 1.33
    #         # Ps = self.burst_strength / 1.33 + (OPG * self.tvd) - (self.mud_weight * self.tvd * 0.052)
    #         # output = "{} + {}{}".format((self.burst_strength / 1.33)-(self.mud_weight * self.tvd * 0.052), self.tvd,'OPG')
    #         # output = r"{} + {}/{}".format((self.mud_weight * 0.052) - (self.burst_strength/self.tvd), "Ps",self.tvd)

    #
    #         # OPG = (self.mud_weight * 0.052 + Ps/self.tvd - self.burst_strength/self.tvd)

    #         # self.burst_pressure = self.calculate_burst_pressure()
    #         # # self.collapse_pressure = self.calculate_collapse_pressure()
    #         # MAP = min(self.burst_pressure, self.collapse_pressure, self.yield_strength)


    #         self.burst_df = self.calculate_burst_df()
    #         self.tension_strength = self.calculate_tension_strength()
    #         self.neutral_point = self.calculate_neutral_point()
    #         self.tension_air = self.calculate_tension_air()
    #         self.tension_buoyed = self.calculate_tension_buoyed()
    #         self.tension_df = self.calculate_tension_df()

    #         labels = ['cement_cu_ft',
    #                   'cement_height',
    #                   'toc',
    #                   'masp',
    #                   'collapse_strength',
    #                   'collapse_load',
    #                   'collapse_df',
    #                   'burst_strength',
    #                   'burst_load',
    #                   'burst_df',
    #                   'tension_strength',
    #                   'tension_df',
    #                   'neutral_point',
    #                   'tension_air',
    #                   'tension_buoyed']
    #         values = [self.cement_cu_ft, self.cmt_height, self.toc, self.masp, self.collapse_strength, self.collapse_load, self.collapse_df, self.burst_strength, self.burst_load,self.burst_df,  self.tension_strength, self.tension_df, self.neutral_point, self.tension_air, self.tension_buoyed, ]
    #         # df_val = pd.DataFrame(data = values, columns=labels)
    #         sr_val = pd.Series(data=values, index=labels)

    # def __init__2(self, side_bar, data_table, data_model, casing_strength_df, frac_gradient):
    #     data_output = ma.get_dataframe_from_qtableview(data_table)
    #     side_bar_data = ma.get_dataframe_from_qtableview(side_bar)
    #     self.frac_gradient = frac_gradient
    #     self.tvd = float(side_bar_data[side_bar_data['1'] == 'TVD']['2'].iloc[0])
    #     self.washout = float(side_bar_data[side_bar_data['1'] == 'Hole Washout']['2'].iloc[0])
    #     self.int_gradient = float(side_bar_data[side_bar_data['1'] == 'Internal Grad.']['2'].iloc[0])
    #     self.mud_weight = float(side_bar_data[side_bar_data['1'] == 'MW']['2'].iloc[0])
    #     self.backup_mud = float(side_bar_data[side_bar_data['1'] == 'Backup Mud']['2'].iloc[0])
    #     self.cement_cu_ft = (float(data_output['Lead Qty']) * float(data_output['Lead Yield'])) + (float(data_output['Tail Qty']) * float(data_output['Tail Yield']))
    #     self.hole_size = float(data_output.loc[data_output.index[-1], 'Hole Size'])
    #     self.set_depth = float(data_output.loc[data_output.index[-1], 'Set Depth'])
    #     self.csg_weight = float(data_output.loc[data_output.index[-1], 'Csg Wt.'])
    #     self.csg_size = float(data_output.loc[data_output.index[-1], 'Csg Size'])
    #     self.csg_grade = data_output.loc[data_output.index[-1], 'Csg Grade']
    #     self.csg_collar = data_output.loc[data_output.index[-1], 'Csg Collar']
    #
    #     self.strength_queried_df = self.get_strength_df(casing_strength_df)
    #
    #
    #     self.csg_internal_diameter = float(self.strength_queried_df.loc[self.strength_queried_df.index[-1], 'I.D.'])
    #     self.annular_volume = self.calculate_cement_volume()
    #     self.cmt_height = self.calculate_cement_height()
    #     self.toc = self.calculate_toc()
    #     self.masp = self.calculate_masp()
    #     self.maps = self.calculate_maps()
    #     self.collapse_strength = self.calculate_collapse_strength()
    #     self.collapse_load = self.calculate_collapse_load()
    #     self.collapse_df = self.calculate_collapse_df()
    #
    #     self.burst_strength = self.calculate_burst_strength()
    #     self.burst_load = self.calculate_burst_load()
    #     self.calcBurstLoad(self.frac_gradient, self.tvd, self.maps, self.mud_weight, self.int_gradient, 0.22, self.backup_mud)
    #
    #     self.tension_strength = self.calculate_tension_strength()
    #     self.neutral_point = self.calculate_neutral_point()
    #     self.tension_air = self.calculate_tension_air()
    #
    #     self.tension_buoyed = self.calculate_tension_buoyed()
    #     self.tension_df = self.calculate_tension_df(tension_strength, tension_buoyed)

    def get_strength_df(self, casing_strength_df):
        possible_joins = ['STC', 'LTC', 'BTC', 'BTB', 'None', 'DQX', 'NT']
        possible_joins = [self.csg_collar] + [join for join in possible_joins if join != self.csg_collar]

        output = casing_strength_df[
            (casing_strength_df['Grade'] == self.csg_grade) &
            (casing_strength_df['JointType'].isin(possible_joins)) &
            (casing_strength_df['NominalWeight'] == float(self.csg_weight)) &
            (casing_strength_df['CasingDiameter'] == float(self.csg_size))]

        output_sorted = output.sort_values('JointType', key=lambda x: x.map({v: i for i, v in enumerate(possible_joins)})).head(1)

        output_sorted = output_sorted.rename(columns={'Grade': 'CsgGrade', 'JointType': 'CsgCollar', 'NominalWeight': 'CsgWt', 'CasingDiameter': 'CsgSize'})

        return output_sorted

    def get_strength_parameters(self, used_df):
        return {
            "OD": used_df['CasingDiameter'].values[0],
            "Nominal Weight": used_df['NominalWeight'].values[0],
            "Grade": used_df['Grade'].values[0],
            "Collapse": used_df['Collapse'].values[0],
            "Internal Yield Pressure": used_df['InternalYieldPressure'].values[0],
            "Joint Type": used_df['JointType'].values[0],
            "Joint Strength": used_df['JointStrength'].values[0],
            "Body Yield": used_df['BodyYield'].values[0],
            "Wall": used_df['Wall'].values[0],
            "ID": used_df['I.D.'].values[0],
            "Drift Diameter": used_df['DriftDiameterAPI'].values[0]
        }

    def calculate_cement_volume(self):
        # Convert inches to feet and calculate area in square feet
        hole_area = math.pi * (self.hole_size / 12) ** 2 / 4
        casing_area = math.pi * (self.csg_size / 12) ** 2 / 4
        return hole_area - casing_area  # result in cubic feet per foot

    def calculate_cement_height(self):
        """
        Calculate the cement height.
        :param hole_size: Hole size in inches
        :param casing_size: Casing size in inches
        :param cement_volume: Cement volume in cubic feet
        :param washout: Hole washout as a percentage (e.g., 10 for 10%)
        :return: Cement height in feet
        """
        if self.csg_size > 0:
            effective_hole_size = self.hole_size * (1 + self.washout / 100)
            annular_volume_per_foot = (effective_hole_size ** 2 - self.csg_size ** 2) / 183.35
            return (1 / annular_volume_per_foot) * self.cement_cu_ft
        else:
            return 0

    def calculate_toc(self):
        """
        Calculate the Top of Cement (ToC).

        :return: Top of Cement in feet
        """
        cement_height = self.calculate_cement_height()
        output = self.set_depth - cement_height
        if output < 0:
            return 0
        return output

    def calculate_masp(self):
        """
        Calculate the Maximum Anticipated Surface Pressure (MASP) considering pore pressure.

        Formula: MASP = max(Pore_Pressure - Internal_Pressure, Mud_Hydrostatic_Pressure - Internal_Pressure)

        :param pore_pressure_gradient: Pore pressure gradient in psi/ft
        :return: MASP in psi
        """
        # You might want to add pore_pressure_gradient as an attribute of the class or pass it as a parameter
        pore_pressure_gradient = 0.465  # Example value, adjust as needed

        mud_hydrostatic_pressure = 0.05194806 * self.tvd * self.mud_weight
        internal_pressure = self.int_gradient * self.tvd
        pore_pressure = pore_pressure_gradient * self.tvd

        masp_from_pore = pore_pressure - internal_pressure
        masp_from_mud = mud_hydrostatic_pressure - internal_pressure
        return max(masp_from_pore, masp_from_mud, 0)

    def calculate_maps(self):
        """
        Calculate the Maximum Anticipated Pressure at the Shoe (MAPS).

        :return: MAPS in psi
        """
        # Calculate MASP
        masp = self.calculate_masp()

        # Calculate hydrostatic pressure of the mud at the shoe
        hydrostatic_pressure_shoe = 0.05194806 * self.mud_weight * self.set_depth
        # Calculate the fracture pressure at the shoe
        frac_pressure_shoe = self.frac_gradient * self.set_depth

        # Calculate MAPS
        maps = min(masp - (self.tvd - self.set_depth) * self.int_gradient,
                   frac_pressure_shoe - hydrostatic_pressure_shoe)

        return max(maps, 0)  # Ensure MAPS is not negative

    def calculate_collapse_strength(self):
        """
        Get the collapse strength from the queried strength dataframe.

        :return: Collapse strength in psi
        """
        return float(self.strength_queried_df['Collapse'].iloc[0])

    def calculate_collapse_load(self):
        """
        Calculate the collapse load at the bottom of the casing.

        :return: Collapse load in psi
        """
        external_pressure = self.set_depth * self.mud_weight * 0.052
        internal_pressure = 0  # Assuming empty casing for worst-case scenario
        return external_pressure - internal_pressure

    def calculate_collapse_df(self):
        """
        Calculate the Collapse Design Factor.

        :return: Collapse Design Factor (dimensionless)       """

        return self.collapse_strength / self.collapse_load if self.collapse_load != 0 else float('inf')

    def calculate_burst_strength(self):
        """
        Get the internal yield pressure (burst strength) from the queried strength dataframe.

        :return: Internal yield pressure in psi
        """
        return float(self.strength_queried_df['InternalYieldPressure'].iloc[0])

    def calculate_burst_load(self):
        """
        Calculate the burst load.

        Formula: Burst Load = MASP + Hydrostatic Pressure of Mud - Internal Pressure

        :return: Burst load in psi
        """
        # test = max((0.05194806 * self.mud_weight * self.tvd), min(I24 * B36,self.masp - self.int_gradient))
        hydrostatic_pressure = 0.05194806 * self.tvd * self.mud_weight
        internal_pressure = self.int_gradient * self.tvd
        return self.masp + hydrostatic_pressure - internal_pressure

    def calculate_burst_load2(self):
        # Gas gradient pressure (assuming 0.1 psi/ft)
        p_gas = 0.1 * self.tvd

        # Internal pressure (Pburst)
        p_burst = max(self.masp, p_gas)

        # External pressure (Pext)
        p_ext = self.mud_weight * self.tvd * 0.052

        # Safety factor
        sf = 1.1

        # Calculate burst load

        self.burst_load = max(p_ext, min(self.frac_init_pressure, self.maps))

        # Calculate burst design factor
        self.burst_d_factor = self.burst_strength / self.burst_load

        # Load = Max[(MW * D * 0.052 + Ps), (FG * D * 0.052)] - (OPG * D * 0.052)

        # Create a DataFrame for burst calculations
        self.burst_df = pd.DataFrame({
            'Depth': [self.tvd],
            'Internal Pressure': [p_burst],
            'External Pressure': [p_ext],
            'Burst Load': [self.burst_load],
            'Burst Strength': [self.burst_strength],
            'Burst Design Factor': [self.burst_d_factor]
        })

        return self.burst_df

    # def calculate_burst_pressure(self):
    #     """
    #     Calculates the burst pressure using the API formula.
    #
    #     Returns:
    #     - burst_pressure (float): Burst pressure in psi.
    #     """
    #     P_b = (2 * self.specified_burst_strength * self.wall_thickness) / (self.csg_size - self.wall_thickness)
    #     return P_b

    def calculate_burst_load3(self):
        """
        Calculates the burst load (force) based on burst pressure.

        Returns:
        - burst_load (float): Burst load in pounds-force (lb_f).
        """
        # Burst Pressure
        P_b = self.calculate_burst_pressure()  # in psi

        # Convert diameter from inches to feet for area calculation if needed
        # Assuming burst load is axial, the relevant area could be the cross-sectional area
        # However, typically burst load refers to pressure, so clarify the requirement

        # For demonstration, assume burst load as force on the cross-sectional area
        # Cross-sectional Area, A = π * (ID / 2)^2
        # ID can be calculated as D - 2t
        ID = self.csg_size - 2 * self.wall_thickness  # Inside Diameter in inches
        radius = ID / 2  # in inches
        area = 3.1416 * (radius ** 2)  # in square inches

        # Convert area to square feet for lb_f
        area_sq_ft = area / 144  # 144 sq in = 1 sq ft

        # Burst Load, F_b = P_b * A
        # Convert psi to psf (pounds force per square foot)
        # 1 psi = 144 psf
        P_b_psf = P_b * 144  # psi to psf
        F_b = P_b_psf * area_sq_ft  # in pounds-force

        return F_b

    def calculate_burst_pressure(self):
        """
        Calculate Burst Pressure.

        Args:
        - self.burst_strength (float): Burst strength in psi.
        - self.wall_thickness (float): Wall thickness in inches.
        - self.csg_size (float): Outside diameter in inches.

        Returns:
        - P_burst (float): Burst pressure in psi.
        """

        return (2 * self.burst_strength * self.wall_thickness) / self.csg_size

    def calculate_collapse_pressure(self):
        """
        Calculate Collapse Pressure.

        Args:
        - S_collapse (float): Collapse strength in psi.
        - t (float): Wall thickness in inches.
        - D (float): Outside diameter in inches.
        - mud_weigh (float): Mud weight in ppg (pounds per gallon).
        - gravity (float): Gravity in ft/s².
        - depth (float): TVD at shoe in feet.

        Returns:
        - P_collapse (float): Collapse pressure in psi.
        """
        # Convert mud weight from ppg to psi (1 ppg ≈ 0.052 * depth in feet)
        mud_pressure = self.mud_weight * 0.052 * self.tvd
        return (2 * self.collapse_strength * self.wall_thickness) / self.csg_size - mud_pressure

    def calculate_burst_df(self):
        """
        Calculate the Burst Design Factor.

        :return: Burst Design Factor (dimensionless)
        """
        burst_strength = self.calculate_burst_strength()
        burst_load = self.calculate_burst_load()
        return burst_strength / burst_load if burst_load != 0 else float('inf')

    # def calculate_tension_strength(self):

    def calculate_tension_strength(self):
        """
        Get the joint strength from the queried strength dataframe.

        :return: Joint strength in lbs
        """
        return float(self.strength_queried_df['JointStrength'].iloc[0])

    def calculate_neutral_point(self):
        return self.tvd * (1 - self.mud_weight / 65.4)

    def calculate_tension_air(self):
        """
        Calculate the tension in air.

        :param casing_weight: Weight of casing (lbs/ft)
        :param setting_depth: Setting depth of the casing (ft)
        :return: Tension in air (kips)
        """
        if self.set_depth == self.max_md_depth:
            total_weight = self.csg_weight * abs(self.set_depth - self.tol)
        else:
            total_weight = self.csg_weight * self.set_depth
        return total_weight / 1000  # Convert to kips

    def calculate_tension_buoyed(self):
        """
        Calculate the buoyed tension.

        :param casing_weight: Weight of casing (lbs/ft)
        :param mud_weight: Mud weight (ppg)
        :param setting_depth: Setting depth of the casing (ft)
        :return: Buoyed tension (kips)
        """
        # self.max_tvd
        # result1 = self.csg_weight * self.set_depth
        result1 = math.pi / 4 * (self.csg_size ** 2 - self.csg_internal_diameter ** 2)
        # result2 = 0.05194806 * self.mud_weight * self.tvd
        if self.set_depth == self.max_md_depth:
            result2 = 0.05194806 * self.mud_weight * abs(self.tvd - self.max_tvd_depth)
            result3 = self.csg_weight * abs(self.set_depth - self.tol)

        else:
            result2 = 0.05194806 * self.mud_weight * self.tvd
            result3 = self.csg_weight * self.set_depth

        output = (result3 - result2 * result1) / 1000
        return output

    # #     abs(self.set_depth - self.tol) -
    #     buoyancy_factor = self.mud_weight / 65.4
    #     if self.set_depth == self.max_depth:
    #         buoyed_weight = self.csg_weight * (1 - buoyancy_factor) * abs(self.set_depth-self.tol)
    #     else:
    #         buoyed_weight = self.csg_weight * (1 - buoyancy_factor) * self.set_depth
    #
    #     # buoyed_weight = self.csg_weight * (1 - buoyancy_factor) * self.set_depth
    #     return buoyed_weight / 1000  # Convert to kips

    # def calculate_tension_strength(self):
    #     """
    #     Calculate the tension strength of the casing.
    #
    #     :param yield_strength: Yield strength of the casing material (psi)
    #     :param casing_area: Cross-sectional area of the casing (sq in)
    #     :return: Tension strength (kips)
    #     """
    #     return (yield_strength * casing_area) / 1000  # Convert to kips

    def calculate_tension_df(self):
        """
        Calculate the tension design factor.

        :param tension_buoyed: Buoyed tension (kips)
        :param tension_strength: Tension strength of the casing (kips)
        :return: Tension design factor (dimensionless)
        """
        if self.tension_buoyed == 0:
            return float('inf')  # To avoid division by zero
        return self.tension_strength / self.tension_buoyed

    def calculate_body_yield(self):
        """
        Get the body yield strength from the queried strength dataframe.

        :return: Body yield strength in lbs
        """
        return float(self.strength_queried_df['BodyYield'].iloc[0])

    def calculate_drift_diameter(self):
        """
        Get the drift diameter from the queried strength dataframe.

        :return: Drift diameter in inches
        """
        return float(self.strength_queried_df['DriftDiameterAPI'].iloc[0])

    def calcBurstLoad(self, frac_gradient, tvd_depth, shoe_press, mud_weight, internal_gradient, next_internal_gradient, backup_mud):
        frac_pressure = frac_gradient * tvd_depth
        part1 = self.calcBurstLoadPart1(backup_mud, mud_weight, tvd_depth)
        part2 = self.calcBurstLoadPart2(frac_pressure, tvd_depth, next_internal_gradient, backup_mud, shoe_press, internal_gradient)
        max_all = max(part1, part2)

        return max_all

    def calcBurstLoadPart1(self, backup_mud, mud_weight, tvd_val):
        return 0.05194806 * (mud_weight - backup_mud) * tvd_val

    def calcBurstLoadPart2(self, frac_pressure, tvd_val, next_internal_gradient, backup_mud, shoe_press, internal_gradient):
        minPart1 = frac_pressure - (tvd_val - tvd_val) * next_internal_gradient - (0.05194806 * backup_mud * tvd_val)
        # =($I$24-($B$19-DxSurvey!H57)*$B$36-(0.05194806*$B$22*DxSurvey!H57))
        minPart2 = shoe_press - internal_gradient * (tvd_val - tvd_val) - (0.05194806 * backup_mud * tvd_val)
        # =$L$24-$B$21*($B$19-DxSurvey!H57)-(0.05194806*$B$22*DxSurvey!H57)

        return min(minPart1, minPart2)

pd.set_option('display.max_columns', None)  # Show all columns when displaying DataFrames
pd.options.mode.chained_assignment = None  # Suppress chained assignment warnings


conn = sqlite3.connect('sample_casing.db')
wb_df = pd.read_sql('SELECT * FROM wb_info', conn)
hole_df = pd.read_sql('SELECT * FROM hole_parameters', conn)
casing_df = pd.read_sql('SELECT * FROM casing', conn)
string_df = pd.read_sql('SELECT * FROM string_parameters', conn)

wb_df['casing_depths'] = wb_df['casing_depths'].apply(lambda row: ast.literal_eval(row))
# conn.close()
# conductor_casing_bottom = 100
# casing_depths = [2450, 9350, 9246, 20512]
# top_of_liner = 9246
# max_depth_md = 20512
# max_depth_tvd = 9208.15



wellbore = WellBoreExpanded(
    name='Wellbore (Planned)', top=wb_df['conductor_casing_bottom'].iloc[0], bottom=wb_df['max_depth_md'].iloc[0], method='top_down', tol=wb_df['top_of_liner'].iloc[0],
    max_md_depth=wb_df['max_depth_md'].iloc[0], max_tvd_depth=wb_df['max_depth_tvd'].iloc[0])
# wellbore = WellBoreExpanded(
#     name='Wellbore (Planned)', top=conductor_casing_bottom, bottom=max_depth_md, method='top_down', tol=top_of_liner,
#     max_md_depth=max_depth_md, max_tvd_depth=max_depth_tvd)
# f"""  Hole Size Csg Size Set Depth Csg Wt. Csg Grade Csg Collar Lead Qty  \
# 0     12.25    9.625      2481    36.0      J-55        LTC      286
#
#   Lead Yield Tail Qty Tail Yield
# 0        2.6      166       1.13
#                 1       2
# 0          Buoyed       y
# 1              MW    11.0
# 2             TVD  2481.0
# 3    Hole Washout      10
# 4  Internal Grad.    0.12
# 5      Backup Mud     0.0
# 6    Internal Mud     0.0"""
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
for i in range(len(wellbore.sections)):
    data_dict = wellbore.sections[i]
    print(data_dict)
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