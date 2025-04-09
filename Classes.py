# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 11:59:48 2025

@author: Γιώργος
"""
import numpy as np

class GeneralWindTurbine:
    def __init__(self, rotor_diameter, hub_height, rated_power, v_in, v_rated, v_out, name=None):
        self.rotor_diameter = rotor_diameter
        self.hub_height = hub_height
        self.rated_power = rated_power
        self.v_in = v_in
        self.v_rated = v_rated
        self.v_out = v_out
        self.name = name

    def get_power(self, v):
        if v < self.v_in or v > self.v_out:
            return 0
        elif self.v_in <= v < self.v_rated:
            return self.rated_power * (v / self.v_rated) ** 3
        elif self.v_rated <= v <= self.v_out:
            return self.rated_power


class WindTurbine(GeneralWindTurbine):
    def __init__(self, rotor_diameter, hub_height, rated_power, v_in, v_rated, v_out, power_curve_data, name=None):
        super().__init__(rotor_diameter, hub_height, rated_power, v_in, v_rated, v_out, name)
        self.power_curve_data = power_curve_data

    def get_power(self, v):
        wind_speeds = self.power_curve_data[:, 0]
        power_values = self.power_curve_data[:, 1]
        # Use np.interp for linear interpolation
        return float(np.interp(v, wind_speeds, power_values))

