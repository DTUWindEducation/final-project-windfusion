# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 12:16:58 2025

@author: Γιώργος
"""

import numpy as np
import matplotlib.pyplot as plt
from Classes import GeneralWindTurbine, WindTurbine

# Load power curve CSV
power_curve_data = np.loadtxt('LEANWIND_Reference_8MW_164.csv', delimiter=',', skiprows=1)

# Extract rated power from data (maximum power value

# Create GeneralWindTurbine object
general_turbine = GeneralWindTurbine(
    rotor_diameter=164,
    hub_height=110,
    rated_power=8000,
    v_in=4,
    v_rated=12.5,
    v_out=25,
    name="General Model"
)

# Create WindTurbine object with CSV data
real_turbine = WindTurbine(
    rotor_diameter=164,
    hub_height=110,
    rated_power=8000,
    v_in=4,
    v_rated=12.5,
    v_out=25,
    power_curve_data=power_curve_data,
    name="Interpolated Model"
)

# Wind speed range
wind_speeds = np.linspace(0, 30, 200)

# Get power output for both turbines
general_power = [general_turbine.get_power(v) for v in wind_speeds]
real_power = [real_turbine.get_power(v) for v in wind_speeds]

# Plot
plt.figure(figsize=(10, 6))
plt.plot(wind_speeds, general_power, label='GeneralWindTurbine (Analytical)', linestyle='--')
plt.plot(wind_speeds, real_power, label='WindTurbine (Interpolated)', linewidth=2)
plt.xlabel('Wind Speed [m/s]')
plt.ylabel('Power Output [kW]')
plt.title('Wind Turbine Power Curve Comparison')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
