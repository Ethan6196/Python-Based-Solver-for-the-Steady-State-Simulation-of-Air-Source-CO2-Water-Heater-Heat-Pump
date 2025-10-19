# Python-Based-Solver-for-the-Steady-State-Simulation-of-Air-Source-CO2-Water-Heater-Heat-Pump
Skripsie Project - 2025
Ensure all the following imports are included when executing the Main.Py

CoolProp --> pip install CoolProp
numpy --> pip install numpy
matplotlib --> pip install matplotlib

Additionally, ensure you have all the files downloaded when executing Main.py.

The files contain the funtions and correlations relevant to the steady-state simulation of the heat pump cycle

Compressor2.py --> determines the mass flow rate and isentropic and volumetric efficiencies
Evaporator.py --> determines the heat transfer correlations and solves the outlet state of the evaporator
Gas_cooler.py --> determines the outlet state of the gas cooler
GC_correlations.py --> determines the heat transfer coefficients for the gas cooler
Expansion_valve1.py --> solves the mass flow rate for the specified up- and downstream pressures of the EV
Main.py --> takes in all the geometric parameters of each component and also the boundary conditions for the simulation

The files with the suffix "_sizing" are for sizing the parameters of the system, namely the evaporator length and the expansion valve throat diameter

GC_Pdrop.py contains the same code and structure as Gas_cooler.py, however it additionally accounts for the pressure drop along the length of the gas cooler.
