import CoolProp.CoolProp as CP
import numpy as np
from scipy.optimize import root_scalar,fsolve


fluid = 'CO2'
SH = 5
T_out = -5 + 273.15 # K
T_wi = 15 + 273.15   # K
m_dot_water = 2 # kg/s
T_4 = T_out - SH

T1 = T_out


P_suction = CP.PropsSI('P', 'T', T_4, 'Q', 1, fluid)

print('Suction Pressure:', f"{P_suction/1e6:.3f} MPa")

# Enthalpy Bandwidth #
h_evap_bandwith = np.array([
    CP.PropsSI('H', 'P', P_suction, 'Q', 0, fluid),
    CP.PropsSI('H', 'P', P_suction, 'Q', 1, fluid)
])

print('Enthalpy Bandwidth for Expansion valve outlet: [', f"{h_evap_bandwith[0]/1e3:.3f}", ',', f"{h_evap_bandwith[1]/1e3:.3f}", '] kJ/kg')


P_discharge = 9e6
r = P_discharge/P_suction

print('Discharge Pressure: ', f"{P_discharge/1e6:.3f} MPa")
print('\n')

# Compressor Geometry #
Bore = 48  # mm
Stroke = 28.5  # mm
N_cylinders = 4
N_compressor = 1200 # rpm

from Compressor2 import Compressor as Comp
from Gas_Cooler import gas_cooler2 as GC

# Gas Cooler Geometry #
N_gc = 15
L_gc = 13 #m
Di_inner_gc = 8.8e-3 # m --> inner diameter of inner tube
Do_inner_gc = 10e-3 # m --> outer diameter of inner tube
Di_outer_gc = 12.6e-3 # m --> inner diameter of outer tube

k_wall = 401  # W/mK

# Expansion Valve #
z = 1

m_dot, w, h1 = Comp(T1, P_suction, P_discharge, Bore, Stroke, N_compressor, N_cylinders)
print("Mass Flow rate of CO2: ", m_dot, 'kg/s')

W_dot = m_dot*w
print('Compressor Work: ', W_dot/1e3, 'kW')

h2 = h1 + w
T2 = CP.PropsSI('T', 'P', P_discharge, 'H', h2, "CO2")
T3, h3, T_water_out, Q_total = GC(N_gc, T2, P_discharge, T_wi, 1e6, L_gc, m_dot, m_dot_water, k_wall, Di_inner_gc, Do_inner_gc, Di_outer_gc, 65, True)

print("\n")
print('Gas Cooler Heat Capacity:', Q_total/1e3, 'kW')
print('Water outlet Temperature:', T_water_out-273.15, 'C')
print("CO2 Temperatures", T1-273.15, T2-273.15, T3-273.15, 'C')

def solve_expansion_valve_throat_diameter(m_dot_c, z, T_0i, P_0i, P_0e, D_throat_guess):
    def expansion_valve_mass_flow(D_throat):
        import CoolProp.CoolProp as CP
        import numpy as np

        fluid = 'CO2'
        P_vc = 0.45 * P_0i
        P_cr = CP.PropsSI('Pcrit', fluid)
        rho_cr = CP.PropsSI('rhocrit', fluid)
        T_cr = CP.PropsSI('Tcrit', fluid)

        if z <= 0 or z > 1:
            raise ValueError('Valve opening z must be between 0 and 1')

        if T_0i >= T_cr:
            P_s = P_cr
            rho_s = rho_cr
        else:
            P_s = CP.PropsSI('P', 'T', T_0i, 'Q', 1, fluid)
            rho_s = CP.PropsSI('D', 'T', T_0i, 'Q', 1, fluid)

        A_o = np.pi * (D_throat**2) / 4
        rho_i = CP.PropsSI('D', 'T', T_0i, 'P', P_0i, fluid)
        F_F = 0.96 - 0.28 * np.sqrt(P_s / P_cr)

        if P_0e > 130e6:
            raise ValueError("P_0e exceeds safe pressure limit")

        F_L = np.sqrt((P_0i - P_0e) / (P_0i - P_vc))
        dP_cr = (F_L**2) * (P_0i - F_F * P_s)

        if np.abs(P_0i - P_0e) < dP_cr:
            X = dP_cr / P_0i
        else:
            X = np.abs(P_0i - P_0e) / dP_cr

        Y = 1 - X * P_0i / (3 * dP_cr)
        Cd = 1.1075 * z**0.4436 * (rho_i / rho_s)**-1.4971 * ((P_0i - P_s)/(X * P_0i))**0.0131
        m_dot = Cd * A_o * Y * np.sqrt(2 * rho_s * P_0i * X)

        return m_dot

    # Define residual function for fsolve
    def residual(D_throat):
        try:
            m_dot_valve = expansion_valve_mass_flow(D_throat)
            return m_dot_valve - m_dot_c
        except Exception:
            return 1e6  # Large residual if calculation fails

    from scipy.optimize import fsolve
    D_solution = fsolve(residual, D_throat_guess)[0]

    return D_solution

D_throat_guess = 18e-3
D_throat = solve_expansion_valve_throat_diameter(m_dot, z, T3, P_discharge, P_suction, D_throat_guess)

print('Throat Diameter: ', D_throat*1e3)
print('COP:', Q_total/W_dot)
