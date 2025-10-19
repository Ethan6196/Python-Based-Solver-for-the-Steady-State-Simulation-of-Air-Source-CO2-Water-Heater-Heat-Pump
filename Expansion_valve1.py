import CoolProp.CoolProp as CP
import numpy as np
from scipy.optimize import fsolve

def solve_expansion_valve_outlet_pressure(m_dot_c, z, T_0i, P_0i, P_0e_guess, D_throat): # RootFinding

    def expansion_valve_mass_flow(P_0e):
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

        F_L = np.sqrt((P_0i - P_0e) / (P_0i - P_vc))
        dP_cr = (F_L**2) * (P_0i - F_F * P_s)

        if np.abs(P_0i - P_0e) < dP_cr:
            X = dP_cr / P_0i
        else:
            X = np.abs(P_0i - P_0e) / dP_cr

        Y = 1 - X * P_0i / (3 * dP_cr)
        Cd = 1.1075 * (z**0.4436) * ((rho_i / rho_s)**-1.4971) * (((P_0i - P_s)/(X * P_0i))**0.0131)

        m_dot = Cd * A_o * Y * np.sqrt(2 * rho_s * P_0i * X)
        
        return m_dot - m_dot_c


    # Solve using fsolve
    P_0e_solution = fsolve(expansion_valve_mass_flow, P_0e_guess, xtol=1e-4)[0]

    return m_dot_c , P_0e_solution
