from Evaporator2 import evaporator_air as EV
from Evaporator2 import Evap_CO2 as CO2
from Evaporator2 import UA_fin
import CoolProp.CoolProp as CP
import numpy as np

from scipy.optimize import fsolve
import numpy as np
import CoolProp.CoolProp as CP

def evaporator_root_solver():
    # Constants
    Do = 22.2e-3  # m
    Di = 18.9e-3  # m
    k = 401       # W/mK
    t_fin = 0.5e-3  # m
    t_spacing = 0.2e-3  # m
    pt = 23e-3    # m
    pl = 35e-3    # m
    Ntubes = 10
    Nrows = 15
    Q_tot = 242.9049e3  # W -- From Smath
    Q_2ph = 233.213e3  # W  -- From Smath
    Q_sh = Q_tot - Q_2ph
    P_gc = 9e6  # Pa
    T_co2_in = -10 + 273.15 # K 
    T_co2_out = -5 + 273.15 # K
    T_gc_out = 25.6601 + 273.15 # K -- From Smath
    T_ai = -5 + 273.15  # K
    v_air = 5  # m/s
    P_a = 101325  # Pa
    fluid = 'CO2'
    m_dot_co2 = 1.3417 / (Ntubes * Nrows) # kg/s -- From Smath

    def residual(L):
        from Evaporator2 import evaporator_air as EV
        from GC_correlations import GC_CO2 as CO2S
        from Evaporator2 import UA_fin
        from Evaporator2 import Evap_CO2 as CO2

        T_co2_avg = (T_co2_in + T_co2_out) / 2
        T_s = (T_ai + T_co2_avg) / 2
        print("Surface Temperature Estimation:", T_s - 273.15, 'C')
        print("\n")

        L_2ph = L * (Q_2ph / Q_tot)
        L_sh = L - L_2ph
        print('2 phase length: ', L_2ph, 'm, Superheat Length: ', L_sh, 'm, Total Length: ', L, 'm')

        # Two Phase Region
        h_air, m_dot_air = EV(P_a, T_ai, T_s, v_air, Nrows, pt, pl, Do, L)
        print("HT_coeff Air:", h_air, 'W/m^2K')
        print("Mass flow of air:", m_dot_air, "kg/s")
        print("\n")

        P_suc = CP.PropsSI('P', 'T', T_co2_in, 'Q', 1, fluid)
        h = CP.PropsSI('H', 'T', T_gc_out, 'P', P_gc, fluid)

        h_co2_2ph = CO2(P_suc, h, Di, L_2ph, m_dot_co2, Q_2ph / (Ntubes * Nrows))
        print("HT-coeff CO2: ", h_co2_2ph, 'W/m^2K')

        UA_2ph, fin_eff_2ph = UA_fin(h_air, h_co2_2ph, k, L_2ph, Di, Do, t_fin, t_spacing, pt, pl, Ntubes, Nrows)
        print("UA Evaporator (2ph):", UA_2ph, 'W/K')
        print("Fin Efficiency (2ph):", fin_eff_2ph)
        print("\n")

        # Superheated Region
        h_co2_sh, _ = CO2S(m_dot_co2, T_co2_in, P_gc, Di, L_sh)
        print("HT-coeff CO2:", h_co2_sh, 'W/m^2K')

        UA_sh, fin_eff_sh = UA_fin(h_air, h_co2_sh, k, L_sh, Di, Do, t_fin, t_spacing, pt, pl, Ntubes, Nrows)
        print("UA Evaporator (sh):", UA_sh, 'W/K')
        print("Fin Efficiency (2ph):", fin_eff_sh)

        print('Total UA:', UA_sh + UA_2ph, 'W/m^2K')
        print("\n")

        Cp_air = CP.PropsSI('C', 'T', T_ai, 'P', 101325, 'air')
        T_ao = T_ai - Q_tot / (m_dot_air * Cp_air)
        print('Air Outlet Temperature:', T_ao - 273.15, 'C')

        NTU_2ph = UA_2ph / (Cp_air * m_dot_air)
        eps = 1 - np.exp(-NTU_2ph)
        print('NTU_2ph = ', NTU_2ph)

        Q_air_2ph = eps * (Cp_air * m_dot_air) * (T_ai - T_co2_in)

        Cp_CO2 = CP.PropsSI('C', 'T', T_co2_out, 'P', P_suc, fluid)
        C_CO2 = Cp_CO2 * m_dot_co2 * Ntubes * Nrows
        C_air = Cp_air * m_dot_air

        c_min = min(C_CO2, C_air)
        c_max = max(C_CO2, C_air)
        c = c_min / c_max

        NTU_sh = UA_sh / c_min
        eps_sh = 1 - np.exp(((NTU_sh**0.22) / c) * (np.exp(-c * NTU_sh ** 0.78) - 1))
        dT = T_ai - T_co2_in
        Q_air_sh = eps_sh * c_min * dT

        Q_air_tot = Q_air_sh + Q_air_2ph

        print("Total Heat Transfer", Q_air_tot / 1e3, 'kW')
        print("Total Heat Transfer ", Q_tot / 1e3, 'kW')
        print("Q Residual = ", (Q_air_tot - Q_tot) / 1e3, 'kW')
        print("======================================")
        print("\n")
        return Q_air_tot - Q_tot

    # Root finding for length L
    sol = fsolve(residual, x0=10)[0]
    print("✅ Optimal Length Found:", sol, "m")
    

# Call root solver
evaporator_root_solver()
