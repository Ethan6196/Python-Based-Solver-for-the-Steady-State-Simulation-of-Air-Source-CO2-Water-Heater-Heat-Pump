import CoolProp.CoolProp as CP
import numpy as np
from Functions import Nu_air as Nusselt

def evaporator_air(P_a, T_ai, T_s, v_air, N_rows, pt, pl, Do, L):

    # Define properties
    fluid = 'air'
    rho = CP.PropsSI('D', 'P', P_a, 'T', T_ai, fluid)
    mu = CP.PropsSI('viscosity', 'P', P_a, 'T', T_ai, fluid)
    Pr = CP.PropsSI('Prandtl', 'P', P_a, 'T', T_ai, fluid)
    k = CP.PropsSI('conductivity', 'P', P_a, 'T', T_ai, fluid)

    Pr_s = CP.PropsSI('Prandtl', 'P', P_a, 'T', T_s, fluid)

    Pd = np.sqrt(((0.5*pt)**2) + (pl**2))
    Ad = (Pd - Do)*L
    if Pd < (pt + Do)/2:
        V_max = v_air*(pt/2*(Pd - Do))
    else:
        V_max = v_air*pt/(pt-Do)

    m_dot = rho*V_max*Ad*2
    Re = (rho*V_max*Do)/mu

    # Calculate Nusselt number
    Nu = Nusselt(Re, Pr, Pr_s, pt, pl, N_rows)

    h_air = Nu*k/Do # Convective heat transfer coefficient

    return h_air, m_dot

def UA_fin(h_air, h_co2, k_wall, L, Di, Do, t_fin, t_spacing, pt, pl, N_tubes, N_rows):

    # Define fin parameters
    m = np.sqrt(2*h_air/(k_wall*t_fin))
    r1 = Do/2
    req = np.sqrt(pt*pl/np.pi)
    rc = req + 0.5*t_fin
    C_2 = 2*r1/(m*((rc**2) - (r1**2)))
    A_fin = 2*np.pi*((req**2) - (r1**2))

    # Define bessel functions & fin effic
    from scipy.special import jv, yv
    # jv: 1st kind
    # yv: 2nd kind

    fin_efficiency = C_2*(yv(1, m*r1)*jv(1, m*rc) - jv(1, m*r1)*yv(1, m*rc))/(jv(0, m*r1)*yv(1, m*rc) + yv(0, m*r1)*jv(1, m*rc))
    
    # Define number of fins 
    import math
    N_fins_per_tube = np.floor(L/(t_fin + t_spacing))
    A_fin_per_tube = N_fins_per_tube*A_fin
    L_fins = N_fins_per_tube*t_fin
    L_tot_fins = N_fins_per_tube*t_spacing
    L_unfinned = L_tot_fins - L_fins

    A_unfinned = np.pi*Do*L_unfinned

    # Define Thermal Resistance
    R_air = 1/(h_air*(A_unfinned + A_fin_per_tube*fin_efficiency)*N_rows*N_tubes)

    R_wall = np.log(Do/Di)/(k_wall*np.pi*L*2*N_tubes*N_rows)

    R_co2 = 1/(h_co2*np.pi*Di*L*N_tubes*N_rows)

    UA = (R_wall + R_co2 + R_air)**-1
    
    return UA

def Evap_CO2( P, h, Di, L, m_dot, Q):

    # Constants
    C_1 = 1.4e4
    C_2 = 0.93

    # vapor and liquid properties
    rho_g = CP.PropsSI('D', 'P', P, 'Q', 1, 'CO2')
    rho_l = CP.PropsSI('D', 'P', P, 'Q', 0, 'CO2')
    mu_g = CP.PropsSI('viscosity', 'P', P, 'Q', 1, 'CO2')
    mu_l = CP.PropsSI('viscosity', 'P', P, 'Q', 0, 'CO2')
    h_l = CP.PropsSI('H', 'P', P, 'Q', 0, 'CO2')
    h_g = CP.PropsSI('H', 'P', P, 'Q', 1, 'CO2')
    h_lg = h_g - h_l
    
    # Vapor quality from enthalpy
    
    x = (h - h_l) / h_lg
    
    
    Xtt = (((1-x)/x)**0.9)*((rho_g/rho_l)**0.5)*((mu_l/mu_g)**0.1)
    
    k_l = CP.PropsSI('conductivity', 'P', P, 'Q', 0, 'CO2')
    Pr_l = CP.PropsSI('Prandtl', 'P', P, 'Q', 0, 'CO2')

    m_dot_l = m_dot*(1-x)
    Re_l = 4*m_dot_l/(np.pi*Di*mu_l)

    h_lo = 0.023*((Re_l)**0.8)*((Pr_l)**0.4)*(k_l/Di)

    A_flow = np.pi*(Di**2)/4 # Cross Sectional Area
    A_ht = np.pi*Di*L # Surface Area

    Bo = (Q*A_flow)/(A_ht*m_dot*h_lg)
    
    h_evap = h_lo*(C_1*Bo + C_2*np.power(1/Xtt, 2/3))

    return h_evap


def Dry_Evaporator2(T_CO2_in, h_CO2_in, P_CO2_in, T_air_in, P_air, L_evap, m_dot_CO2, v_air,
                   k_wall, Di, Do, N_rows, N_tubes, pt, pl, t_fin, t_spacing):
    
    from scipy.optimize import fsolve
    from GC_correlations import GC_CO2_2

    # 1. Saturation Properties
    T_sat = CP.PropsSI('T', 'P', P_CO2_in, 'Q', 0, 'CO2')
    h_g = CP.PropsSI('H', 'P', P_CO2_in, 'Q', 1, 'CO2')
    Cp_air = CP.PropsSI('C', 'T', T_air_in, 'P', P_air, 'air')

    # Early exit: no phase change if air is too cold
    if T_air_in <= T_sat:
        return T_CO2_in, h_CO2_in, T_air_in, 0.0, 0.0

    # 2. Air-side properties (assumed constant along L)
    Q_required = m_dot_CO2 * (h_g - h_CO2_in)

    # Calculate wall temperature
    def resid_T_wall(T_avg):
        h_air, m_dot_air = evaporator_air(P_air, T_air_in, T_avg, v_air, N_rows, pt, pl, Do, L_evap)
        h_co2 = Evap_CO2(P_CO2_in, h_CO2_in, Di, L_evap, m_dot_CO2 / (N_tubes * N_rows), Q_required/(N_tubes*N_rows))
        T_wall = (T_air_in*h_air + T_CO2_in*h_co2)/(h_air+h_co2)

        return T_avg - T_wall
    
    T_avg = fsolve(resid_T_wall, x0=T_air_in)[0]

    h_air, m_dot_air = evaporator_air(P_air, T_air_in, T_avg, v_air, N_rows, pt, pl, Do, L_evap)
    C_air = Cp_air * m_dot_air

    # 3. Define residual for L_tp (two-phase length)
    def residual_Ltp(L_tp):
        h_air, m_dot_air = evaporator_air(P_air, T_air_in, T_avg, v_air, N_rows, pt, pl, Do, L_tp)
        C_air = Cp_air * m_dot_air
        h_co2 = Evap_CO2(P_CO2_in, h_CO2_in, Di, L_tp, m_dot_CO2 / (N_tubes * N_rows), Q_required/(N_tubes*N_rows))
        UA = UA_fin(h_air, h_co2, k_wall, L_tp, Di, Do, t_fin, t_spacing, pt, pl, N_tubes, N_rows)
        
        NTU = UA / C_air
        eps = 1 - np.exp(-NTU)
        Q_air = eps * C_air * (T_air_in - T_sat)
        return Q_air - Q_required

    sol = fsolve(residual_Ltp, x0=L_evap)
    L_2p = sol[0]

    # 6. Superheat region (if any)
    L_sph = max(L_evap - L_2p, 0)
    if L_sph > 0 and L_2p > 0:
        
        C_air = Cp_air * m_dot_air
        h_co2, Cp_co2 = GC_CO2_2(m_dot_CO2 / (N_tubes * N_rows), P_CO2_in, Di, L_sph)
        UA_sh = UA_fin(h_air, h_co2, k_wall, L_sph, Di, Do, t_fin, t_spacing, pt, pl, N_tubes, N_rows)
        
        C_CO2 = Cp_co2 * m_dot_CO2
        c_min = min(C_CO2, C_air)
        c_max = max(C_CO2, C_air)
        c = c_min / c_max
        NTU_sh = UA_sh / c_min
        eps_sph = 1 - np.exp(((NTU_sh**0.22)/c)*(np.exp(-c * NTU_sh ** 0.78) - 1))
        delta_T_sph = T_air_in - T_sat
        Q_sh = eps_sph * c_min * delta_T_sph
    else:
        Q_sh = 0.0

    # 7. Total heat transfer
    Q_total = Q_required + Q_sh
    h_CO2_out = h_CO2_in + Q_total / m_dot_CO2
    T_CO2_out = CP.PropsSI('T', 'H', h_CO2_out, 'P', P_CO2_in, 'CO2')
    T_air_out = T_air_in - Q_total / (m_dot_air * Cp_air)

    return T_CO2_out, h_CO2_out, T_air_out, Q_total, L_2p
