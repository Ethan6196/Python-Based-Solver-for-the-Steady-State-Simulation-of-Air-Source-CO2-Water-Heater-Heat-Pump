import CoolProp.CoolProp as CP
import numpy as np
import matplotlib.pyplot as plt

def evaporator_air(P_a, T_ai, T_s, v_air, N_rows, pt, pl, Do, L):
    from Functions import Nu_air as Nusselt
    # Define properties
    fluid = 'air'
    rho = CP.PropsSI('D', 'P', P_a, 'T', T_ai, fluid)
    mu = CP.PropsSI('viscosity', 'P', P_a, 'T', T_ai, fluid)
    Pr = CP.PropsSI('Prandtl', 'P', P_a, 'T', T_ai, fluid)
    k = CP.PropsSI('conductivity', 'P', P_a, 'T', T_ai, fluid)
    # Since we know that air is the hotter fluid
    # Assume T_s is some average between CO2 and air and to be constant

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

    R_air = 1/(h_air*(A_unfinned + A_fin_per_tube*fin_efficiency)*N_rows*N_tubes)

    R_wall = np.log(Do/Di)/(k_wall*np.pi*L*2*N_tubes*N_rows)

    R_co2 = 1/(h_co2*np.pi*Di*L*N_tubes*N_rows)

    UA = (R_wall + R_co2 + R_air)**-1
    
    return UA, fin_efficiency

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
    #x = np.clip(x, 1e-5, 1 - 1e-5)  # Prevents numerical issues
    
    
    Xtt = (((1-x)/x)**0.9)*((rho_g/rho_l)**0.5)*((mu_l/mu_g)**0.1)
    
    k_l = CP.PropsSI('conductivity', 'P', P, 'Q', 0, 'CO2')
    Pr_l = CP.PropsSI('Prandtl', 'P', P, 'Q', 0, 'CO2')

    m_dot_l = m_dot#*(1-x)
    Re_l = 4*m_dot_l/(np.pi*Di*mu_l)

    h_lo = 0.023*((Re_l)**0.8)*((Pr_l)**0.4)*(k_l/Di)

    A_flow = np.pi*(Di**2)/4
    A_ht = np.pi*Di*L

    Bo = (Q*A_flow)/(A_ht*m_dot*h_lg)
    
    h_evap = h_lo*(C_1*Bo + C_2*np.power(1/Xtt, 2/3))

    return h_evap

