import numpy as np
import CoolProp.CoolProp as CP
import matplotlib.pyplot as plt

# User Defined Functions
from Functions import  Friction, Nusselt
def GC_CO2(m_dot, T_i, P_gc, D, L):
    
    fluid = 'CO2' # Specify refrigerant
    
    mu_i = CP.PropsSI('V', 'T', T_i, 'P', P_gc, fluid)
    k_i = CP.PropsSI('conductivity', 'T', T_i, 'P', P_gc, fluid)
    Pr_i = CP.PropsSI('Prandtl', 'T', T_i, 'P', P_gc, fluid)
    Cp_i = CP.PropsSI('C', 'T', T_i, 'P', P_gc, fluid)
    A = np.pi*(D**2)/4
    G = m_dot/A
    Re_bulk = G*D/mu_i
    f_co2 = Friction(Re_bulk)

    if Re_bulk > 2300:
        Nu_co2 = Nusselt(f_co2, Re_bulk, Pr_i)
    else:
        Nu_co2 = 4.36
   
    h_co2 = Nu_co2*k_i/D

    return h_co2, Cp_i

def GC_water(m_dot, T_i, P_gc, Do, Di, L):
    fluid = 'water'

    Dh = Do - Di
    
    mu_i = CP.PropsSI('V', 'T', T_i, 'P', P_gc, fluid)
    k_i = CP.PropsSI('conductivity', 'T', T_i, 'P', P_gc, fluid)
    Pr_i = CP.PropsSI('Prandtl', 'T', T_i, 'P', P_gc, fluid)
    Cp_i = CP.PropsSI('C', 'T', T_i, 'P', P_gc, fluid)

    A = np.pi*((Do**2) - (Di**2))/4
    G = m_dot/A

    Re_bulk = G*Dh/mu_i

    f_w = Friction(Re_bulk)
    Nu_w = Nusselt(f_w, Re_bulk, Pr_i)

    h_w = Nu_w * k_i / Dh

    return h_w, Cp_i

def GC_CO2_2(m_dot, P_gc, D, L):
    
    fluid = 'CO2' # Specify refrigerant
    
    mu_i = CP.PropsSI('V', 'Q', 1, 'P', P_gc, fluid)
    k_i = CP.PropsSI('conductivity', 'Q', 1, 'P', P_gc, fluid)
    Pr_i = CP.PropsSI('Prandtl', 'Q', 1, 'P', P_gc, fluid)
    Cp_i = CP.PropsSI('C', 'Q', 1, 'P', P_gc, fluid)

    Re_bulk = 4*m_dot/(mu_i*np.pi*D )
    f_co2 = Friction(Re_bulk)

    if Re_bulk > 2300:
        Nu_co2 = Nusselt(f_co2, Re_bulk, Pr_i)
    else:
        Nu_co2 = 4.36
   
    h_co2 = Nu_co2*k_i/D

    return h_co2, Cp_i