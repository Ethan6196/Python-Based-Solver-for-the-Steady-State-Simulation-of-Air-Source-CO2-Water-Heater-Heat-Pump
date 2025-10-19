import CoolProp.CoolProp as CP
import numpy as np


def Compressor(T_0i, p_0i, p_0e, Bore, Stroke, N_comp, n_cyl):

    # Define Compressor Geometric parameters
    Bore = Bore/1000
    Stroke = Stroke/1000
    Isen_eff = np.zeros(5)
    Vol_eff = np.zeros(5)

    r = p_0e/p_0i
    V_cyl = np.pi*((Bore/2)**2)*Stroke
    V_tot = V_cyl*n_cyl
    #V_tot = 1.5e-6
    #print(V_tot)
    Vd = V_tot*N_comp * 2*np.pi/60
    #print(Vd)
    fluid = 'CO2'

    # Define Efficency Correlations
    Isen_eff[0] = 0.6162 + (0.0611) * r + (-0.0123) * r**2
    Isen_eff[1] = 0.5839 + (0.0905) * r + (-0.0181) * r**2
    Isen_eff[2] = 0.7310 + (0.0162) * r + (-0.0116) * r**2
    Isen_eff[3] = 0.5736 + (0.1371) * r + (-0.0295) * r**2
    Isen_eff[4] = 0.5217 + (0.1818) * r + (-0.0398) * r**2

    Vol_eff[0] = 1.0840 + (-0.1088) * r + (0.0045) * r**2
    Vol_eff[1] = 1.0406 + (-0.0818) * r + (0.0000) * r**2
    Vol_eff[2] = 1.1280 + (-0.1304) * r + (0.0052) * r**2
    Vol_eff[3] = 1.1372 + (-0.1156) * r + (0.0031) * r**2
    Vol_eff[4] = 1.1372 + (-0.1156) * r + (0.0031) * r**2

    # Assume that there is always a set degree of Superheat
    h_0i = CP.PropsSI('H', 'P', p_0i, 'T', T_0i, fluid)
    s_in = CP.PropsSI('S', 'P', p_0i, 'T', T_0i, fluid)
    h_0es = CP.PropsSI('H', 'P', p_0e, 'S',s_in, fluid)

    rho = CP.PropsSI('D', 'T', T_0i, 'P', p_0i, fluid)
   
    
    Isen_efficiency = -0.26 + 0.7952*r - 0.2803*(r**2) + 0.0414*(r**3) - 0.0022*(r**4)
    Vol_efficiency = 0.9207 - 0.0756*(r) + 0.0018*(r**2)


    
    # Assign relevant efficiency correlations for temperature range they are valid for
    
    if T_0i <= -10+273.15:
        m_dot = Vol_eff[0]*Vd*rho
        w = (h_0es - h_0i)/Isen_eff[0]
       
    elif T_0i > -10+273.15 and T_0i <= -5+273.15:
        m_dot = Vol_eff[1]*Vd*rho
        w = (h_0es - h_0i)/Isen_eff[1]
       

    elif T_0i > -5+273.15 and T_0i <= 273.15:
        m_dot = Vol_eff[2]*Vd*rho
        w = (h_0es - h_0i)/Isen_eff[2]
       
    elif T_0i > 273.15 and T_0i <= 5 + 273.15:
        m_dot = Vol_eff[3]*Vd*rho
        w = (h_0es - h_0i)/Isen_eff[3]
      
    else:
        m_dot = Vol_eff[4]*Vd*rho        
        w = (h_0es - h_0i)/Isen_eff[4]

   
    return m_dot, w, h_0i

