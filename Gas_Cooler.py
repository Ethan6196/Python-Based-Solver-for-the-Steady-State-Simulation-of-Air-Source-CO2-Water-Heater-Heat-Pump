import CoolProp.CoolProp as CP
import numpy as np
from Functions import UA
from GC_correlations import GC_CO2, GC_water
import matplotlib.pyplot as plt
import pandas as pd

# --- THIS IS THE ONE THAT WORKS ---
def gas_cooler2(N_gc, T_CO2_in, P_CO2_in, T_water_in, P_water, L_gc, m_CO2, m_water, k_wall, 
               di_inner, do_inner, di_outer, N_segments, plot = True):
    # Initialize variables
    dx = L_gc/(N_segments)
    
    m_dot_CO2 = m_CO2/N_gc
    m_dot_water = m_water/N_gc

   # Initialize temperature arrays
    T_hot = np.linspace(T_CO2_in, T_water_in +10, N_segments+1)    # Hot fluid from left to right
    T_cold = np.linspace(T_CO2_in - 10, T_water_in, N_segments+1)  # Cold fluid from right to left
    T_cold[-1] = T_water_in 

    # Solve the energy balance equations
    max_it = 200

    Q_arr = np.zeros(N_segments)
    UA_arr = np.zeros(N_segments)

    # Main iteration loop
    for iteration in range(max_it):

        Tc_old = T_cold.copy()
        Th_old = T_hot.copy()

        # Temperature profile loop

        for i in range(N_segments):
          
            h_co2, Cp_co2 = GC_CO2(m_dot_CO2, T_hot[i], P_CO2_in, di_inner, dx)
            h_water, Cp_water = GC_water(m_dot_water, T_cold[i], P_water, di_outer, do_inner, dx)

            UA_gc = UA(h_co2, do_inner, di_inner, k_wall, dx, h_water)
            UA_arr[i] = UA_gc
            dT1 = T_hot[i] - T_cold[i]
            dT2 = Th_old[i+1] - Tc_old[i+1]

            # Avoid divison by 0
            if abs(dT1 - dT2) > 1e-6 and dT1 > 0 and dT2 > 0:
                dT_lm = (dT1 - dT2)/np.log(dT1 / dT2)
               
            else:
                dT_lm = (dT1 + dT2)/2

            Q = UA_gc*dT_lm
            Q_arr[i] = Q

            # Update Temperature Profile
            T_hot[i+1] = T_hot[i] - Q/(m_dot_CO2*Cp_co2)
            T_cold[i] = Tc_old[i+1] + Q/(m_dot_water*Cp_water)


        T_cold[-1] = T_water_in
        T_hot[0] = T_CO2_in
       
        Q_tot = np.sum(Q_arr)
        
        # Define convergence Criteria
        c_err = (np.abs(Tc_old - T_cold))/np.abs(Tc_old)
        h_err = (np.abs(Th_old - T_hot))/np.abs(Th_old)
        err1 = np.max([c_err, h_err])

        iteration += 1

        if err1 <= 1e-4:
            
            break
        
    T_CO2_out = T_hot[-1]
    h_CO2_out = CP.PropsSI('H', 'T', T_CO2_out, 'P', P_CO2_in, 'CO2')
    UA_tot = sum(UA_arr)
    T_water_out = T_cold[0]
    Q_total = Q_tot*N_gc
    '''
    if plot == True:
        x = np.linspace(0, L_gc, N_segments+1)
        xx = x/L_gc
        x2 = np.linspace(0, L_gc, N_segments)
        xx2 = x2/L_gc
        fig, axs = plt.subplots(3, 1, figsize=(8, 10))

        axs[0].plot(xx, T_hot-273.15, label="Hot fluid [°C]", color='red')
        axs[0].plot(xx, T_cold-273.15, label="Cold fluid [°C]", color='blue')
        axs[0].set_ylabel("Temperature [°C]")
        axs[0].legend()
        axs[0].grid(True)
        axs[0].set_title("Temperature Profile (non-dimensional length)")

        axs[1].plot(xx2, Q_arr, label='Segment Heat Transfer [W]', color='green')
        axs[1].set_ylabel("Q per segment [W]")
        axs[1].grid(True)
        axs[1].legend()

        axs[2].plot(xx2, UA_arr, label='UA per segment [W/K]', color='purple')
        axs[2].set_xlabel("Non-dimensional Length [-]")
        axs[2].set_ylabel("UA per segment [W/K]")
        axs[2].grid(True)
        axs[2].legend()

        plt.tight_layout()
        plt.show()
        '''    

    return T_CO2_out, h_CO2_out, T_water_out, Q_total
