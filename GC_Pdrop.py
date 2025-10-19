import CoolProp.CoolProp as CP
import numpy as np
import matplotlib.pyplot as plt
from GC_correlations import GC_CO2, GC_water
from Functions import UA, Friction
import pandas as pd

def gas_cooler2(N_gc, T_CO2_in, P_CO2_in, T_water_in, P_water, L_gc, m_dot_co2, m_dot_w, k_wall, 
               di_inner, do_inner, di_outer, N_segments, plot = True):
    # Initialize variables
    dx = L_gc/(N_segments+1)

    m_dot_CO2 = m_dot_co2/N_gc
    m_dot_water = m_dot_w/N_gc

    P_co2 = np.zeros(N_segments+1)
    P_w = np.ones(N_segments+1)*P_water
    P_co2[0] = P_CO2_in

   # Initialize temperature arrays
    T_hot = np.linspace(T_CO2_in, T_water_in +10, N_segments+1)            # Hot fluid from left to right
    T_cold = np.linspace(T_CO2_in - 10, T_water_in, N_segments+1)          # Cold fluid from right to left
    T_cold[-1] = T_water_in
    T_w = np.zeros(N_segments+1)
    T_w[0] = T_hot[0] - 20    

    # Solve the energy balance equations
    max_it = 100

    Q_arr = np.zeros(N_segments)
    UA_arr = np.zeros(N_segments)
    Cp_arr = np.zeros(N_segments)

    for iteration in range(max_it):
        Th_old = T_hot.copy()
        Tc_old = T_cold.copy()
        for i in range(N_segments):
            
            if P_co2[i] < 0 or P_w[i] < 0:
                print('Gas Cooler is too small or mass flow too high')
                break

            ## Pressure Loss in CO2 Pipe
            mu_i_c = CP.PropsSI('viscosity', 'T', T_hot[i],'P', P_co2[i], 'CO2')
            rho_c = CP.PropsSI('D', 'T', T_hot[i],'P', P_co2[i], 'CO2')
            A_co2 = np.pi*(di_inner/2)**2
            
            V_c = m_dot_CO2/(rho_c*A_co2)
            
            Re_bulk_co2 = m_dot_CO2*di_inner/(A_co2*mu_i_c)
            f_co2 = Friction(Re_bulk_co2)
            
            dP_co2 = f_co2*(dx/di_inner)*rho_c*(V_c**2)/2
            
            P_co2[i+1] = P_co2[i] - dP_co2

            ## Pressure Loss in Water side
            mu_i_w = CP.PropsSI('viscosity', 'T', T_cold[i],'P', P_w[i], 'water')
            rho_w = CP.PropsSI('D', 'T', T_cold[i], 'P', P_w[i], 'water')
            A_w = np.pi*(di_outer**2 - do_inner**2)/4

            V_w = m_dot_water/(rho_w*A_w)

            dh = di_outer - do_inner
            Re_bulk_w = m_dot_water*dh/(A_w*mu_i_w)
            f_w = Friction(Re_bulk_w)

            dP_w = f_w*(dx/dh)*rho_w*(V_w**2)/2
            
            P_w[i] = P_w[i+1] - dP_w

            ## Heat Transfer
            h_co2, Cp_co2 = GC_CO2(m_dot_CO2, T_hot[i], P_co2[i], di_inner, dx)
            h_water, Cp_water = GC_water(m_dot_water, T_cold[i], P_w[i], di_outer, do_inner, dx)

            Cp_arr[i] = Cp_co2

            UA_gc = UA(h_co2, do_inner, di_inner, k_wall, dx, h_water)
            UA_arr[i] = UA_gc
            dT1 = T_hot[i] - T_cold[i]
            dT2 = T_hot[i+1] - T_cold[i+1]

            if abs(dT1 - dT2) > 1e-6 and dT1 > 0 and dT2 > 0:
                dT_lm = (dT1 - dT2)/np.log(dT1 / dT2)
                #print(1)
            else:
                dT_lm = dT1
                #print(0)

            Q = UA_gc*dT_lm
            Q_arr[i] = Q
            

            T_hot[i+1] = T_hot[i] - Q/(m_dot_CO2*Cp_co2)
            T_cold[i] = T_cold[i+1] + Q/(m_dot_water*Cp_water)

        T_cold[-1] = T_water_in
        T_hot[0] = T_CO2_in
        P_co2[0] = P_CO2_in
        P_w[-1] = P_water

        c_err = (np.abs(Tc_old - T_cold))
        h_err = (np.abs(Th_old - T_hot))
        err1 = np.max([c_err, h_err])
        #print(Q_tot/1e3)
        h_wi = CP.PropsSI('H', 'T', T_water_in, 'P', P_water, 'Water')
        h_wo = CP.PropsSI('H', 'T', T_cold[0], 'P', P_water, 'Water')
        Q_water = m_dot_water*(h_wo-h_wi)

        h_ci = CP.PropsSI('H', 'T', T_CO2_in, 'P', P_CO2_in, 'CO2')
        h_co = CP.PropsSI('H', 'T', T_hot[-1], 'P', P_CO2_in, 'CO2')
        Q_co2 = m_dot_CO2*(h_ci - h_co)

        #print(Q_co2, Q_water)
        Q_tot = np.sum(Q_arr)
    
        err2 = np.abs(Q_water - Q_co2)/Q_tot
        #print("It: ", iteration, "h_err", max(h_err), "c_err", max(c_err))

        UA_tot = np.sum(UA_arr)

        err = max(err1, err2)
        if max(c_err) <= 1e-3 and max(h_err) <= 1e-3:
            #print("Converged!")
            break

    else:
        print("Did not converge in max iterations.")
    
    T_CO2_out = T_hot[-1]
    h_CO2_out = CP.PropsSI('H', 'T', T_CO2_out, 'P', P_CO2_in, 'CO2')

    T_water_out = T_cold[0]
    Q_total = Q_tot*N_gc
    print(UA_tot)
    if plot == True:
        x = np.linspace(0, L_gc, N_segments+1)
        xx = x/L_gc
        x2 = np.linspace(0, L_gc, N_segments)
        xx2 = x2/L_gc
        fig, axs = plt.subplots(4, 1, figsize=(8, 10))

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

        axs[3].plot(xx2, Cp_arr/1000, label='Specific Heat per segment [kJ/kgK]', color='red')
        axs[3].set_xlabel("Non-dimensional Length [-]")
        axs[3].set_ylabel("Specific Heat [kJ/kgK]")
        axs[3].grid(True)
        axs[3].legend()

        plt.tight_layout()
        plt.show()

    return T_CO2_out, h_CO2_out, T_water_out, Q_total, P_co2[-1], P_w[0]


def gas_cooler22(N_gc, T_CO2_in, P_CO2_in, T_water_in, P_water, L_gc, m_dot_co2, m_dot_w, k_wall, 
               di_inner, do_inner, di_outer, N_segments, plot=True, save_excel=True, excel_filename="T_hot_profile.xlsx"):
    # Initialize variables
    dx = L_gc/(N_segments+1)

    m_dot_CO2 = m_dot_co2/N_gc
    m_dot_water = m_dot_w/N_gc

    P_co2 = np.zeros(N_segments+1)
    P_w = np.ones(N_segments+1)*P_water
    P_co2[0] = P_CO2_in

    # Initialize temperature arrays
    T_hot = np.linspace(T_CO2_in, T_water_in +10, N_segments+1)            # Hot fluid from left to right
    T_cold = np.linspace(T_CO2_in - 10, T_water_in, N_segments+1)          # Cold fluid from right to left
    T_cold[-1] = T_water_in
    T_w = np.zeros(N_segments+1)
    T_w[0] = T_hot[0] - 20    

    # Solve the energy balance equations
    max_it = 100

    Q_arr = np.zeros(N_segments)
    UA_arr = np.zeros(N_segments)
    Cp_arr = np.zeros(N_segments)

    for iteration in range(max_it):
        Th_old = T_hot.copy()
        Tc_old = T_cold.copy()
        for i in range(N_segments):
            
            if P_co2[i] < 0 or P_w[i] < 0:
                print('Gas Cooler is too small or mass flow too high')
                break

            ## Pressure Loss in CO2 Pipe
            mu_i_c = CP.PropsSI('viscosity', 'T', T_hot[i],'P', P_co2[i], 'CO2')
            rho_c = CP.PropsSI('D', 'T', T_hot[i],'P', P_co2[i], 'CO2')
            A_co2 = np.pi*(di_inner/2)**2
            
            V_c = m_dot_CO2/(rho_c*A_co2)
            
            Re_bulk_co2 = m_dot_CO2*di_inner/(A_co2*mu_i_c)
            f_co2 = Friction(Re_bulk_co2)
            
            dP_co2 = f_co2*(dx/di_inner)*rho_c*(V_c**2)/2
            
            P_co2[i+1] = P_co2[i] - dP_co2

            ## Pressure Loss in Water side
            mu_i_w = CP.PropsSI('viscosity', 'T', T_cold[i],'P', P_w[i], 'water')
            rho_w = CP.PropsSI('D', 'T', T_cold[i], 'P', P_w[i], 'water')
            A_w = np.pi*(di_outer**2 - do_inner**2)/4

            V_w = m_dot_water/(rho_w*A_w)

            dh = di_outer - do_inner
            Re_bulk_w = m_dot_water*dh/(A_w*mu_i_w)
            f_w = Friction(Re_bulk_w)

            dP_w = f_w*(dx/dh)*rho_w*(V_w**2)/2
            
            P_w[i] = P_w[i+1] - dP_w

            ## Heat Transfer
            h_co2, Cp_co2 = GC_CO2(m_dot_CO2, T_hot[i], P_co2[i], di_inner, dx)
            h_water, Cp_water = GC_water(m_dot_water, T_cold[i], P_w[i], di_outer, do_inner, dx)

            Cp_arr[i] = Cp_co2

            UA_gc = UA(h_co2, do_inner, di_inner, k_wall, dx, h_water)
            UA_arr[i] = UA_gc
            dT1 = T_hot[i] - T_cold[i]
            dT2 = T_hot[i+1] - T_cold[i+1]

            if abs(dT1 - dT2) > 1e-6 and dT1 > 0 and dT2 > 0:
                dT_lm = (dT1 - dT2)/np.log(dT1 / dT2)
            else:
                dT_lm = dT1

            Q = UA_gc*dT_lm
            Q_arr[i] = Q
            

            T_hot[i+1] = T_hot[i] - Q/(m_dot_CO2*Cp_co2)
            T_cold[i] = T_cold[i+1] + Q/(m_dot_water*Cp_water)

        T_cold[-1] = T_water_in
        T_hot[0] = T_CO2_in
        P_co2[0] = P_CO2_in
        P_w[-1] = P_water

        c_err = (np.abs(Tc_old - T_cold))
        h_err = (np.abs(Th_old - T_hot))
        err1 = np.max([c_err, h_err])

        h_wi = CP.PropsSI('H', 'T', T_water_in, 'P', P_water, 'Water')
        h_wo = CP.PropsSI('H', 'T', T_cold[0], 'P', P_water, 'Water')
        Q_water = m_dot_water*(h_wo-h_wi)

        h_ci = CP.PropsSI('H', 'T', T_CO2_in, 'P', P_CO2_in, 'CO2')
        h_co = CP.PropsSI('H', 'T', T_hot[-1], 'P', P_CO2_in, 'CO2')
        Q_co2 = m_dot_CO2*(h_ci - h_co)

        Q_tot = np.sum(Q_arr)
    
        err2 = np.abs(Q_water - Q_co2)/Q_tot

        UA_tot = np.sum(UA_arr)

        err = max(err1, err2)
        if max(c_err) <= 1e-3 and max(h_err) <= 1e-3:
            break

    else:
        print("Did not converge in max iterations.")
    
    T_CO2_out = T_hot[-1]
    h_CO2_out = CP.PropsSI('H', 'T', T_CO2_out, 'P', P_CO2_in, 'CO2')

    T_water_out = T_cold[0]
    Q_total = Q_tot*N_gc
    print(UA_tot)

    #  SAVE TO EXCEL
    if save_excel:
        x = np.linspace(0, L_gc, N_segments+1)
        df = pd.DataFrame({
            "Position [m]": x,
            "T_hot [K]": T_hot,
            "T_hot [°C]": T_hot - 273.15
        })
        df.to_excel(excel_filename, index=False)
        print(f"Temperature profile saved to {excel_filename}")

    #  PLOTTING 
    if plot:
        x = np.linspace(0, L_gc, N_segments+1)
        xx = x/L_gc
        x2 = np.linspace(0, L_gc, N_segments)
        xx2 = x2/L_gc
        fig, axs = plt.subplots(4, 1, figsize=(8, 10))

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

        axs[3].plot(xx2, Cp_arr/1000, label='Specific Heat per segment [kJ/kgK]', color='red')
        axs[3].set_xlabel("Non-dimensional Length [-]")
        axs[3].set_ylabel("Specific Heat [kJ/kgK]")
        axs[3].grid(True)
        axs[3].legend()

        plt.tight_layout()
        plt.show()

    return T_CO2_out, h_CO2_out, T_water_out, Q_total, P_co2[-1], P_w[0]


T_CO2_out, h_CO2_out, T_water_out, Q_total, P_co2_out, P_water_out = gas_cooler22(N_gc=15,
    T_CO2_in = 102.5601896 + 273.15,        
    P_CO2_in = 8588.003131e3,           
    T_water_in = 15 + 273.15,     
    P_water = 2e6,            
    L_gc = 13,               
    m_dot_co2 = 1.35378432,        
    m_dot_w = 2,        
    k_wall = 401,              
    di_inner = 8.8e-3,
    do_inner = 10e-3,
    di_outer = 12.6e-3,
    N_segments = 65,
    plot=True
)

print(f"T_CO2_out     = {T_CO2_out - 273.15:.2f} °C")
print(f"h_CO2_out     = {h_CO2_out / 1000:.2f} kJ/kg")
print(f"T_water_out   = {T_water_out - 273.15:.2f} °C")
print(f"Q_total       = {Q_total / 1000:.2f} kW")
print('P_co2_out: ', P_co2_out/1e6, 'MPa')
print('P_water_out: ', P_water_out/1e6, 'MPa')

