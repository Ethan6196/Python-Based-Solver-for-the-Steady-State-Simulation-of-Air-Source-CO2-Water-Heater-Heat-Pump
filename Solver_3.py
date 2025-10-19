import numpy as np
import CoolProp.CoolProp as CP
from scipy.optimize import root_scalar,fsolve, least_squares
from Gas_Cooler import gas_cooler2 as GC
from Compressor2 import Compressor as comp
from Expansion_valve1 import solve_expansion_valve_outlet_pressure as EV
from Evaporator import Dry_Evaporator2 as evap

def P_dis_solver(tol, T_1, P_suction, P_dis_init, Bore, Stroke, N_comp, n_cyl, N_gc, T_wi, P_water, L_gc, m_dot_w, 
                 k_wall, di_inner, do_inner, di_outer, N_seg, z, D_throat ):

    def Pressure(P_discharge):

        P_discharge = float(P_discharge) # Cast variable type
            
        m_c, w_c, h_1 = comp(T_1, P_suction, P_discharge,
                                     Bore, Stroke, N_comp, n_cyl)
        h_2 = h_1 + w_c

        T_2 = CP.PropsSI('T','P',P_discharge,'H',h_2,'CO2')
        T_3, h_3, T_wo_loc, Q_gc = GC(N_gc,
                    T_2, P_discharge, T_wi, P_water, L_gc,
                    m_c, m_dot_w, k_wall, di_inner, do_inner,
                    di_outer, N_seg, False
                )
            
        _, P_3 = EV(m_c, z, T_3, P_discharge, P_suction, D_throat)

        return (P_suction - P_3)

    res_dis = fsolve(Pressure, x0=P_dis_init)
    P_dis = res_dis[0]
    
    return P_dis 


def solver(N_gc, tol, max_it, SH,
           P_suc_init, P_dis_init,
           Bore, Stroke, n_cyl, N_comp,
           L_gc, di_inner, do_inner, di_outer, k_wall,
           P_water, T_wi, N_seg, m_dot_w,
           z, D_throat, v_air, T_ai,
           pt, pl, Nt, Nr, P_air,
           Di, Do, L_evap, t_fin, t_spacing):

    # Outputs    
    m_dot = np.zeros(4)
    P_0 = np.zeros(4)
    T_0 = np.zeros(4)
    h_0 = np.zeros(4)
    q = np.zeros(4)
    T_wo = T_ao = w = 0

    def Energy(P_suction):
        nonlocal m_dot, P_0, T_0, h_0, q, T_wo, T_ao, w # Save data Globally
        
        P_suction = float(P_suction)
        if P_suction >= CP.PropsSI('Pcrit', 'CO2') or P_suction < 1e6:
            return P_suc_init
        
        T_sat = CP.PropsSI('T','P',P_suction,'Q',1,'CO2')
        T_1 = T_sat + SH

        P_dis = P_dis_solver(tol, T_1, P_suction, P_dis_init, Bore, Stroke, N_comp, n_cyl, N_gc,
                             T_wi, P_water, L_gc, m_dot_w, k_wall, di_inner, do_inner, di_outer, N_seg, z, D_throat)
        
        # now recompute full cycle (also guarded)
        try:

            m_c, w_c, h_1 = comp(T_1, P_suction, P_dis,
                                 Bore, Stroke, N_comp, n_cyl)
            
            h_2 = h_1 + w_c

            T_2 = CP.PropsSI('T', 'P', P_dis, 'H', h_2, 'CO2')
            
            T_3, h_3, T_wo_loc, Q_gc = GC(N_gc,
                T_2, P_dis, T_wi, P_water, L_gc,
                m_c, m_dot_w, k_wall, di_inner, do_inner,
                di_outer, N_seg, False
            )

            _, P_3 = EV(m_c, z, T_3, P_dis, P_suction, D_throat)
            
            Q_evaporator = Q_gc - w_c*m_c
           
            h_4 = h_3
            T_4 = CP.PropsSI('T', 'P', P_suction, 'H', h_4, 'CO2')
            T_evap, h_evap, T_ao_loc, Q_evap, L_2p = evap(
                T_4, h_4, P_suction, T_ai, P_air, L_evap,
                m_c, v_air, k_wall,
                Di, Do, Nr, Nt, pt, pl, t_fin, t_spacing
            ) 

        except ValueError:
            # if anything goes off the rails, push outer solver
            return 2e6
        
        m_dot[:] = m_c
        P_0[:]   = [P_suction, P_dis, P_dis, P_suction]
        h_0[:]   = [h_1, h_2, h_3, h_3]
        T_0[:]   = [T_1, T_2, T_3, T_4]
        q[:]     = [Q_evap, 0, Q_gc, 0] / m_c
        T_wo, T_ao, w = T_wo_loc, T_ao_loc, w_c
        print('2 phase Length', L_2p)
        print('SH length ', L_evap - L_2p)

        # residual for superheat target
        return (Q_evap - Q_evaporator)/np.abs(Q_evaporator)


    res = fsolve(Energy, x0=P_suc_init)
    

    return m_dot, P_0, T_0, q, w, h_0, T_wo, T_ao
