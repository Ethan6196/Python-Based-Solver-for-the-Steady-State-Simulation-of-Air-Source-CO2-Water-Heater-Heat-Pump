import numpy as np
import CoolProp.CoolProp as CP
import matplotlib.pyplot as plt


### Semi-hermetic Compressor Geometric Parameters ###

Bore = 48 #mm
Stroke = 28.5 #mm
n_cyl = 4 # Number of Cylinders
N_compressor = 1100 #RPM

### Gas Cooler Geometric Parameters ###
N_gc = 15
L_gc = 20 #m
Di_inner_gc = 8.8e-3 # m --> inner diameter of inner tube
Do_inner_gc = 10e-3 # m --> outer diameter of inner tube
Di_outer_gc = 12.6e-3 # m --> inner diameter of outer tube

### Expansion Valve geometric Parameters ###

D_throat = 16e-3 #m Throat area of EV
z = 1

### Evaporator geometric parameters ###

pt = 23e-3 #m transvers tube pitch
pl = 35e-3 #m longitudinal tube pitch
Di = 18.9e-3 #m inner tube diameter
Do = 22.2e-3 #m outer tube diameter
t_spacing = 0.5e-3 #m fin spacing
t_fin = 0.2e-3 #m fin thickness
N_rows = 15
N_tubes = 10 #Tubes per Row
L_evap = 6 #m

k_wall = 401 #Thermal Conductivity of wall

### Initialise Variables ###

N = 4 # Total number of nodes

### Inlet Conditions ###

T_ai = 0 + 273.15
T_wi = 10 + 273.15
P_a = 101325
P_w = 2e6
m_dotw = 1.4175
v_air = 5
SH = 5
P1 = 2.2e6
P2 = 8.5e6 
n = 45
tol = 1e-6
max_it = 1000

from Solver_3 import solver as solve

m_dot, P_0, T_0, q, w, h_0, T_wo, T_ao = solve(N_gc,tol, max_it, SH, P1, P2, Bore, Stroke, n_cyl,
                                         N_compressor, L_gc, Di_inner_gc, Do_inner_gc, Di_outer_gc, 
                                         k_wall, P_w, T_wi, n, m_dotw, z, D_throat, v_air,
                                         T_ai, pt, pl, N_tubes, N_rows, P_a, Di, Do, L_evap, t_fin, t_spacing)

print('m_dot = ', m_dot, 'kg/s')
print('T_0 = ', T_0 -273.15, '°C')
print('P_0 = ', P_0/1e3, 'kPa')
print('h_0 = ', h_0/1e3, 'kJ/kg')
print('q = ', (q)/1000, 'kJ')
print('w = ', w/1000, 'kJ')
print('W_dot = ',  w*m_dot/1000, 'kW')
print('Q_dot = ',  (q)*m_dot/1000, 'kW')
print('T_wo = ', T_wo - 273.15, 'C')
print('T_ao = ', T_ao - 273.15, 'C')
print('COP = ', q[2]/(w))

# Determine Flow Regime in expansion valve
P_cr = CP.PropsSI('Pcrit', 'CO2')
T_cr = CP.PropsSI("Tcrit", 'CO2')
P_vc = 0.45*P_0[2]

if T_0[2] >= T_cr:
    P_s = P_cr
    
else:
    P_s = CP.PropsSI('P', 'T', T_0[2], 'Q', 1, "CO2")

F_F = 0.96 - 0.28 * np.sqrt(P_s / P_cr)

F_L = np.sqrt((P_0[2] - P_0[3]) / (P_0[2] - P_vc))
dP_cr = (F_L**2) * (P_0[2] - F_F * P_s)

if (P_0[2]-P_0[3])<dP_cr and P_0[3]<P_s:
    print('Two-Phase, Flashing Flow')
elif (P_0[2]-P_0[3])<dP_cr and P_0[3]>P_s:
    print('Two-Phase, Cavitation Flow')
else:
    print('Two-Phase, Choked Flow')
       
input('Press Enter to continue')

labels = ['1', '2', '3', '4', '1']
fluid = 'CO2'
# Create vapor dome (only below critical temperature)
T_crit = CP.PropsSI('Tcrit', fluid)
T_vals = np.linspace(220, T_crit - 0.1, 300)

P_sat = [CP.PropsSI('P', 'T', T, 'Q', 0, fluid) for T in T_vals]
h_liq = [CP.PropsSI('H', 'T', T, 'Q', 0, fluid) for T in T_vals]
h_vap = [CP.PropsSI('H', 'T', T, 'Q', 1, fluid) for T in T_vals]

# Plot
plt.figure(figsize=(10, 6))

# Plot vapor dome
plt.plot(h_liq, np.array(P_sat)/1e6, 'b--', label='Saturated Liquid')
plt.plot(h_vap, np.array(P_sat)/1e6, 'r--', label='Saturated Vapor')

h_cycle = np.array([h_0[0], h_0[1], h_0[2], h_0[3], h_0[0]])
P_cycle = np.array([P_0[0], P_0[1], P_0[2], P_0[3], P_0[0]])
# Plot cycle
plt.plot(h_cycle, np.array(P_cycle)/1e6, 'ko-', label='CO₂ Cycle')

# Annotate cycle points
for i, label in enumerate(labels):
    plt.annotate(label, (h_cycle[i], P_cycle[i]/1e6), textcoords="offset points", xytext=(0,10), ha='center')

# Labels and layout
plt.xlabel('Enthalpy [J/kg]')
plt.ylabel('Pressure [MPa]')
plt.title('Transcritical CO₂ Cycle on P-h Diagram with Vapor Dome')
plt.yscale('log')
plt.grid(True, which='both', linestyle='--')
plt.legend()
plt.tight_layout()
plt.show()

s_liq = [CP.PropsSI('S', 'T', T, 'Q', 0, fluid) for T in T_vals]
s_vap = [CP.PropsSI('S', 'T', T, 'Q', 1, fluid) for T in T_vals]

# Cycle entropy values (use each h and P to get s)
S1 = CP.PropsSI('S', 'T', T_0[0], 'P', P_0[0], 'CO2')
S2 = CP.PropsSI('S', 'T', T_0[1], 'P', P_0[1], 'CO2')
S3 = CP.PropsSI('S', 'T', T_0[2], 'P', P_0[2], 'CO2')
Q = CP.PropsSI('Q', 'H', h_0[3], 'P', P_0[3], 'CO2' )
S4 = CP.PropsSI('S', 'T', T_0[3], 'Q', Q, 'CO2')

s_cycle = np.array([S1, S2, S3, S4, S1])
T_cycle = np.array([T_0[0], T_0[1], T_0[2], T_0[3], T_0[0]])
# Plot T–s diagram
plt.figure(figsize=(10, 6))

# Plot vapor dome
plt.plot(s_liq, T_vals - 273.15, 'b--', label='Saturated Liquid')
plt.plot(s_vap, T_vals - 273.15, 'r--', label='Saturated Vapor')

# Plot CO₂ cycle on T–s
plt.plot(s_cycle, T_cycle - 273.15, 'ko-', label='CO₂ Cycle')

# Annotate points
for i, label in enumerate(labels):
    plt.annotate(label, (s_cycle[i], T_cycle[i] - 273.15), textcoords="offset points", xytext=(0,10), ha='center')

# Labels and formatting
plt.xlabel('Entropy [J/kg·K]')
plt.ylabel('Temperature [°C]')
plt.title('Transcritical CO₂ Cycle on T–s Diagram with Vapor Dome')
plt.grid(True, linestyle='--')
plt.legend()
plt.tight_layout()
plt.show()
