import numpy as np

def Friction(Re):
    f = np.power((1.82*np.log10(Re) - 1.64),-2)
    return f

def Nusselt(f,Re,Pr):
    Nu = ((f/8)*(Re-1000)*Pr)/(1.07 + 12.7*np.sqrt(f/8)*(np.power(Pr,2/3) - 1))
    return Nu

def UA(h_i, Do, Di, k_pipe, L, h_o):
    Rth = 1/(h_i*np.pi*Di*L) + np.log(Do/Di)/(2*np.pi*k_pipe*L) + 1/(h_o*np.pi*Do*L)
    UA = 1/Rth
    return UA

import numpy as np


def Nu_air(Re, Pr, Prs, pt, pl, Nr):

    # Correlation applicability range
    if Re>0 and Re<=500:
        Nu = 1.04*(Re**0.4)*(Pr**0.36)*(Pr/Prs)**0.25
    elif Re>500 and Re<=1000:
        Nu = 0.71*(Re**0.5)*(Pr**0.36)*(Pr/Prs)**0.25
    elif Re>1000 and Re<2e5:
        Nu = 0.35*((pt/pl)**0.2)*(Re**0.6)*(Pr**0.36)*(Pr/Prs)**0.25
    else:
        Nu = 0.031*((pt/pl)**0.2)*(Re**0.8)*(Pr**0.36)*(Pr/Prs)**0.25
        
        # Correction factor if Rows<16
        if Nr ==1:
            Nu = 0.7*Nu
        elif Nr==2:
            Nu = 0.76*Nu
        elif Nr==3:
            Nu = 0.84*Nu
        elif Nr==4:
            Nu = 0.89*Nu
        elif Nr==5:
            Nu = 0.93*Nu
        elif Nr==6 or Nu==7:
            Nu = 0.96*Nu
        else:
            Nu=Nu

    return Nu
