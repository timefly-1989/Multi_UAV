import numpy as np
U_tip = 120
P_0 = 0.2
P_i = 0.2
v_0 = 4.03
d_0 = 0.6
p = 1.225
s = 0.05
A = 0.503
def energy_consumption(V):
    return (1+3*V**2/(U_tip**2))*P_0+P_i*np.sqrt(np.sqrt(1+V**4/(4*v_0**4))-V**2/(2*v_0**2))+d_0*p*s*A*V**3

def energy_distance_estimation(v, t, e):
    return v*t/e
