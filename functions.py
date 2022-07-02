import random
from math import pi,log
import numpy as np

def monte_carlo(x,C_0,C_i,C_ij):
    N = len(x)
    r_1 = random.random()
    tau = log(1/r_1)/C_0
    r_2 = random.random()
    temp = 0.0
    temp_i = -1
    while(temp<r_2*C_0):
        temp_i+=1
        temp+= C_i[temp_i]
    i = temp_i

    r_3 = random.random()
    temp = 0.0
    k =int((N-1)*(N-2)/2 - (N-temp_i-2)*(N-temp_i-3)/2 - 2)
    temp_j = temp_i+1
    while(temp<C_i[temp_i]*r_3):
        k+=1
        temp+= C_ij[k]
        temp_j+=1
    j = temp_j
    if i<N and j<N:
        return(tau,i,j)
    else:
        return(monte_carlo(x,C_0,C_i,C_ij))

def initial_coeff_calc(C_ij,N):
    C_i = np.zeros(N)
    C_0 = 0.0
    k = 0
    for i in range(N):
        C_i[i] = np.sum(C_ij[k:k+(N-i-1)])
        k+= N-i
    C_0 = np.sum(C_i)
    return(C_i,C_0)

def coeff_recalc(x):
    N = len(x)
    T = 1800
    k_B = 1.380649 * 10**(-23)
    rho = 1800.0
    n_0 = 10**18  # particle number density
    V = N/n_0
    coeff = ((6/pi)** (2/3) )* (pi*k_B*T/(2*rho))**(1/2)/V
    k = 0
    C_ij = np.zeros(int((N-1)*(N)/2))
    for i in range(N-2):
        for j in range(i+1,N-2):
            C_ij[k] = coeff * (1/x[i] + 1/x[j])**(1/2)  * (x[i]**(1/3) + x[j]**(1/3))**2
            k +=1
    C_i = np.zeros(N)
    k = 0
    for i in range(N):
        C_i[i] = np.sum(C_ij[k:k+(N-i-1)])
        k+= N-i
    C_0 = np.sum(C_i)
    return(C_0,C_i,C_ij)