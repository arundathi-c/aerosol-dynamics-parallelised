from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
from math import pi
from functions import monte_carlo, initial_coeff_calc, coeff_recalc
import time
import sys

stdoutOrigin = sys.stdout
sys.stdout = open("log.txt", "w")

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

if rank == 0:
    start = time.time()

runs = 16  # number of times the simulation is run to get an accurate representation of the function

N = 1000  # number of particles
T = 1800.0  # The temperature , it remains constant throughout the process
n_0 = 10 ** 18  # particle number density
V = N / n_0
rho = 1800.0
dia = 1.5 * 10 ** (-9)
k_B = 1.380649 * 10 ** (-23)

initial_mass = rho * 4 * pi * dia ** 3 / (3.0 * 8.0)
x = initial_mass * np.ones(N)  # this array contains the values of mass of the particles
coeff = ((6 / pi) ** (2 / 3)) * (pi * k_B * T / (2 * rho)) ** ((1 / 2) / V)
C_ij = (coeff * (1 / initial_mass + 1 / initial_mass) ** (1 / 2) * (initial_mass ** (1 / 3) + initial_mass ** (1 / 3)) ** 2) * np.ones(int(N * (N - 1) / 2))
C_i = np.zeros(N)
C_0 = 0.0
(C_i, C_0) = initial_coeff_calc(C_ij, N)

t_stop = 1.0 * 10 ** (-3)  # total time in seconds for which the simulation takes place in milli second
dt = 0.01  # the time interval at which the values are calculated
t_sample = np.zeros(100)  # set of time to find the particle size distribution
t_sample = np.linspace(0, t_stop, num=100)

runs = 16

mass = np.zeros((100))
t = 0.0  # initialising the time variable
iter = 0
for a in range(int(runs / size)):
    while t < t_stop:
        (tau, i, j) = monte_carlo(x, C_0, C_i, C_ij)
        x_new = x[i] + x[j]
        x = np.delete(x, (i, j))
        x = np.append(x, x_new)
        (C_0, C_i, C_ij) = coeff_recalc(x)
        t = t + tau
        if t > t_sample[iter]:
            mass[iter] = np.average(x)
            iter = iter + 1

        if len(x) <= 1:
            break

if rank == 0:
    mass_gather = np.zeros(size * 100)
    comm.Gather(mass, mass_gather, root=0)
    end = time.time()
    mass_avg = np.zeros(100)
    for i in range(size):
        mass_avg = mass_avg + mass_gather[i * 100: (i + 1) * 100]
    mass_avg = mass_avg / (size * 1.0)
    print(mass_avg)
    print("Time taken = ", end - start)