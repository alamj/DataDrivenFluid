import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Written by Dr Jahrul Alam (alamj@mun.ca)
# as part of teaching data-driven fluid dynamics
#
# First, Burgers equation is solved by spectral method
# using numpy FFT

# Then, POD modes are computed
# using numpy SVD
#

Tmax = 1.0
M = 600
dt = Tmax/M
t = np.linspace(0,Tmax,M)

# Spatial domain for Burgers equation

nu = 0.0125/8    # viscosity or diffusion constant
L = 2          # Length of domain
N = 512        # Number of grid points
dx = L/N       # spatial step size


# To implement periodic boundary condtions in spectral method
# we can remove the last point, taken care of by FFT
# Note, additional care must be taken
#       for other types of boundary conditions

x = np.arange(0,L,dx) # spatial grid
kx = 2*np.pi*np.fft.fftfreq(N, d=dx)  # wavenumber grid

# Initial condition
u0 = np.sin(2*np.pi*x/L) # + 0.5*np.sin(np.pi*x)

def BurgersPDE(u,t,kx,nu):
    uhat = np.fft.fft(u)
    d_uhat = (1j)*kx*uhat
    dd_uhat = -np.power(kx,2)*uhat
    d_u = np.fft.ifft(d_uhat)
    d2u = np.fft.ifft(dd_uhat)
    dudt = -u * d_u + nu*d2u
    return dudt.real 

u = odeint(BurgersPDE,u0,t,args=(kx,nu))

sol = u.T

# show a contour plot of the solution
plt.figure(figsize=(12,5))
plt.subplot(1,3,1)
plt.pcolormesh(t,x,sol,shading='auto')

# show few snapshots of the solution
plt.subplot(1,3,2)
for j in range(0,sol.shape[1],100):
    plt.plot(x,sol[:,j])



# U is m_train number of snapshots
#   for POD training

m_train = 200
U = sol[:,:m_train]

# Perform SVD
Phi,Sig,Psi = np.linalg.svd(U, full_matrices = False)

# Project U onto low-dimensional POD modes

rank = 16              # dimension of POD approximation space
Phi = Phi[:, : rank]
Psi = Psi[: rank, : ]
Sig = Sig[: rank]

U_rank = np.linalg.multi_dot([Phi*Sig, Psi])

# show a contour plot the projected data

t_rank = t[:m_train]          # time for projected region
plt.subplot(1,3,3)
plt.pcolormesh(t_rank, x, U_rank,shading='auto')


plt.figure(2)
plt.subplot(2,2,1)
plt.plot(x,Phi[:,0])

plt.subplot(2,2,2)
plt.plot(x,Phi[:,1])

plt.subplot(2,2,3)
plt.plot(x,Phi[:,2])

plt.subplot(2,2,4)
plt.plot(x,Phi[:,3])

plt.show()


