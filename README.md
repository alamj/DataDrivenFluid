# DataDrivenFluid
Illustration of Proper Orthogonal Decomposition (POD) and (DMD)

Example 1 (xburgers_pod.py): 
         Solve 1D viscous Burgers equation using using spectral method.
         
         First part of the code uses FFT from numpy to discretize Burgers equation using periodic boundary condition.
         Then, it uses scipy.odeint to integrate discretized Burgers equation.
         Then, it uses matplotlib.pyplot to visualize space time solution and time snapshots.
         
         This code can be modified to solve a similar 1D PDE, such as wave equation or heat equation. 
         
         Second part of the code uses SVD from mumpy.linalg to compute POD modes. 
         It illustrates low-dimensional approximation of the solution.
         Then, it shows first 4 POD modes.
