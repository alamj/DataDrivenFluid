# DataDrivenFluid
Illustration of Proper Orthogonal Decomposition (POD) and (DMD)

Data    : flow past a cylinder at Reynolds number, Re = 400. There are two data files.

          cyldata6h.csv contains  600 snapshots of the vorticity at a time step of dt = 0.125
          the first data (~2.2GB) can be downloaded from this link:
          https://www.dropbox.com/s/7r5djn0vykkhgtk/cyldata6h.csv?dl=0
          
          cyldata1k.csv contains 1000 snapshots of the vorticity at a time step of dt = 0.125
            
          The grid consists of 768 x 192 points in the streamwise and spanwise directions. 
          The field is stored as an 1D vector of size 768*192 = 147456
          
Example : (xburgers_pod.py): 
         Solve 1D viscous Burgers equation using using spectral method.
         
         First part of the code uses FFT from numpy to discretize Burgers equation using periodic boundary condition.
         Then, it uses scipy.odeint to integrate discretized Burgers equation.
         Then, it uses matplotlib.pyplot to visualize space time solution and time snapshots.
         
         This code can be modified to solve a similar 1D PDE, such as wave equation or heat equation. 
         
         Second part of the code uses SVD from mumpy.linalg to compute POD modes. 
         It illustrates low-dimensional approximation of the solution.
         Then, it shows first 4 POD modes.
