# DataDrivenFluid
Illustration of Proper Orthogonal Decomposition (POD) and (DMD)

Data    : flow past a cylinder at Reynolds number, Re = 200. There are two data files.
          The data can be downloaded from the given link.

          cyldata6h.csv contains  600 snapshots of the vorticity at a time step of dt = 0.125
          the first data (~2.2GB) can be downloaded from this link:
          https://www.dropbox.com/s/7r5djn0vykkhgtk/cyldata6h.csv?dl=0
          
          cyldata1k.csv contains 1000 snapshots of the vorticity at a time step of dt = 0.125
          the second data (~3.6GB) can be downloaed from the following link:
          https://www.dropbox.com/s/nmb9yqxyoswqgkz/cyldata1k.csv?dl=0
            
          The grid consists of 768 x 192 points in the streamwise and spanwise directions. 
          The field is stored as an 1D vector of size 768*192 = 147456
          
          A smaller data (~121MB), cyl_data_clean.csv, contains the vorticity at a time step of dt = 0.02
          https://www.dropbox.com/s/gjyimyf2eo1aajd/cyl_data_clean.csv?dl=0
          This field is stored as 151 of 1D vectors of size 199*499.
          See the notebook to read this data. 
          
          There are two example codes (python script and nobebook) showing how to read and visualize the data. 
          
Example : (xburgers_pod.py): 
         Solve 1D viscous Burgers equation using using spectral method.
         
         First part of the code uses FFT from numpy to discretize Burgers equation using periodic boundary condition.
         Then, it uses scipy.odeint to integrate discretized Burgers equation.
         Then, it uses matplotlib.pyplot to visualize space time solution and time snapshots.
         
         This code can be modified to solve a similar 1D PDE, such as wave equation or heat equation. 
         
         Second part of the code uses SVD from mumpy.linalg to compute POD modes. 
         It illustrates low-dimensional approximation of the solution.
         Then, it shows first 4 POD modes.

         (pod_dmd_lab1.ipynb)
         This notebook shows how to compute POD modes using SVD

         (pod_admm_lab2.ipynb)
         This notebook provides an implementation of ADMM. It provides a soft-threshold scheme and shows code to add salt and paper noise.

         To use this notebook, download one of the cylinder flow data posted in dropbox link, and 
         replace "code/spiral_flow_clean128.csv" according to the downloaded data file name, and
         replace values of nx and ny according to the downloaded data. 

         (pod_dmd_lab3.ipynb)
         This notebook provides an implementation of DMD method. 
