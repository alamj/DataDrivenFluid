import numpy as np
import matplotlib.pyplot as plt

# Author Dr Jahrul Alam (alamj@mun.ca)
# An example code that reads the cylinder flow data
#
# file name of the CSV format data

data_file = "cyldata6h.csv"
#data_file = "cyldata6h.csv"

# load the data
U = np.loadtxt(data_file,delimiter=",")

# Each column of U is a time snapshot of the vorticity of a 2D flow
# The grid of the flow consists of 768 points along the x-direction
# and 192 points along the y-direction
# the domain is [0, 32] x [0 8]

nx = 768
ny = 192
x = np.linspace(0,32,nx)
y = np.linspace(0,8,ny)

# let us print the size of U and check if the number of rows equals nx*ny

print("dim = ", U.shape[0], " x ", U.shape[1], ", nx*ny = ", nx*ny)

# let us visualize two snapshots of the vorticity field

plt.figure(figsize=(10,1))
plt.subplot(1,2,1)
plt.pcolormesh(x,y,np.reshape(U[:,10],(ny,nx)), cmap="RdBu_r",vmin=-1,vmax=1)
plt.subplot(1,2,2)
plt.pcolormesh(x,y,np.reshape(U[:,200],(ny,nx)), cmap="RdBu_r",vmin=-1,vmax=1)

plt.show()



