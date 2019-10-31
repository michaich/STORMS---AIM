import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm


t = np.loadtxt("times.dat")
plt.plot(t[:,0],t[:,1],'r-o',lw=2)
plt.xlabel(" Resolution (pixels per side)")
plt.ylabel(" Time (sec) ")
plt.grid()
#plt.show()
plt.savefig('times.png')