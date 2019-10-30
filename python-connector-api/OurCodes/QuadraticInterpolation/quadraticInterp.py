import pandas as pd 
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import os
from matplotlib import cm

def ReadArray(name):
	data = np.genfromtxt(name,deletechars="data",skip_header=2,delimiter=";")	
	x     = data[0,1:]
	y     = data[1:,0]
	value = data[1:,1:]
	return x,y,value

	


def QuadInterp(v1,v2,v3):
	#v1 = v(t=-1)
	#v2 = v(t= 0)
	#v3 = v(t= 1)

	T = np.linspace(0,1,3)

	dt = 1.0
	
	l =v3.shape[0]
	
	d2t = (v3 - 2.0 * v2 + v1)/dt/dt 
	#d1t = 0.5* (v3 - v1) / dt
	d1t =  (v3 - v2) / dt


	#dx = np.zeros((l,l))
	#dy = np.zeros((l,l))
	#dx[1:l-2,:]  = 0.5*(v2[2:l-1,:] - v2[0:l-3,:])
	#dx[0  ,:] = v2[1  ,:] - v2[0  ,:]
	#dx[l-1,:] = v2[l-1,:] - v2[l-2,:]
	#dy[:,1:l-2]  = 0.5*(v2[:,2:l-1] - v2[:,0:l-3])
	#dy[:,0  ] = v2[:,1  ] - v2[:,0  ]
	#dy[:,l-1] = v2[:,l-1] - v2[:,l-2]
	

	k = 101
#	for t in T:
	t = 0.5
	print(k,t)
	k += 1
	k = 103
	#sx = -dx[2:l-1,1:l-2]#(-dx[2:l-1,1:l-2]+dx[0:l-3,1:l-2])*0.5
	#sy = -dy[1:l-2,2:l-1]#(-dy[1:l-2,2:l-1]+dy[1:l-2,0:l-3])*0.5 
	v = d2t*(0.5*t*t) + d1t*t + v2 
	plt.imshow(v,vmin=20,vmax=32)
	plt.colorbar()
	plt.savefig("./images/" + str(k) + ".png") 
	plt.gcf().clear()




if __name__ == "__main__":
	os.system("rm images/*")
	x1,y1,v1 = ReadArray("test1.csv")	
	plt.imshow(v1,vmin=20,vmax=32)
	plt.colorbar()
	plt.savefig("./images/" + str(0) + ".png") 
	plt.gcf().clear()

	x2,y2,v2 = ReadArray("test2.csv")	
	plt.imshow(v2,vmin=20,vmax=32)
	plt.colorbar()
	plt.savefig("./images/" + str(102) + ".png") 
	plt.gcf().clear()


	x3,y3,v3 = ReadArray("test3.csv")	
	plt.imshow(v3,vmin=20,vmax=32)
	plt.colorbar()
	plt.savefig("./images/" + str(202) + ".png") 
	plt.gcf().clear()

	QuadInterp(v1,v2,v3)

	#os.system( "cd ./images ; ffmpeg -r 3 -f image2 -i %d.png -vcodec libx264 -crf 60 -pix_fmt yuv420p test.mp4 ; open test.mp4 ; cd ..") 

