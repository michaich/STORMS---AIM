import pandas as pd 
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import os,scipy
from matplotlib import cm
from scipy.interpolate import griddata
from scipy import ndimage



def Coarsen(U,l):
	
	s = 4
	Uavg = np.copy(U)
	
	for i in range (0,l,s):
		for j in range (0,l,s):
			
			Uavg_ = 0.0
			Vavg_ = 0.0
			for k1 in range(0,s):
				for k2 in range(0,s):
					Uavg_ += U[i+k1][j+k2]
			Uavg_ = Uavg_ / s/s
			for k1 in range(0,s):
				for k2 in range(0,s):
					Uavg[i+k1][j+k2] = Uavg_


			
		
	return Uavg
	
			
			



def PlotVorticity(omega,x,y,xMax,yMax,k):

	grid_x, grid_y = np.mgrid[0.0:xMax:(l*1j), 0.0:yMax:(l*1j)]     	

	grid = griddata( (x.ravel(),y.ravel()) ,omega.ravel(), (grid_x,grid_y) ,method='linear',fill_value='nan')

	plt.imshow(grid.T, extent=(0.0,xMax,0.0,yMax),origin='lower',cmap=cm.jet)
		
	plt.colorbar()

	plt.savefig("./images/" + str(k) + ".png") 

	plt.gcf().clear() 
	


class Field:
	def __init__(self, name):
		dataP = np.genfromtxt(name + "_P.csv",deletechars="data",skip_header=2,delimiter=";")	
		datar = np.genfromtxt(name + "_r.csv",deletechars="data",skip_header=2,delimiter=";")	
		dataU = np.genfromtxt(name + "_U.csv",deletechars="data",skip_header=2,delimiter=";")	
		dataV = np.genfromtxt(name + "_V.csv",deletechars="data",skip_header=2,delimiter=";")	
		self.x = dataP[0,1:]
		self.y = dataP[1:,0]
		self.P = dataP[1:,1:]
		self.r = datar[1:,1:]
		self.U = dataU[1:,1:]
		self.V = dataV[1:,1:]
		

def Interpolation(f1,f2,xMax,yMax):

	l = f1.U.shape[0]
	print(l)
	T            = np.linspace(0,6*3600,11)	
	dt_particles = T[1] - T[0]		
	dt           = 6.*3600.
	
	dUdt = (f2.U - f1.U)/dt
	dVdt = (f2.V - f1.V)/dt

	x = np.zeros((l,l))
	y = np.zeros((l,l))
	for i in range (0,l):
		for j in range (0,l):
			x[i][j] = f1.x[i]
			y[i][j] = f1.y[j]


	deltaX = xNew[1][1]-xNew[0][0]
	deltaY = yNew[1][1]-yNew[0][0]

	gradVx = np.gradient(f1.V ,deltaX,axis=0)
	gradUy = np.gradient(f1.U ,deltaY,axis=1)
	
	Omega = gradVx - gradUy


	U = np.copy(f1.U)
	V = np.copy(f1.V)

	Ubar = np.zeros((l,l))
	Vbar = np.zeros((l,l))


	k = 0
	PlotVorticity(Omega,x,y,xMax,yMax,k)

	for t in T:
		k +=1
		print(k,t)
		
		U  =  f1.U + t*dUdt  
		V  =  f1.V + t*dVdt  	
		
		Ubar = Coarsen(U,l)
		Vbar = Coarsen(V,l)

		x +=  dt_particles*Ubar
		y +=  dt_particles*Vbar
		PlotVorticity(Omega,x,y,xMax,yMax,k)

		grid_x, grid_y = np.mgrid[0.0:xMax:(l*1j), 0.0:yMax:(l*1j)]     	
		grid = griddata( (x.ravel(),y.ravel()) ,Omega.ravel(), (grid_x,grid_y) ,method='linear',fill_value=0)

		Omega = np.copy(grid)

		for i in range (0,l):
			for j in range (0,l):
				x[i][j] = f1.x[i]
				y[i][j] = f1.y[j]


	
	gradVx = np.gradient(f2.V ,deltaX,axis=0)
	gradUy = np.gradient(f2.U ,deltaY,axis=1)	
	Omega = gradVx - gradUy
	k = 444
	PlotVorticity(Omega,x,y,xMax,yMax,k)
  
	
def measure(lat1, lon1, lat2, lon2):
	d2r = np.pi/180.0
	R = 6378.137 # Radius of earth in KM
	dLat = (lat2 - lat1)*d2r
	dLon = (lon2 - lon1)*d2r
	a = np.sin(0.5*dLat)**2 + np.cos(lat1*d2r)*np.cos(lat2*d2r)*np.sin(0.5*dLon) **2
	c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1.0-a));
	d = R * c;
	return d * 1000 #meters



if __name__ == "__main__":

	#os.system("rm images/*")

	f1 = Field("ONE")
	f2 = Field("TWO")


	x0 = 0.0
	y0 = 0.0

	y1 = 0.0
	x1 = measure(15,60,15,70) #hardcoded latitude and longtitudes

	x2 = x1
	y2 = measure(15,70,25,70)

	x3 = 0.0
	y3 = y2 



	f1.x = np.linspace(x0,x1,f1.x.shape[0])
	f1.y = np.linspace(y0,y2,f1.y.shape[0])

	f2.x = np.linspace(x0,x1,f2.x.shape[0])
	f2.y = np.linspace(y0,y2,f2.y.shape[0])


	l = f2.x.shape[0]
	xNew = np.zeros((l,l))
	yNew = np.zeros((l,l))
	for i in range (0,l):
		for j in range (0,l):
			xNew[i][j] = f1.x[i]
			yNew[i][j] = f1.y[j]

	gamma = 1.4

	c        = np.sqrt(gamma*f1.P/f1.r)
	velocity = np.sqrt(f1.U*f1.U+f1.V*f1.V)/c 
	grid_x, grid_y = np.mgrid[np.min(xNew):np.max(xNew):(l*1j), np.min(yNew):np.max(yNew):(l*1j)]     		
	grid = griddata( (xNew.ravel(),yNew.ravel()) ,velocity.ravel(), (grid_x,grid_y) ,method='linear',fill_value='nan')
	plt.imshow(grid.T, extent=(np.min(xNew),np.max(xNew),np.min(yNew),np.max(yNew)),origin='lower',cmap=cm.jet)
	plt.colorbar()
	plt.savefig("./images/start.png") 
	plt.gcf().clear() 

	c        = np.sqrt(gamma*f2.P/f2.r)
	velocity = np.sqrt(f2.U*f2.U+f2.V*f2.V)/c
	grid_x, grid_y = np.mgrid[np.min(xNew):np.max(xNew):(l*1j), np.min(yNew):np.max(yNew):(l*1j)]     		
	grid = griddata( (xNew.ravel(),yNew.ravel()) ,velocity.ravel(), (grid_x,grid_y) ,method='linear',fill_value='nan')
	plt.imshow(grid.T, extent=(np.min(xNew),np.max(xNew),np.min(yNew),np.max(yNew)),origin='lower',cmap=cm.jet)
	plt.colorbar()
	plt.savefig("./images/end.png") 
	plt.gcf().clear() 
	
	#os.system( "cd ./images ; ffmpeg -r 3 -f image2 -i %d.png -vcodec libx264 -crf 60 -pix_fmt yuv420p test.mp4 ; open test.mp4 ; cd ..") 

	Interpolation(f1,f2,x1,y2)

