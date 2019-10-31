import meteomatics.api as api
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import os
import cv2
import time



def interpolate_frames(frame, coords, flow, n_frames):
    frames = [frame]
    for f in range(1, n_frames):
        alpha = float(f)/float(n_frames)
        pixel_map = coords + alpha * flow
        inter_frame = cv2.remap(frame, pixel_map, None, cv2.INTER_LINEAR)
        frames.append(inter_frame)
    return frames


def OpticalFlow(Gold, Gnew, nframes):

    t0 = time.time()

    gridoldF=np.float64( Gold ).T
    gridnewF=np.float64( Gnew ).T


    minF_old = np.min(gridoldF)
    maxF_old = np.max(gridoldF)

    minF_new = np.min(gridnewF)
    maxF_new = np.max(gridnewF)

    maxAll = max(maxF_new,maxF_old)
    minAll = min(minF_new,minF_old)

    aux = 1.0 / (maxAll-minAll) 

    gridoldF = (gridoldF-minAll)*aux*256
    gridnewF = (gridnewF-minAll)*aux*256

    t1 = time.time()



    t2 = time.time()

    optflow_params = [0.5, 7, 4*15, 3, 7, 1.5, 0]
    flowUp   = cv2.calcOpticalFlowFarneback(gridnewF, gridoldF, None, *optflow_params)
    flowDown = cv2.calcOpticalFlowFarneback(gridoldF, gridnewF, None, *optflow_params)
    
    t3 = time.time()

    
    

    lx = gridold.shape[0]
    ly = gridold.shape[1]
    ycoords, xcoords = np.mgrid[0:ly, 0:lx]
    coords = np.float32(np.dstack([xcoords, ycoords]))
    

    inter_framesUp   = np.array(interpolate_frames(gridoldF, coords, flowUp  , nframes))
    inter_framesDown = np.array(interpolate_frames(gridnewF, coords, flowDown, nframes))
    inter_frames     = []

    t4 = time.time()
    

    N = len(inter_framesUp)

    for i in range(N):
        alpha = float(i)/float(N)
        inter_frames.append(inter_framesUp[i]*(1.0-alpha) + inter_framesDown[-i]*alpha)
    
    inter_frames.append(gridnewF)
    
    t5 = time.time()

    print("Running for ", nframes, "frames.")
    print("Normalization time=",t1-t0)
    print("Optical Flow time =",t3-t2)
    print("Interpolation time=",t4-t3)
    print("final loop time   =",t5-t4)
    print("Total time        =",t5-t0)

    return inter_frames


username = 'aim_2019'
password = '2RwK18J01at4'
#now = dt.datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)

now = dt.datetime(year=2019,month=10, day=28,hour=0, minute=0, second=0)
timenew = now
timeold = now - dt.timedelta(hours=6)


#parameter_grid = 'msl_pressure:Pa'
parameter_grid = 'wind_speed_10m:kmh'
#parameter_grid = ['t_2m:C', 'precip_1h:mm']
#parameter_grid = 'low_cloud_cover:p'


# KYARR position, 28.10.19, 12h00 GMT
# LAT: 18.5
# LON: 64.4

#Nice weather close to France
#lat_N = 45
#lon_W = -25
#lat_S = 35
#lon_E = 5

#Hurricane
lat_N = 25
lon_W = 60
lat_S = 15
lon_E = 70


res_lat = 0.02
res_lon = 0.02


try:
    gridold = api.query_grid(timeold, parameter_grid, lat_N, lon_W, lat_S, lon_E, res_lat, res_lon,username, password)
    #print (df_grid.head())
except Exception as e:
    print("Failed, the exception is {}".format(e))

try:
    gridnew = api.query_grid(timenew, parameter_grid, lat_N, lon_W, lat_S, lon_E, res_lat, res_lon,username, password)
    #print (df_grid.head())
except Exception as e:
    print("Failed, the exception is {}".format(e))
    

gridnew = gridnew.to_numpy()
gridold = gridold.to_numpy()





inter_frames = OpticalFlow(gridold, gridnew, 10)





fig = plt.figure(dpi=100)
viewer = fig.add_subplot(111)
plt.ion()
fig.show()

for i in range(len(inter_frames)):
    viewer.clear()
    viewer.imshow(inter_frames[i].T)
    plt.pause(.04)
    fig.canvas.draw()


for i in range(len(inter_frames)):
        plt.figure()
        plt.imshow(inter_frames[i].T)
        plt.savefig('./%02d.png' % i)