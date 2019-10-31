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

    t1 = time.time()

    gridoldF=np.float64( Gold ).T
    gridnewF=np.float64( Gnew ).T


    minF_old = np.min(gridoldF)
    maxF_old = np.max(gridoldF)

    minF_new = np.min(gridnewF)
    maxF_new = np.max(gridnewF)

    maxAll = max(maxF_new,maxF_old)
    minAll = min(minF_new,minF_old)

    aux = 1.0 / (maxAll-minAll) 

    gridoldF = (gridoldF-minAll)*aux* 65535 
    gridnewF = (gridnewF-minAll)*aux* 65535 

    t2 = time.time()


    optflow_params = [0.5, 6, 50, 3, 5, 1.2, 0]
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
        inter_frames.append( inter_framesUp[i]*(1.0-alpha) + inter_framesDown[-i]*alpha) 
    
    inter_frames.append(gridnewF)


    inter_frames = inter_frames / (aux* 65535) + minAll


    
    t5 = time.time()

    print("Running for ", nframes, "frames.")
    print("Normalization time=",t2-t1)
    print("Optical Flow time =",t3-t2)
    print("Interpolation time=",t4-t3)
    print("final loop time   =",t5-t4)
    print("Total time        =",t5-t1)

    return inter_frames









username = 'aim_2019'
password = '2RwK18J01at4'

now = dt.datetime(year=2019,month=10, day=28,hour=0, minute=0, second=0)

#parameter_grid = 'msl_pressure:Pa'
parameter_grid = 'wind_speed_10m:kmh'
#parameter_grid = ['t_2m:C', 'precip_1h:mm']
#parameter_grid = 'low_cloud_cover:p'


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


res_lat = 0.03#0.025 #(lat_N - lat_S)/512.  
res_lon = 0.03#0.025 #(lon_E - lon_W)/512.  

g = []

for i in range (13):
    print(i)
    try:
        g.append(api.query_grid(now - dt.timedelta(hours=13-i), parameter_grid, lat_N, lon_W, lat_S, lon_E, res_lat, res_lon,username, password))
    except Exception as e:
        print("Failed, the exception is {}".format(e))


gridnew = g[12].to_numpy()
gridold = g[0 ].to_numpy()
ValMin = np.min(gridnew)
ValMax = np.max(gridnew)

ValMin = min(ValMin,np.min(gridold))
ValMax = max(ValMax,np.max(gridold))

inter_frames = OpticalFlow(gridold, gridnew, 13*10)



for i in range(len(inter_frames)):
        print(i,i/65,i/10)

        plt.subplot(1,3,1)
        plt.imshow(g[i/65],vmin=ValMin,vmax=ValMax)
        plt.title("Real data (6h)",fontsize=8)
        plt.axis('off')
        plt.subplot(1,3,2)
        plt.imshow(g[i/10],vmin=ValMin,vmax=ValMax)
        plt.title("Real data (1h)",fontsize=8)
        plt.axis('off')
        plt.subplot(1,3,3)
        plt.imshow(inter_frames[i].T,vmin=ValMin,vmax=ValMax)
        plt.title("Interpolation (10min)",fontsize=8)
        plt.axis('off')
        
        plt.savefig('./%03d.png' % i)
   


    #    gridnew_ = g[i+1].to_numpy()
    #    gridold_ = g[i  ].to_numpy()
    #    ValMin = np.min(gridnew_)
    #    ValMax = np.max(gridnew_)
    #    ValMin = min(ValMin,np.min(gridold_))
    #    ValMax = max(ValMax,np.max(gridold_))
    #    inter_frames1 = OpticalFlow(gridold_, gridnew_, 6)
    #            
#
#    #    for j in range(len(inter_frames1)):        
#    #        plt.subplot(1,3,1)
#    #        plt.imshow(g[i],vmin=ValMin,vmax=ValMax)
#    #        plt.title("Real data",fontsize=8)
#    #        plt.axis('off')
#    #        plt.subplot(1,3,2)
#    #        plt.imshow(inter_frames[i].T,vmin=ValMin,vmax=ValMax)
#    #        plt.title("Interpolation (1h)",fontsize=8)
#    #        plt.axis('off')
#    #        plt.subplot(1,3,3)
#    #        plt.imshow(inter_frames1[j].T,vmin=ValMin,vmax=ValMax)
    #        plt.title("Interpolation (10min)",fontsize=8)
#            plt.axis('off')
#            k = i*6+j


