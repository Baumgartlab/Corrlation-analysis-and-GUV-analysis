import os
import sys
from os import listdir
import numpy as np
import pandas as pd
import scipy
from scipy import ndimage

for file1 in os.listdir('.'):
	if '_autocor.txt' in file1: 
		parts=file1.split('_')
		name = file1[0:len(file1)- len(parts[len(parts)-1])-1 ]
		data_np = np.loadtxt(file1)

		matrix = np.ones(data_np.shape, dtype=bool)                                                                               # CREATE AN ARRAY OF X AND Y DIMENSION = image_size. EVERY POSITION HAS VALUE OF ONE                 
		matrix[511,511] = False                                                                                                              # SET VALUE AT THE COORDINATE OF THE CIRCLE CENTER TO ZERO
		distance_from_center= ndimage.distance_transform_edt(matrix)                                                                     # FINDS DISTANCE OF ALL COORDINATES W.R.T TO COORDINATE WITH VALUE 0

		radavg = open(name+'_radial_avgautocor.txt','w')
		radavg.write('time\tval\n')
		for r in range(0,512):
			values = data_np[np.where((distance_from_center > r-0.5) & (distance_from_center <= r +0.5))]
			radial_avg = np.mean(values)
			radavg.write(str(r)+'\t'+str(radial_avg)+'\n')
		radavg.close()
