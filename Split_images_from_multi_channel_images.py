import os
from os import listdir
import random
import cv2 
import numpy as np 
import sys
from datetime import datetime
from scipy import ndimage
from PIL import Image

number_of_channels = 0
if len(sys.argv)!=2:
    print('need number of channels')
    sys.exit()
else:
	number_of_channels = int(sys.argv[1])

if number_of_channels ==0:
    print('Number of channels needs to be atleast 1.')
    sys.exit()

check=0                                                                                                 # FLAG TO CHECK PRESENCE OF TIF FILE IN DIRECTORY

# CONVERT SPACES IN FILE NAME WITH UNDERSCORES
for file1 in os.listdir('.'):
    if file1.endswith('.tif'):
        if ' ' in file1:
            parts=file1.split(' ')
            name=parts[0]
            for i in range(1,len(parts)):
                name+='_'+parts[i]
            os.rename(file1, name)

for file1 in os.listdir('.'):
    if file1.endswith('.tif'):                                                                          # FIND ALL TIF STACK IMAGES 
        print(file1)
        name=file1[:len(file1)-4]                                                                       # NAME OF FILE WITHOUT EXTENSION   
        check=1                                                                                         # FLAG CHANGED TO ONE.
        img = Image.open(file1)                                                                         # OPEN AND LOAD TIF STACK IMAGE                
        img.load()

        # CHECK TO MAKE SURE EXPORTING METHOD IS CORRECT. IF IMAGES ARE SINGLE IMAGES INSTEAD OF A STACK OF IMAGES, ERROR MESSAGE IS DISPLAYED AND PROGRAM TERMINATES
        if img.n_frames==1:
            print(file1)
            print('\nThis image is a single page Tif image. Analysis needs multi page Tif images. Please export images in multi Tif format. Bye.')
            sys.exit()

        number_of_sets_of_images = int(img.n_frames/number_of_channels)                                 # NUMBER OF IMAGE SETS. WOULD BE GREATER THAN 1 IN TIME SERIES, OR FRAP EXPERIMENTS
        if number_of_sets_of_images >= 1:
            current_pos=0
            count=1
            for i in range(1,number_of_channels+1):                                               # LOOP OVER EACH SET OF IMAGES
                count=1
                for j in range(current_pos, current_pos+(int(img.n_frames/number_of_channels))):                                                      # LOOP OVER INDIVIDUAL FRAMES
                    img.seek(j)                                                                         # FIND INDIVIDUAL FRAMES

                    img.save(name+'_%s_C00%s.tiff'%(count,i,))                                             # SAVE INDIVIDUAL FRAME 16 BIT

                    count+=1
                current_pos= current_pos+(int(img.n_frames/number_of_channels))

        img.close() 
if check==0:
    print('\nNo tif files in directory. Bye')
    sys.exit()
