import os
from os import listdir
import random
import cv2 
import numpy as np 
import sys
from datetime import datetime
from scipy import ndimage
from PIL import Image

def process_image(img,name_of_image):

    img_array = np.asarray(img)                                                         # CONVERT IMAGE TO ARRAY
    img_array8 = np.divide(img_array, 16)
    img_array8_1 = np.multiply(np.divide(img_array8, np.amax(img_array8)),250).astype('uint8')

    img8 = Image.fromarray(img_array8_1)                                                                      # CONVERT ARRAY TO IMAGE
    img8.save(name_of_image+'_8bit.tiff')                                                          # SAVE INDIVIDUAL FRAME 8 BIT
                    
    img8= cv2.imread(name_of_image+'_8bit.tiff',cv2.IMREAD_COLOR)                    
    gray=  cv2.cvtColor(img8, cv2.COLOR_BGR2GRAY)
    img8_blur=cv2.medianBlur(gray,21)
    mean, std, med = int(np.mean(img8_blur)), int(np.std(img8_blur)), int(np.median(img8_blur))

    cv2.imwrite(name_of_image+'_8bit.tiff', img8_blur)
    return(None)


def contrast_it_up(filename, alpha, beta):  #give beta as 20 for lipid channel and 0 for TD 
    img= cv2.imread(filename,cv2.IMREAD_COLOR)
    lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    adjusted = cv2.convertScaleAbs(final, alpha=alpha, beta=beta)                      # ADJUST CONTRAST WITH ALPHA AND BETA VALUE. BETA IS SET TO 0 FOR NOW AS WE DO NOT WANT TO OVERLY INCREASE THE IMAGE BRIGHTNESS
    cv2.imwrite(filename[:(len(filename)-5)]+'_contrasted.tiff', adjusted)

    duller_adjusted = adjusted_array = np.asarray(adjusted)                                                             # CONVERT CONTRASTED IMAGE TO A NUMPY ARRAY
    mean, med = int(np.mean(adjusted_array)), int(np.median(adjusted_array))                                            # FIND MEAN AND MEDIAN OF THE CONTRASTED IMAGE PIXEL VALUES
 

    duller_adjusted[np.where(duller_adjusted < med)] = 0                                                               # POINTS WITH INTENSITY LOWER THAN MEAN TO BE SET TO 0
    duller_adjusted[np.where(duller_adjusted >= med)] = duller_adjusted[np.where(duller_adjusted >= med)] - med      # POINTS WITH INTENSITY HIGHER THAN MEAN IS SUBTRACTED BY THE MEAN VALUE. 
    duller_adjusted = Image.fromarray(duller_adjusted)                                                                  # CONVERT BACKGROUND SUBTRACTED ARRAY BACK TO AN IMAGE

    duller_adjusted.save(filename[:(len(filename)-5)]+'_contrasted.tiff')                                               # SAVE CONTRAST AND BG CORRECTED IMAGE

    duller_adjusted= cv2.imread(filename[:(len(filename)-5)]+'_contrasted.tiff', cv2.IMREAD_COLOR)                      # OPEN IMAGE AS A OPENCV IMAGE IN COLOUR
    dull_gray = cv2.cvtColor(duller_adjusted, cv2.COLOR_BGR2GRAY)

    return(dull_gray)                                                     # RETURN OPENCV IMAGE BACK TO FIND CIRCLE FUNCTION


def function_intensity(reference_points, images, max_thickness, number,frame_number):
    a, b, r = reference_points[0], reference_points[1], reference_points[2]                                                                 # SEPERATE THE CENTER OF THE CIRLE AND RADIUS                                                    
    center=np.array((b,a))                                                                                                                  # CONVERT CENTER TO A NUMPY ARRAY
    image_size = 512
    counts = []
    thicknesses = []                                                                                                                        # SET IMAGE SIZE
    inner_count =[]
    outer_count = []
    intensity_ring, intensity_ring_bg_correction=[], []                                                                                                                    # INITIALISE VARIABLE THAT WILL STORE THE VESICLE RING INTENSITIES OF THE DIFFERENT CHANNELS 
    mean_intensity_outer_bg, std_intensity_outer_bg = [], []                                                                                # INITIALISE VARIABLE THAT WILL STORE OUTER BACKGROUND INTENSITIES (MEAN AND STD DEV) OF THE DIFFERENT CHANNELS
    mean_intensity_inner_bg, std_intensity_inner_bg = [], []                                                                                # INITIALISE VARIABLE THAT WILL STORE INNER BACKGROUND INTENSITIES (MEAN AND STD DEV) OF THE DIFFERENT CHANNELS     
    delta=1

    for i in range(0,number_of_channels):                                                                                                   # LOOP OVER ALL THE CHANNELS
        if names[i]!='NA':                                                                                                                  # IF THE LOOP IS A FLUORESCENT CHANNEL
            max_intensity=0.0
            sum_of_intensity, sum_of_intensity_norm =0.0,0.0
            data_to_write = np.zeros((r+max_thickness,3), dtype=float)
            name=file1[:len(file1)-len(file1.split('_')[len(file1.split('_'))-1])] \
            +'C00'+str(i+1)+file1.split('_')[len(file1.split('_'))-1].split(reference_name)[1]                                              # NAME OF FILE CORRESPONDING TO THE CHANNEL

            data_file = open(name[:len(name)-5]+'_'+str(delta)+'_radial_data.txt','w')
            image=images[i]                                                                                                                 # SELECT GRAYSCALE IMAGE CORRESPONDING TO THE CURRENT CHANNEL

            image_as_array = np.array(image)                                                                                                # CONVERT GRAYSCALE IMAGE TO A NUMPY ARRAY
            image_as_array_bg_corr=np.zeros(image_as_array.shape, dtype=int)

            matrix = np.ones(image_as_array.shape, dtype=bool)                                                                               # CREATE AN ARRAY OF X AND Y DIMENSION = image_size. EVERY POSITION HAS VALUE OF ONE                 
            matrix[a,b] = False                                                                                                              # SET VALUE AT THE COORDINATE OF THE CIRCLE CENTER TO ZERO

            distance_from_center= ndimage.distance_transform_edt(matrix)                                                                     # FINDS DISTANCE OF ALL COORDINATES W.R.T TO COORDINATE WITH VALUE 0

            background_ring_points = np.where( (distance_from_center >= 200) &  image_as_array>0)
            mean_background = np.mean(image_as_array[background_ring_points[1],background_ring_points[0]])
            print(mean_background)
            image_as_array_bg_corr = image_as_array - mean_background
            image_as_array_bg_corr[np.where(image_as_array_bg_corr<0)] = 0

            for offset_from_center in range (0,r+max_thickness):
                ring_points= np.where( (distance_from_center >= offset_from_center) &  (distance_from_center <= offset_from_center+ delta))   
                ring_points_array = image_as_array_bg_corr[ring_points[1],ring_points[0]]
                ring_points_mean, ring_points_std = np.mean(ring_points_array), np.std(ring_points_array)
                if ring_points_mean>max_intensity:
                    max_intensity=ring_points_mean
                data_to_write[offset_from_center]=[offset_from_center,ring_points_mean,ring_points_std]
            

            #for offset_from_center in range (0,r+max_thickness):
            for offset_from_center in range (0,200):
                data_file.write(str(data_to_write[offset_from_center][0])+'\t'+"{:.2f}".format(data_to_write[offset_from_center][1])+'\t'+"{:.2f}".format(data_to_write[offset_from_center][2])+'\t'+"{:.2f}".format(data_to_write[offset_from_center][1]/max_intensity)+'\t'+"{:.2f}".format(data_to_write[offset_from_center][2]/max_intensity)+'\n')
                sum_of_intensity+=data_to_write[offset_from_center][1]
                sum_of_intensity_norm+=data_to_write[offset_from_center][1]/max_intensity
            data_file.close()
            summary_of_intensity.write(str(frame_number)+'\t'+"{:.2f}".format(sum_of_intensity)+'\t'+"{:.2f}".format(sum_of_intensity_norm)+'\n')
        else:
            # IF CHANNEL IS NOT A FLUORESCENT CHANNEL, ATTACH VALUE 'None' TO THE ARRAY
            thicknesses.append(None)
            intensity_ring.append(None)
            intensity_ring_bg_correction.append(None)
            mean_intensity_outer_bg.append(None)
            std_intensity_outer_bg.append(None)
            mean_intensity_inner_bg.append(None)
            std_intensity_inner_bg.append(None)
            counts.append(None)
            inner_count.append(None)
            outer_count.append(None)

# THE PARAMETERS THAT CONTROL THE CIRCLE FINDING AND CONTRASTING FUNCTIONS 

parameter2,  alpha_initial, beta_initial, alpha, beta = 80, 1, 100, 0.8, 100
max_thickness=255         # SOME CRAZY LARGE DISTANCE AWAY FROM DETECTED CIRCLE
zoom='4'                  # MAGNIFICATION ASSUMED TO BE 4 UNLESS OTHEWISE SPECIFIED IN THE FILENAME 

# THRESHOLD FOR GOOD CIRCLES FOUND
distance_discrepancy = 30     # MAX DISTANCE BETWEEN CIRCLES FOUND IN DIFFERENT CHANNELS OF THE SAME IMAGE TO BE CONSIDERED A GOOD CIRCLE FIT. DISTANCE OF CENTER COORDINATES AND RADIUS CHECKED. 

# GET TOTAL NUMBER OF CHANNELS PRESENT IN THE IMAGE STACK
number_of_channels= int(input('\nTotal number of channels in image (including transmission channel): '))

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
        if img.n_frames==1 and number_of_channels>1:
            print(file1)
            print('\nThis image is a single page Tif image. Analysis needs multi page Tif images. Please export images in multi Tif format. Bye.')
            sys.exit()

        number_of_images_per_channel = int(img.n_frames/number_of_channels)                                 # NUMBER OF IMAGE SETS. WOULD BE GREATER THAN 1 IN TIME SERIES, OR FRAP EXPERIMENTS
        print(number_of_images_per_channel, img.n_frames)
        if number_of_images_per_channel >= 1:
            #for j in range(0, number_of_channels): 
            if number_of_channels==1:
                for i in range(0,number_of_images_per_channel): 
                    channel_number=1
                    img.seek(i)
                    img.save(name+'_%s_C00%s.tiff'%(i+1,channel_number,))
                    name_of_image=name+'_%s_C00%s'%(i+1,channel_number,)
                    process_image(img,name_of_image)
            else:
                counter=1
                for i in range(0,number_of_images_per_channel):
                    channel_number=1
                    while(channel_number<=number_of_channels):
                #for i in range(0,img.n_frames):                                               # LOOP OVER EACH SET OF IMAGES
                    
                        img.seek(counter-1)                                                                         # FIND INDIVIDUAL FRAMES

                        img.save(name+'_%s_C00%s.tiff'%(i+1,channel_number,))                                             # SAVE INDIVIDUAL FRAME 16 BIT
                        name_of_image=name+'_%s_C00%s'%(i+1,channel_number,)
                        process_image(img,name_of_image)

                        counter=counter+1
                        channel_number+=1

        img.close() 
if check==0:
    print('\nNo tif files in directory. Bye')
    sys.exit()
file_num=0

print('\nMoving left to right in the multi page image, provide names for the different channels.\n\nStarting at C001 and ending at C00%s\n'%(number_of_channels,))
print('Provide names only for the fluorescent channel (eg: protein, lipid, red, blue, etc.)\n')
print('If the unique file identifier (C001-%s) does not correspond to a fluorescent channel, type n/N\n'%(number_of_channels,))
print('If incorrect name is typed, type exit/Exit/EXIT to end the program.\n')


number_of_fluorescent_channels = 0

names = []
reference_name=''

for k in range(0,number_of_channels):                   # LOOP OVER ALL THE CHANNELS

    channel_name = input('C00'+str(k+1)+' name: ')      # GET NAME OF THE CHANNEL

    if str.lower(channel_name)=='exit':                 # EXIT FLAG CHECK
        print('Adios. See you next time.')
        sys.exit()
    else:                                                
        if str.lower(channel_name)!='n':                # IF THE CHANNEL CORRESPONDS TO A FLUORESCENT CHANNEL
            names.append(channel_name)                  # USE USER INPUT AS THE NAME OF THE CHANNEL
            reference_name='C00'+str(k+1)
            number_of_fluorescent_channels+=1
        else:
            names.append('NA')                          # IF THE CHANNEL CORRESPONDS TO A NON-FLUORESCENT CHANNEL, MARK SEPERATELY AS SUCH

if number_of_fluorescent_channels==0:                   # CHECK TO ENSURE AT LEAST 1 OF THE CHANNELS IS A FLUORESCENT CHANNEL
    print('Need at least 1 fluorescent channel. See you later. Adios\n')
    sys.exit()

startTime = datetime.now()                              # TIME COUNTER IS STARTED

summary_of_intensity=open('summary_of_intensity_bg_corr.txt','w')
summary_of_intensity.write('Frame#\nSum_of_radial_norm\n')


name_8bit=''            # NAME OF THE FILE CONTAINING THE 8BIT VERSION OF THE IMAGE. 
bad_cases=0
number = 0

for file1 in os.listdir('.'):
    if file1.endswith('.tiff') and reference_name in file1 and 'points_of_interest' not in file1 and '8bit' not in file1 and 'contrasted' not in file1 and 'denoised' not in file1: # ONLY ANALYSE THE 16 BIT VERSION OF THE TIFF FILES AND NOT THEIR MODIFIED VERSIONS. 
        
        image,  reference_points, count, bad_detection =[], [255,255,0], 0, 0
        
        for i in range(0,number_of_channels):
            if names[i]!='NA':
                name=file1[:len(file1)-len(file1.split('_')[len(file1.split('_'))-1])] \
                +'C00'+str(i+1)+file1.split('_')[len(file1.split('_'))-1].split(reference_name)[1] 
                
                frame_number= file1[:len(file1)-len(file1.split('_')[len(file1.split('_'))-1])].split('_')[len(file1[:len(file1)-len(file1.split('_')[len(file1.split('_'))-1])].split('_'))-2]
                print(frame_number)

                img16 = Image.open(name)                                                            # OPEN 16 BIT IMAGE

                img16_as_array = np.asarray(img16)                                                  # CONVERT 16 BIT IMAGE TO ARRAY

                image.append(img16_as_array)                                                        # APPEND THE ARRAY TO THE SERIES OF CHANNEL IMAGES. 

                img16.close() 

                bad_written = 0
        
                                                                # IF AT LEAST ONE CHANNEL HAD A CIRCLE, AND IF MULTIPLE IMAGES HAD CIRCLES IN GOOD AGREEMENT CALCULATE THE INTENSITIES OF THE RINGS FOR EACH CHANNEL 
                function_intensity(reference_points, image, max_thickness, number,frame_number)

        number+=1

print('Time taken ',(datetime.now() - startTime))                                                   # PRINT TOTAL TIME TAKE FOR THE ANALYSIS. 
summary_of_intensity.close()
# CLOSE THE INTENSITY FILE AND THE MANUAL CHECK FILE 


