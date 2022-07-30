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

def find_circle(filename, parameter2, alpha, beta):                                     # give alpha as 2 for lipid channel and 1 for TD. 
    par1=50                                                                             # default VALUE IS 50
    par2=parameter2
    resolution_of_output=1                                                              # 1 means accumulator has the same resolution as the input image
    minDist = 150                                                                       # min distance between 2 circles in the image. Larger value reduces multiple fits for same circle

    img= cv2.imread(filename,cv2.IMREAD_COLOR)                                          # OPEN IMAGE AS A OPENCV IMAGE IN COLOUR
    gray=  cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                                        # CONVERT RGB IMAGE TO GRAYSCALE IMAGE
    for i in range(0,30-int(alpha*10)):                                                    # alpha max value is 3.
        gray_blurred = cv2.blur(gray, (3, 3))                                           # CONVERY GRAYSCALE IMAGE TO A BLURRED IMAGE TO MAKE CIRCLE FINDING EASIER. 
        alpha += 0.1                                                                 # ALPHA VALUE IS INCREMENTALLY INCREASED TO FEED TO CONTRAST FUNCTION
        for j in range (0,20):
            par2=par2+j                                                                 # PARAMETER2 VALUE IS INCREMENTALLY INCREASED TO FEED TO HOUGHCIRCLE FUNCTION

            detected_circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT, resolution_of_output, minDist, param1 = par1, param2 = par2, minRadius = 30, maxRadius = 400) # LOOK FOR CIRCLES AND RETURNS CIRCLE CENTER AND RADIUS

            if detected_circles is not None:                                            # IF ATLEAST 1 CIRCLE IS FOUND
                plural=len(detected_circles[0, :])                                      # NUMBER OF CIRCLES FOUND
                if plural==1:                                                           # IF ONLY 1 CIRCLE IS FOUND
                    return(detected_circles)                                            # RETURN THE CIRCLE POSITION AND RADIUS
        par2=parameter2                                                                 # RESET PARAMETER 2 VALUE TO BASE VALUE
        gray = contrast_it_up(filename, alpha, beta)                        # ENHANCE IMAGE CONTRAST, VALUE RETURNED IS IN GRAYSCALE IMAGE IN OPENCV FORMAT
    print(detected_circles)
    return(None)                                                                        # IF EITHER CIRCLE FINDING DID NOT FIND A CIRCLE OR FOUND MULTIPLE CIRCLES, 'None' IS RETURNED


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

    for i in range(0,number_of_channels):                                                                                                   # LOOP OVER ALL THE CHANNELS
        if names[i]!='NA':                                                                                                                  # IF THE LOOP IS A FLUORESCENT CHANNEL

            name=file1[:len(file1)-len(file1.split('_')[len(file1.split('_'))-1])] \
            +'C00'+str(i+1)+file1.split('_')[len(file1.split('_'))-1].split(reference_name)[1]                                              # NAME OF FILE CORRESPONDING TO THE CHANNEL

            image=images[i]                                                                                                                 # SELECT GRAYSCALE IMAGE CORRESPONDING TO THE CURRENT CHANNEL

            image_as_array = np.array(image)                                                                                                # CONVERT GRAYSCALE IMAGE TO A NUMPY ARRAY

            matrix = np.ones(image_as_array.shape, dtype=bool)                                                                               # CREATE AN ARRAY OF X AND Y DIMENSION = image_size. EVERY POSITION HAS VALUE OF ONE                 
            matrix[a,b] = False                                                                                                              # SET VALUE AT THE COORDINATE OF THE CIRCLE CENTER TO ZERO

            distance_from_center= ndimage.distance_transform_edt(matrix)                                                                     # FINDS DISTANCE OF ALL COORDINATES W.R.T TO COORDINATE WITH VALUE 0

            outer_bg_points = np.where( (distance_from_center >= r + max_thickness) &  (distance_from_center <= r+ 3*max_thickness))                                                        # OUTER BACKGROUND POINTS: FIND ALL POINTS THAT ARE r + max_thickness OR GREATER DISTANCE FROM CIRCLE CENTER
            inner_bg_points = np.where(distance_from_center <= r - max_thickness)                                                            # INNER BACKGROUND POINTS: FIND ALL POINTS THAT ARE r - max_thickness OR LESSER DISTANCE FROM CIRCLE CENTER

            outer_bg = image_as_array[outer_bg_points[1],outer_bg_points[0]]                                                                 # ARRAY CONTAINING THE VALUE OF ALL OUTER BACKGROUND POINTS
            inner_bg = image_as_array[inner_bg_points[1],inner_bg_points[0]]                                                                 # ARRAY CONTAINING THE VALUE OF ALL INNER BACKGROUND POINTS

            inner_bg_mean, inner_bg_std = np.mean(inner_bg), np.std(inner_bg)                                                                # INNER BACKGROUND MEAN AND STANDARD DEVIATION
            outer_bg_mean, outer_bg_std = np.mean(outer_bg), np.std(outer_bg)                                                                # OUTER BACKGROUND MEAN AND STANDARD DEVIATION

            intensity_0 = 0                                                                                                                  # TO STORE PREVIOUS ITERATION'S INTENSITY VALUE
            count_0 = 0                                                                                                                      # TO STORE PREVIOUS ITERATION'S NUMBER OF PIXELS WITHIN RING
            count_0_in = 0
            count_0_out = 0
            count_0_on = 0
            array_useful_points_0 = np.zeros((image_size,image_size), dtype=int)

            for thickness in range(1,max_thickness):                                                                                         # LOOP OVER INCREASING THICKNESS

                array_useful_points_1=np.zeros((image_size,image_size), dtype=int)                                                                            # CREATE AN ARRAY OF X AND Y DIMENSION = image_size. EVERY POSITION HAS VALUE OF ZERO
                temp1=np.zeros((image_size,image_size), dtype=int)
                points_of_interest_array = np.where(abs(r - distance_from_center) <= thickness)                                              # SET OF POINTS THAT FALL WITHIN r +/- thickness   

                points_inside = np.where( (distance_from_center < r) &  (distance_from_center >= r-thickness))

                count_1_in = len(image_as_array[points_inside[1], points_inside[0]])
                points_outside = np.where( (distance_from_center > r) &  (distance_from_center <= r+thickness))
                count_1_out = len(image_as_array[points_outside[1], points_outside[0]])

                points_on_circle = np.where(distance_from_center==r) 
                count_1_on = len(image_as_array[points_on_circle[1], points_on_circle[0]])

                array_useful_points_1[points_inside[1], points_inside[0]]  = image_as_array[points_inside[1], points_inside[0]]-(inner_bg_mean+inner_bg_std)                   # STORE POINTS THAT ARE WITHIN THE RING r +/- thickness IN THE TEMP ARRAY
                array_useful_points_1[points_outside[1], points_outside[0]]  = image_as_array[points_outside[1], points_outside[0]]-(outer_bg_mean+outer_bg_std)
                array_useful_points_1[points_on_circle[1], points_on_circle[0]]  = image_as_array[points_on_circle[1], points_on_circle[0]]-(inner_bg_mean+outer_bg_mean+outer_bg_std+inner_bg_std)/2
                array_useful_points_1[np.where(array_useful_points_1<0)] = 0

                temp1[points_of_interest_array[1], points_of_interest_array[0]] = image_as_array[points_of_interest_array[1], points_of_interest_array[0]]   
                                                                                                                            # FIRST ITERATION

                intensity_1 = np.sum(image_as_array[points_of_interest_array[1], points_of_interest_array[0]])                               # STORE CURRENT ITERATION'S INTENSITY VALUE
                count_1 = len(image_as_array[points_of_interest_array[1], points_of_interest_array[0]])                                      # STORE CURRENT ITERATION'S NUMBER OF PIXELS WITHIN RING

                if (intensity_1 - intensity_0)<= ((inner_bg_mean+inner_bg_std) *(count_1_in-count_0_in)+ (outer_bg_mean+outer_bg_std)*(count_1_out-count_0_out))  or thickness==max_thickness-1:        # CHECK IF INCREASE IN INTENSITY IS LOWER THAN AVERAGE BACKGROUND VALUE + 1 STD DEV

                    # APPEND RING INTENSITY, INNER AND OUTER BG MEAN AND STD DEV INTENSITIES TO THE RESPECTIVE ARRAY
                    if thickness==1:
                        intensity_ring.append(intensity_1)
                        intensity_ring_bg_correction.append(np.sum(array_useful_points_1))

                        counts.append(count_1)
                        thicknesses.append(thickness)
                    else:
                        intensity_ring.append(intensity_0)
                        intensity_ring_bg_correction.append(np.sum(array_useful_points_0))
                        counts.append(count_0)
                        thicknesses.append(thickness-1)
                    mean_intensity_outer_bg.append(outer_bg_mean)
                    std_intensity_outer_bg.append(outer_bg_std)
                    mean_intensity_inner_bg.append(inner_bg_mean)
                    std_intensity_inner_bg.append(inner_bg_std)
                    inner_count.append(count_1_in)
                    outer_count.append(count_1_out)

                    array_useful_points_1_8bit = np.divide(array_useful_points_1,16)                                                                               # CONVERT 16 BIT TO 8 BIT WITH CORRECT SCALING
                    array_useful_points_1_8bit = np.multiply(np.divide(array_useful_points_1_8bit, np.amax(array_useful_points_1_8bit)),250).astype('uint8')
                    temp_img = Image.fromarray(array_useful_points_1_8bit)                                                                                        # CONVERT ARRAY TO IMAGE
                    temp_img.save(name[:len(name)-5]+'_points_of_interest.tiff')                                                             # SAVE IMAGE
                    break
                else:                                                                                                                        # IF INTENSITY GAIN IS GREATER THAN BACKGROUND INTENSITY                   
                    intensity_0 = intensity_1                                                                                                # UPDATE PREVIOUS INTENSITY WITH CURRENT 
                    count_0 = count_1                                                                                                        # UPDATE PREVIOUS PIXEL COUNT WITH CURRENT
                    count_0_in = count_1_in
                    count_0_out = count_1_out
                    count_0_on = count_1_on
                    array_useful_points_0=array_useful_points_1
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

    # WRITE INTENSITY OF RING AND INNER AND OUTER BACKGROUND FOR ALL THE CHANNELS
    intensity.write(str(frame_number)+'\t'+str(r)+'\t'+str(zoom)+'\t'+"{:.2f}".format(float(r)/float(zoom))+'\t')
    bg_data.write(str(frame_number)+'\t'+str(r)+'\t')
    for i in range(0,number_of_channels):
        if names[i]!='NA':
            intensity.write(str(thicknesses[i])+'\t'+str(intensity_ring[i])+'\t'+"{:.2f}".format(intensity_ring[i]/(2*np.pi*float(r)))+'\t'+"{:.2f}".format(intensity_ring_bg_correction[i])+'\t'+"{:.2f}".format(intensity_ring_bg_correction[i]/(2*np.pi*float(r)))+'\t')
            bg_data.write(str(inner_count[i])+'\t\t'+"{:.2f}".format(mean_intensity_inner_bg[i])+'\t\t'+"{:.2f}".format(std_intensity_inner_bg[i])+'\t\t'+str(outer_count[i])+'\t\t'+"{:.2f}".format(mean_intensity_outer_bg[i])+'\t\t'+"{:.2f}".format(std_intensity_outer_bg[i])+'\t\t')
        
    intensity.write(name_8bit[:len(name_8bit)-5]+'\n')
    bg_data.write(name_8bit[:len(name_8bit)-5]+'\n')


parameter2,  alpha_initial, beta_initial, alpha, beta = 80, 1, 100, 0.8, 100
max_thickness=25         # SOME CRAZY LARGE DISTANCE AWAY FROM DETECTED CIRCLE
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
        if img.n_frames==1:
            print(file1)
            print('\nThis image is a single page Tif image. Analysis needs multi page Tif images. Please export images in multi Tif format. Bye.')
            sys.exit()

        number_of_images_per_channel = int(img.n_frames/number_of_channels)                                 # NUMBER OF IMAGE SETS. WOULD BE GREATER THAN 1 IN TIME SERIES, OR FRAP EXPERIMENTS
        print(number_of_images_per_channel, img.n_frames)
        if number_of_images_per_channel >= 1:
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

intensity=open('Intensities.txt','w')
intensity.write('Frame#\tRadius\tZoom\t1XRad\t')
bg_data = open('Background_intensities.txt','w')
bg_data.write('Radius\t')
for i in range(0,number_of_channels):
    if names[i]!='NA':
        intensity.write('#'+str(i+1)+'Thx\t'+'#'+str(i+1)+'\t'+'Norm#'+str(i+1)+'\t'+'#'+str(i+1)+'wBC\t'+'Norm#'+str(i+1)+'wBC\t')
        bg_data.write('#'+str(i+1)+'InnerCount\t#'+str(i+1)+'InnerMean\t#'+str(i+1)+'InnerStd\t#'+str(i+1)+'OuterCount\t#'+str(i+1)+'OuterMean\t#'+str(i+1)+'OuterStd\t')
bg_data.write('Filename\n')
intensity.write('Filename\n')

check=open('Files_for_manual_check.txt','w')
check.write('#Circles\tFilename\n')

name_8bit=''            # NAME OF THE FILE CONTAINING THE 8BIT VERSION OF THE IMAGE. 
bad_cases=0
number = 0
for file1 in os.listdir('.'):
    if file1.endswith('.tiff') and reference_name in file1 and 'points_of_interest' not in file1 and '8bit' not in file1 and 'contrasted' not in file1 and 'denoised' not in file1: # ONLY ANALYSE THE 16 BIT VERSION OF THE TIFF FILES AND NOT THEIR MODIFIED VERSIONS. 
        


        for seg in range(0,len(file1.split('_'))):                                                  # GET THE MAGNIFICATION OF THE IMAGE IF INCLUDED IN THE FILE NAME
            if 'X' in file1.split('_')[seg] or 'x'  in file1.split('_')[seg] :
                zoom=(file1.split('_'))[seg][0]

        image,  reference_points, count, bad_detection =[], [0.0,0.0,0.0], 0, 0                     # DECLARE VARIABLES THAT WILL STORE THE GRAYSCALE IMAGES, CIRCLE DETAILS, PIXEL COUNT AND FLAG FOR QUALITY OF CIRCLES DETECTED 

        bad_written = 0
        for i in range(0,number_of_channels):                                                       # LOOP OVER ALL THE CHANNELS


            if names[i]!='NA':                                                                      # IF THE CHANNEL IS A FLUORESCENT CHANNEL
                name=file1[:len(file1)-len(file1.split('_')[len(file1.split('_'))-1])] \
                +'C00'+str(i+1)+file1.split('_')[len(file1.split('_'))-1].split(reference_name)[1]  # GET THE NAME OF THE FILE CORRESPONDING TO THIS CHANNEL. 
                name_8bit = name[:len(name)-5]+'_8bit.tiff'                                         # CORRESPONDING NAME OF THE 8BIT VERSION OF THE IMAGE
                frame_number= file1[:len(file1)-len(file1.split('_')[len(file1.split('_'))-1])].split('_')[len(file1[:len(file1)-len(file1.split('_')[len(file1.split('_'))-1])].split('_'))-2]
                print(file1, name)
                detected_circles = find_circle(name_8bit, parameter2, alpha, beta)                  # FIND CIRCLES PRESENT IN THE IMAGE

                print(detected_circles)


                if detected_circles is not None:                                                    # CHECK IF A CIRCLE WAS FOUND
                    if count==0:                                                                    # FIRST CIRCLE FOUND IN THE SET OF IMAGES
                        reference_points=detected_circles[0, :][0]                                  # REFERENCE POINT STORES THE CENTER AND RADIUS OF THE CIRCLE
                        count+=1
                    else:                                                                           # IF A CIRCLE IS FOUND IN MORE THAN ONE CHANNEL                       
                        
                        if abs(reference_points[2]- detected_circles[0, :][0][2]) > distance_discrepancy:             # CHECK IF RADIUS OF THE DETECTED CIRCLES ARE WITHIN 20 
                            bad_detection = 1
                            print('radius discrepency')
                            check.write('Radius discrepency\t'+name_8bit+'\n')
                        elif abs(reference_points[0]- detected_circles[0, :][0][0]) > distance_discrepancy or abs(reference_points[1]- detected_circles[0, :][0][1]) > distance_discrepancy: # CHECK IF X OR Y COORDINATE OF CENTER IS MORE THAN 20 PX AWAY FROM PREVIOUS DETERMINATION
                            bad_detection = 1
                            print('center discrepency')
                            check.write('center discrepency\t'+name_8bit+'\n')
                        else:
                            reference_points = ((reference_points*count) + detected_circles[0, :][0])       # IF CIRCLE IS DEEMED A GOOD FIT, SUM UP THE RADII AND CENTERS' COORDINATES
                            count+=1
                            reference_points = reference_points/count                                       # AVERAGE THE CENTER COORDINATES AND RADII TO GET MEAN POSITION OF THE CIRCLE FIT. 

                img16 = Image.open(name)                                                            # OPEN 16 BIT IMAGE

                img16_as_array = np.asarray(img16)                                                  # CONVERT 16 BIT IMAGE TO ARRAY

                image.append(img16_as_array)                                                        # APPEND THE ARRAY TO THE SERIES OF CHANNEL IMAGES. 

                img16.close() 
                if count==0:                                                                        # IF NO CIRCLES WERE FOUND IN ANY OF THE CHANNELS, WRITE IN SEPERATE FILE FOR MANUAL CHECKS.
                    check.write('None\t'+file1+'\n')
                else:
                    reference_points=np.uint16(np.around(reference_points))                         # CONVERT THE CENTER COORDINATES AND RADII TO UNSIGNED INTERGERS. 
            else:
                image.append(None)                                                                  # IF THIS CHANNEL WAS A NON FLUORESCENT CHANNEL, APPEND 'None' TO THE IMAGES. 
        
        
        if count!=0 and bad_detection==0:                                                           # IF AT LEAST ONE CHANNEL HAD A CIRCLE, AND IF MULTIPLE IMAGES HAD CIRCLES IN GOOD AGREEMENT CALCULATE THE INTENSITIES OF THE RINGS FOR EACH CHANNEL 
            function_intensity(reference_points, image, max_thickness, number,frame_number)
        else:
            bad_cases+=1

        number+=1

print('Time taken ',(datetime.now() - startTime))                                                   # PRINT TOTAL TIME TAKE FOR THE ANALYSIS. 

# CLOSE THE INTENSITY FILE AND THE MANUAL CHECK FILE 

intensity.close()   
check.close()

