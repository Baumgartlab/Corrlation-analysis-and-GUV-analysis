import os
from os import listdir
import random
import cv2 
import numpy as np 
import sys
from datetime import datetime
from scipy import ndimage
from PIL import Image
from libtiff import TIFF

def process_image(img,name_of_image):

    img_array = np.asarray(img)                                                         # CONVERT IMAGE TO ARRAY
    img_array8 = np.divide(img_array, 16)
    img_array8_1 = np.multiply(np.divide(img_array8, np.amax(img_array8)),250).astype('uint8')

    img8 = Image.fromarray(img_array8_1)                                                                      # CONVERT ARRAY TO IMAGE
    img8.save(name_of_image+'_8bit.tiff')                                                          # SAVE INDIVIDUAL FRAME 8 BIT
                    
    img8= cv2.imread(name_of_image+'_8bit.tiff',cv2.IMREAD_COLOR)                    
    gray=  cv2.cvtColor(img8, cv2.COLOR_BGR2GRAY)
                    #gray_blurred = cv2.blur(gray, (2, 2))
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

    return(dull_gray)   
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



def delete_guv_and_interior(name):

    image1 = cv2.imread(name,0)
    img_array = np.asarray(image1)        


    img_array8 = np.divide(img_array, 16)
    img_array8_1 = np.multiply(np.divide(img_array8, np.amax(img_array8)),250).astype('uint8')
    img8 = Image.fromarray(img_array8_1)
    img8.save('temp.tiff')

    name_8bit = name[:len(name)-5]+'_8bit.tiff'
    detected_circles = find_circle(name_8bit, parameter2, alpha, beta) 
    
    reference_points=[0.0,0.0,0.0]
    if detected_circles is not None:
        reference_points=detected_circles[0, :][0] 
        reference_points=np.uint16(np.around(reference_points))

    a, b, r = reference_points[0], reference_points[1], reference_points[2] 
    print(a,b,r)
    img8 = cv2.imread('temp.tiff', 0) 
    img8_contrasted = contrast_it_up(name_8bit, alpha, beta)
    img8_blur=cv2.blur(img8, (5, 5))

    ret3,th3 = cv2.threshold(img8_contrasted,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)                    
    contours, hierarchy = cv2.findContours(th3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    cv2.drawContours(image1, contours, -1, (175, 235, 0), thickness=cv2.FILLED)
    cv2.imwrite(name[:len(name)-5]+'_contours.tiff', image1)

    img16_as_array = np.array(Image.open(name), dtype=np.uint16)


    matrix = np.ones(img16_as_array.shape, dtype=bool)                                                                               # CREATE AN ARRAY OF X AND Y DIMENSION = image_size. EVERY POSITION HAS VALUE OF ONE                 
    matrix[a,b] = False    

    distance_from_center= ndimage.distance_transform_edt(matrix) 
    point_inside_GUV = np.where(distance_from_center<=r)


    array_useful_points = np.zeros(img16_as_array.shape, dtype=np.uint16)
    array_useful_points= img16_as_array

    for j in range(0,len(contours)):

        cnt = contours[j]
        x=cnt[:,:, :-1]
        y=cnt[:,:, -1:]

        x_pos,y_pos,w,h = cv2.boundingRect(cnt)
        area0 = cv2.contourArea(cnt)
        if area0>0:
                            
            for x_loc in range(x_pos, x_pos+w):
                for y_loc in range(y_pos, y_pos+h):
                    dist = cv2.pointPolygonTest(cnt,(x_loc,y_loc), True)
                    if dist>0:
                        
                        array_useful_points[y_loc,x_loc]=0

    
    array_useful_points[point_inside_GUV[1],point_inside_GUV[0]]=0

    tiff = TIFF.open(name[:len(name)-5]+'_contour_deleted.tiff', mode='w')
    tiff.write_image(array_useful_points)
    tiff.close()



parameter2,  alpha_initial, beta_initial, alpha, beta = 80, 1, 100, 0.8, 100
max_thickness=100         # SOME CRAZY LARGE DISTANCE AWAY FROM DETECTED CIRCLE
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
for file1 in os.listdir('.'):
    if file1.endswith('.tiff') and reference_name in file1 and 'contour' not in file1 and 'temp' not in file1 and '8bit' not in file1 : # ONLY ANALYSE THE 16 BIT VERSION OF THE TIFF FILES AND NOT THEIR MODIFIED VERSIONS. 
        


        for seg in range(0,len(file1.split('_'))):                                                  # GET THE MAGNIFICATION OF THE IMAGE IF INCLUDED IN THE FILE NAME
            if 'X' in file1.split('_')[seg] or 'x'  in file1.split('_')[seg] :
                zoom=(file1.split('_'))[seg][0]

        image,  reference_points, count, bad_detection =[], [0.0,0.0,0.0], 0, 0                     # DECLARE VARIABLES THAT WILL STORE THE GRAYSCALE IMAGES, CIRCLE DETAILS, PIXEL COUNT AND FLAG FOR QUALITY OF CIRCLES DETECTED 

        for i in range(0,number_of_channels):                                                       # LOOP OVER ALL THE CHANNELS


            if names[i]!='NA':                                                                      # IF THE CHANNEL IS A FLUORESCENT CHANNEL
                name=file1[:len(file1)-len(file1.split('_')[len(file1.split('_'))-1])] \
                +'C00'+str(i+1)+file1.split('_')[len(file1.split('_'))-1].split(reference_name)[1]  # GET THE NAME OF THE FILE CORRESPONDING TO THIS CHANNEL. 


                delete_guv_and_interior(name)                  # FIND CIRCLES PRESENT IN THE IMAGE

