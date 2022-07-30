%number_of_channels=3;
%command = sprintf('/Library/Frameworks/Python.framework/Versions/3.7/bin/python3 ../breakup_multiple_images.py %d', number_of_channels)
%[status,cmdout] = system(command)
% performs autocorrelation on all tiff files in directory
myDir = uigetdir;
myFiles = dir(fullfile(myDir,'*.tiff'));
for k = 1:length(myFiles)
    baseFileName = myFiles(k).name;
    write_name = extractBetween(baseFileName,1,strlength(baseFileName)-5) +"_autocor.txt";
    im = imread(baseFileName);
    im_autocor= normxcorr2(im,im);
    writematrix(im_autocor, write_name ,'Delimiter','tab');
    for r = 1:length(myFiles)
        baseFileName_check = myFiles(r).name;
        if strcmp(baseFileName_check,baseFileName) == 0
            im2 = imread(baseFileName_check);
            im_crosscor= normxcorr2(im2,im);
            write_name_cross = extractBetween(baseFileName,1,strlength(baseFileName)-5)+"_"+extractBetween(baseFileName_check,1,strlength(baseFileName_check)-5)+"_crosscor.txt";
            writematrix(im_crosscor, write_name_cross ,'Delimiter','tab');
        end
    end
end
command = '/Library/Frameworks/Python.framework/Versions/3.7/bin/python3 ~/Desktop/radial_average.py'
[status,cmdout] = system(command)
