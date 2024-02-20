# Load imports
import os 
import cv2
import numpy as np
import align
import filters



'''
function get_images() definition
Parameter, image_directory, is the directory 
holding the images
'''

def get_images(image_directory):
    X = []
    y = []
    extensions = ('jpg','png','gif')
    
    '''
    Each subject has their own folder with their
    images. The following line lists the names
    of the subfolders within image_directory.
    '''
    subfolders = os.listdir(image_directory)
    for subfolder in subfolders:
        print("Loading images in %s" % subfolder)
        if os.path.isdir(os.path.join(image_directory, subfolder)): # only load directories
            subfolder_files = os.listdir(
                    os.path.join(image_directory, subfolder)
                    )
            for file in subfolder_files:
                if file.endswith(extensions): # grab images only
                    # read the image using openCV
                    new_img = cv2.imread(os.path.join(image_directory, subfolder, file))
                    # cv2.imshow('ori', new_img)

                    #opening morphological transformation - not used
                    # kernelSizes = [(3, 3)]
                    # # loop over the kernels sizes
                    # for kernelSize in kernelSizes:
                    #     # construct a rectangular kernel from the current size and then
                    #     # apply an "opening" operation
                    #     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernelSize)
                    #     opening = cv2.morphologyEx(new_img, cv2.MORPH_OPEN, kernel)
                    #     cv2.imshow("Opening: ({}, {})".format(
                    #         kernelSize[0], kernelSize[1]), opening)


                    # rotate - use this for Experiment 2
                    # new_img = align.rotate(
                    #     os.path.join(image_directory, subfolder, file)
                    # )
                    # cv2.imshow('rotated', new_img)

                    # Use the below specific filters for Experiment 3
                    # sharpen
                    # new_img = filters.sharpen(new_img)
                    # cv2.imshow('Sharpened', new_img)

                    # Unsharp masking 5x5
                    # new_img = filters.unsharp_masking55(new_img)
                    # cv2.imshow('masked', new_img)

                    # Gaussian Blur 3x3
                    new_img = filters.gaussianBlur(new_img)
                    # cv2.imshow('blurred', new_img)

                    # normalize
                    norm_img = np.zeros((800,800))
                    new_img = cv2.normalize(new_img,  norm_img, 0, 255, cv2.NORM_MINMAX)

                    # resize the image
                    width = 100
                    height = 100
                    dim = (width, height)
                    new_img = cv2.resize(new_img, dim)
                    final_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
                    # cv2.imshow('gray',final_img)
                    # cv2.waitKey(0)

                    # add the resized image to a list X
                    X.append(final_img)
                    # add the image's label to a list y
                    y.append(subfolder)
    
    print("All images are loaded")     
    # return the images and their labels      
    return X, y
                    
                
            