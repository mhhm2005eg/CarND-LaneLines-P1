# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 00:05:04 2018

@author: mhhm2
"""
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
from common import *



def test_image(img_path):
    #reading in an image
    image = mpimg.imread(img_path)

    ysize = image.shape[0]
    xsize = image.shape[1]
    
    #printing out some stats and plotting
    print('This image is:', type(image), 'with dimensions:', image.shape)
    
    #getting the grayscal image
    gray = grayscale(image)
    fig1 = plt.figure(1)
    plt.imshow(gray, cmap='gray')
    
    # Define a kernel size and apply Gaussian smoothing
    kernel_size = 11
    blur_gray = gaussian_blur(gray, kernel_size)
    fig2 = plt.figure(2)
    plt.imshow(blur_gray, cmap='gray')
    
    # Define our parameters for Canny and apply
    low_threshold = 50
    high_threshold = 100
    edges = canny(blur_gray, low_threshold, high_threshold)
    fig3 = plt.figure(3)
    plt.imshow(edges, cmap='gray')
    
    #Mask the edges
    cut = 20
    p1 = (int(xsize/cut), int(ysize))
    p2 = (int(xsize/2-xsize/cut), int(ysize/2+2*ysize/cut))
    p3 = (int(xsize/2+xsize/cut), int(ysize/2+2*ysize/cut))
    p4 = (int(xsize-xsize/cut), int(ysize))
    vertices = np.array([[p1,  p2,  p3, p4]], dtype=np.int32)
    masked_edges = region_of_interest(edges, vertices)
    fig4 = plt.figure(4)
    plt.imshow(masked_edges, cmap='gray')
    
    
    #Hough transform --> get the lines
    # Define the Hough transform parameters
    # Make a blank the same size as our image to draw on
    rho = 6
    theta = (np.pi / 180)
    threshold = 75
    min_line_length = 10
    max_line_gap = 10
    line_img = hough_lines(masked_edges, rho, theta, threshold, min_line_length, max_line_gap)
    

    
    fig5 = plt.figure(5)
    plt.imshow(line_img)  # if you wanted to show a single color channel image called 'gray', for example, call as plt.imshow(gray, cmap='gray')
    if DEBUG != 0:
        #lines = [np.concatenate((vertices[0][0] , vertices[0][1]), axis=0)]
        lines = [[list(p1+p2), list(p2+p3),list(p3+p4),list(p1+p4)]]
        print (lines)
        draw_lines(line_img, lines, color=[0, 255, 255], thickness=5)
    
    #Fill area inbetween the lines
    #getting the grayscal image
    #line_img_gray = grayscale(line_img)
    #kernel_size = 3
    #line_img_blur_gray = gaussian_blur(line_img_gray, kernel_size)
    #line_img_blur_rgb = conver_2_RGB(line_img_blur_gray)
    #fig10 = plt.figure(10)
    #plt.imshow(line_img_blur_rgb)
    
    # Create a "color" binary image to combine with line image
    #color_edges = np.dstack((line_img, line_img, line_img))
    
    #add the lines to the originalimage
    final = weighted_img(line_img, image)
    fig6 = plt.figure(6)
    plt.imshow(final)  # if you wanted to show a single color channel image called 'gray', for example, call as plt.imshow(gray, cmap='gray')
    #cv2.imwrite()./test_images_output/
    #print (os.path.splitext(img_path)[0])
    save_image(os.path.splitext(os.path.basename(img_path))[0])
    
    
def test_images():
    images = find_files(main_dir+"/test_images", ".jpg")
    print(images)
    for img_path in images:
        test_image(img_path)