# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 14:49:07 2018

@author: mhhm2
"""
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

from moviepy.editor import VideoFileClip
from IPython.display import HTML

from common import *

DEBUG = 0
#TEST = "VIDEO_TEST"

def pipeline(image):
    #reading in an image
    #image = mpimg.imread(img_path)

    ysize = image.shape[0]
    xsize = image.shape[1]
    
    #getting the grayscal image
    gray = grayscale(image)

    
    # Define a kernel size and apply Gaussian smoothing
    kernel_size = 11
    blur_gray = gaussian_blur(gray, kernel_size)

    # Define our parameters for Canny and apply
    low_threshold = 50
    high_threshold = 100
    edges = canny(blur_gray, low_threshold, high_threshold)

    #Mask the edges
    cut = 20
    p1 = (int(xsize/cut), int(ysize))
    p2 = (int(xsize/2-xsize/cut), int(ysize/2+2*ysize/cut))
    p3 = (int(xsize/2+xsize/cut), int(ysize/2+2*ysize/cut))
    p4 = (int(xsize-xsize/cut), int(ysize))
    vertices = np.array([[p1,  p2,  p3, p4]], dtype=np.int32)
    masked_edges = region_of_interest(edges, vertices)

    
    #Hough transform --> get the lines
    # Define the Hough transform parameters
    # Make a blank the same size as our image to draw on
    rho = 6
    theta = (np.pi / 180)
    threshold = 75
    min_line_length = 20
    max_line_gap = 100
    line_img = hough_lines(masked_edges, rho, theta, threshold, min_line_length, max_line_gap)
    

    
    #fig5 = plt.figure(5)
    #plt.imshow(line_img)  # if you wanted to show a single color channel image called 'gray', for example, call as plt.imshow(gray, cmap='gray')
    if DEBUG != 0:
        #lines = [np.concatenate((vertices[0][0] , vertices[0][1]), axis=0)]
        lines = [[list(p1+p2), list(p2+p3),list(p3+p4),list(p1+p4)]]
        print (lines)
        draw_lines(line_img, lines, color=[0, 255, 255], thickness=5)
    

    #add the lines to the originalimage
    final = weighted_img(line_img, image)

    return final
    
    


def test_video(vid_path):
    white_output = main_dir+"/"+vid_out_dir+"/"+os.path.splitext(os.path.basename(vid_path))[0]+"_output.mp4"
    #white_output = os.path.splitext(os.path.basename(vid_path))[0]+"_ouput.mp4"
    print(vid_path)
    clip1 = VideoFileClip(vid_path)
    video_clip = clip1.fl_image(pipeline)
    video_clip.write_videofile(white_output, audio=False)
    video_clip.reader.close()
    video_clip.audio.reader.close_proc()
    
    
def test_videos():
    videos = find_files(main_dir+"/test_videos", ".mp4")
    print(videos)
    for vid_path in videos:
        test_video(vid_path)
    