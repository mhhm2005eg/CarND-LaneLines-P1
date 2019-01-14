import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
from common import *
from test_images import test_images, test_image
from test_videos import test_videos, test_video

if not os.path.exists(img_out_dir):
    os.makedirs(img_out_dir)
    
if not os.path.exists(vid_out_dir):
    os.makedirs(vid_out_dir)
    
def main():
   #test_images()
   #test_image("./test_images/whiteCarLaneSwitch.jpg")
   test_videos()
   #test_video(".\\test_videos\\challenge.mp4");
    
if __name__ == "__main__":
    #lane_color()
    #lane_region()
    #lane_region_color()
    #canny_test1()
    main()