from PIL import Image
import cv2
import numpy
import sys
import time

img1 = None
img2 = None
global db1
global db2
print("Initiating")
vid1=cv2.VideoCapture(0)
vid2=cv2.VideoCapture(1)
print("Initiating the camera")
while True:
    print("In reading")
    rval1, im1 = vid1.read()
    rval2, im2 = vid2.read()
    print("read")
    import config
    print("db1:",config.db1)
    print("db2:",config.db2)
    print("nc1:",config.nc1)
    print("nc2:",config.nc2)

    if(config.cameraoneon==True and config.cameratwoon==True):
        #im1=cv2.imread('./speakerone.jpg')
        #im2=cv2.imread('./speakertwo.jpg')
        images_1_2_h = np.hstack((im1, im2))
        cv2.imshow('Video',images_1_2_h)
        #print("thisconditionworks")
        i=0
    if(config.cameraoneon==True and config.cameratwoon==False):
        #im1=cv2.imread('./speakerone.jpg')
        cv2.imshow('Video',im1)
        #print("thisconditionworkstwo")
        i=1
    if(config.cameratwoon==True and config.cameraoneon==False):
        #im2=cv2.imread('./speakertwo.jpg')
        #print("thisconditionworksthree")
        cv2.imshow('Video',im2)
        i=2
    if(config.cameraoneon==False and config.cameratwoon==False):
        im3=cv2.imread('./noone.jpg')
        #print("noworks")
        cv2.imshow('Video',im3)
        i=3

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
