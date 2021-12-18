import cv2
import numpy as np

cap0 = cv2.VideoCapture(0)
cap1 = cv2.VideoCapture(1)

while(True):
        import config
        ret0,im1 = cap0.read()
        ret1,im2 = cap1.read()
        if(config.cameraoneon==True and config.cameratwoon==True):
            images_1_2_h = np.hstack((im1, im2))
            cv2.imshow('Video',images_1_2_h)
            i=0
        if(config.cameraoneon==True and config.cameratwoon==False):
            cv2.imshow('Video',im1)
            i=1
        if(config.cameratwoon==True and config.cameraoneon==False):
            cv2.imshow('Video',im2)
            i=2
        if(config.cameraoneon==False and config.cameratwoon==False):
            im3=cv2.imread('./noone.jpg')
            cv2.imshow('Video',im3)
            i=3
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

#cap0.release()
#cap1.release()
#cv2.destroyAllWindows()

