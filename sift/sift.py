import numpy as np
import cv2 
 
img = cv2.imread('img.jpg')
gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
 
sift = cv2.SIFT_create()
kp = sift.detect(gray,None)
 
img=cv2.drawKeypoints(gray,kp,img)
cv2.imshow('sift_keypoints', img)
cv2.waitKey(0)  # Espera hasta que se presione una tecla
cv2.destroyAllWindows()  # Cierra todas las ventanas

cv2.imwrite('sift_keypoints.jpg',img)