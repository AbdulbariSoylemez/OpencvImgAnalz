import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt 

foto=cv.imread("img/sınav.jpg")
foto=cv.cvtColor(foto,cv.COLOR_BGR2GRAY)
foto=cv.resize(foto,(600,800))
plt.figure()
plt.imshow(foto,cmap="gray")
plt.axis("off")


ret, thresh1 = cv.threshold(foto, 120, 255, cv.THRESH_BINARY) 
ret, thresh2 = cv.threshold(foto, 120, 255, cv.THRESH_BINARY_INV) 
ret, thresh3 = cv.threshold(foto, 120, 255, cv.THRESH_TRUNC) 
ret, thresh4 = cv.threshold(foto, 120, 255, cv.THRESH_TOZERO) 
ret, thresh5 = cv.threshold(foto, 120, 255, cv.THRESH_TOZERO_INV) 

#thresh6 = cv.adaptiveThreshold(foto, 255, cv.ADAPTIVE_THRESH_MEAN_C, 
                                          #cv.THRESH_BINARY, 199, 5) 
thresh6 = cv.adaptiveThreshold(foto, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv.THRESH_BINARY, 199, 5) 
  
thresh7 = cv.adaptiveThreshold(foto, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv.THRESH_BINARY, 149, 10) 
  

  
def show1(thresh1, thresh2, thresh3, thresh4, thresh5):
    cv.imshow('Binary Threshold', thresh1) 
    cv.imshow('Binary Threshold Inverted', thresh2) 
    cv.imshow('Truncated Threshold', thresh3) 
    cv.imshow('Set to 0', thresh4) 
    cv.imshow('Set to 0 Inverted', thresh5)

#show1(thresh1, thresh2, thresh3, thresh4, thresh5) 

def show2(thresh6, thresh7):
    cv.imshow('Adaptive Mean', thresh6) 
    cv.imshow('Adaptive Gaussian', thresh7)

show2(thresh6, thresh7) 

if cv.waitKey(0) & 0XFF==ord("q"):
     cv.destroyAllWindows()
