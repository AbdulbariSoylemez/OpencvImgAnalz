import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# üç farklı bulanıklaştırma işlemi var bunlar 
# 1)ortalama bulanıklaştırma 2)gaussian bulanıklaştırma 3)Medyan bulanıklaştırma

Bulanık=cv.imread("img/gelBuraya.jpg")
Bulanık=cv.cvtColor(Bulanık,cv.COLOR_BGR2RGB)

plt.figure(),plt.imshow(Bulanık),plt.axis("off"),plt.title("orjinal")

"""
Ortalama bulanıkkaştırma 
"""

ortalama=cv.blur(Bulanık,(10,10)) # (3,3) değerleri ile oyanayarak daha da bulanıklaştıram oranını artırabilirsiniz 
plt.figure(),plt.imshow(ortalama),plt.axis("off"),plt.title("Ortalama bulanıklaştırma")


"""
gaussian Bulanıklaştırma
"""

gaussian=cv.GaussianBlur(Bulanık,ksize=(15,15),sigmaX=7)
plt.figure(),plt.imshow(gaussian),plt.axis("off"),plt.title("Gaussian bulanıklaştırma")


"""
MedianBlur Bulanıklaştırma
"""


MedianBulur=cv.medianBlur(Bulanık,ksize=(15))

plt.figure(),plt.imshow(MedianBulur),plt.axis("off"),plt.title("MedianBlur bulanıklaştırma"),plt.show()



