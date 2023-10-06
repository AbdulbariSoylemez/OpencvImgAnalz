import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

blueandred=cv.imread("img/blueandred.jpg")
blueandred=cv.cvtColor(blueandred,cv.COLOR_BGR2RGB)

plt.figure(),plt.imshow(blueandred),plt.title("İmg")

print(blueandred.shape)

histSize = [256] 
channels = [0]  
mask = None       
img_hist = cv.calcHist([blueandred], channels, mask, histSize, ranges=[0, 256])

plt.figure(),plt.plot(img_hist),plt.title("İmg in histogram dağılımı") 

# Fotograf üzerinde renk dağılımlarını gösterelim 
color=("b","g","r")
plt.figure()
for i ,c in enumerate(color):
     hist=cv.calcHist([blueandred], channels = [i] , mask = None , histSize=[256], ranges=[0, 256])
     plt.plot(hist,color=c)

# Farkılı bir resim üzerinde maskeleme işlemi yaparak renk grafiğini oluşturalım 


gök=cv.imread("img/DSCF8078.JPG")
gök=cv.cvtColor(gök,cv.COLOR_BGR2RGB)
plt.figure(),plt.imshow(gök),plt.title("gökyüzü")

mask=np.zeros(gök.shape[:2],np.uint8)
mask[1500:2000,2000:2600]=255


masked=cv.bitwise_and(gök,gök,mask=mask)
plt.figure(),plt.imshow(masked),plt.title("maskeleme işlemi")


masked = cv.calcHist([masked], channels, mask, histSize, ranges=[0, 256])
plt.figure(),plt.plot(masked),plt.title("masked in histogram dağılımı")










