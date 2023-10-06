import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np 

cat_dog=cv.imread("img/dog_cat.jpg")
cat_dog=cv.cvtColor(cat_dog,cv.COLOR_RGB2BGR)

plt.figure(),plt.imshow(cat_dog),plt.title("dog and cat {}".format(str(cat_dog.shape)))

# Yeniden boyutlandıralım 
new_width = int(cat_dog.shape[1] * 0.8)
new_height = int(cat_dog.shape[0] * 0.8)
cat_dog=cv.resize(cat_dog,(new_width,new_height))
plt.figure(),plt.imshow(cat_dog),plt.title("dog and cat {}".format(str(cat_dog.shape)))

# Resim üzerinde yazı yazma
cat_dog=cv.putText(cat_dog," Cat and Dog ",(25,50),1,2,(100,100,100),1)
plt.figure(),plt.imshow(cat_dog),plt.title("İsimleri   {}".format(str(cat_dog.shape)))


# Resmi siyah beyaz yapma 
cat_dog_gray = cv.cvtColor(cat_dog, cv.COLOR_BGR2GRAY)
cv.imshow("zero",cat_dog_gray)


# Fotografımıza threshold "THRESH_BINARY" uygulayalım 50 THRESHHOLD değeri üzerindekileri beyaz yap altındakileri siyah yap 
ret,thres=cv.threshold(cat_dog_gray,thresh=100,maxval=255,type=cv.THRESH_BINARY)
cv.imshow("thresholdlu fotograf ",thres)

# Orjinal fotografımızı tekrar yazalım 
cat_dog=cv.imread("img/dog_cat.jpg")

bulanık=cv.GaussianBlur(cat_dog,(5,5),7)
cv.imshow("Bulanık",bulanık)



# Orjinal resimimize laplacian gradyan  uygulayalım 

kutu=np.ones((5,5),np.uint8)

gradyan=cv.morphologyEx(cat_dog,cv.MORPH_GRADIENT,kutu)

cv.imshow("Gradiyan fotograf",gradyan)

# Orjinal resimimizin Histogram grafiğini çizelim 
cat_dog=cv.imread("img/dog_cat.jpg")

color=("b","g","r")
plt.figure()
for i ,c in enumerate(color):
     hist=cv.calcHist([cat_dog], channels = [i] , mask = None , histSize=[256], ranges=[0, 256])
     plt.plot(hist,color=c)


plt.show()