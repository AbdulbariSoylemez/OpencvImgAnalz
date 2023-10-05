import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import warnings 
warnings.filterwarnings("ignore")

Happy=cv.imread("img/happy.jpg")
Happy=cv.cvtColor(Happy,cv.COLOR_BGR2RGB)
plt.figure(),plt.imshow(Happy,cmap="gray"),plt.title("Orjinal"),plt.axis("off")

"""
Erozyon (Erosion) işlemi ,  ön taraftaki nesnenin sınırlarını aşındırmayı sağlar.

"""

kutu = np.ones((5,5),dtype=np.uint8)

erozyon = cv.erode(Happy,kutu,iterations=3) # “iterations” değeri, görüntüye kaç kez erozyon uygulanacağını belirler
plt.figure(),plt.imshow(erozyon,cmap="gray"),plt.title("Erozyon"),plt.axis("off")


"""
Genişleme (Dilation) işlemi , Genişleme erozyonun tam tersidir
"""

kutu=np.ones((5,5),dtype=np.uint8)
genişleme=cv.dilate(Happy,kutu,iterations=3)
plt.figure(),plt.imshow(genişleme,cmap="gray"),plt.title("Genişleme"),plt.axis("off") 

"""
Açılma (Opening), Açılma işlemi, erozyon ve genişlemenin peş peşe kullanılmasıdır.
"""
Happy=cv.imread("img/happy.jpg",0)
kutu=np.ones((5,5),dtype=np.uint8)
whiteNoise = np.random.randint(0, 2, size=Happy.shape[:2]) # Beyaz gürültüyü oluşturuyoruz 
whiteNoise=whiteNoise*255
whiteNoise = cv.resize(whiteNoise, (Happy.shape[1], Happy.shape[0]))
noise_img = whiteNoise + Happy
plt.figure(),plt.imshow(noise_img,cmap="gray"),plt.title("NoiseİMG"),plt.axis("off") 

Açılma = cv.morphologyEx(noise_img.astype(np.float32), cv.MORPH_OPEN, kutu) #  bu çarpık görüntü üzerine morfolojik açılma işlemi uyguluyoruz. Açılma işlemi, görüntüdeki küçük nesneleri veya gürültüyü gidermek için kullanılır.
plt.figure(),plt.imshow(Açılma,cmap="gray"),plt.title("Açılma"),plt.axis("off") 


"""
Kapatma (Closing), Açmanın tam tersidir ,Ön plandaki nesnelerin içindeki küçük delikleri ve ya nesne üzerindeki küçük siyah noktaları kapatmak için kullanışlıdır.

"""

kutu = np.ones((5,5),dtype=np.uint8)

blackNoise = np.random.randint(0,2,size=Happy.shape[:2])
blackNoise = blackNoise*-255
noise_img = blackNoise + Happy
noise_img[noise_img <=-245] = 0

closing = cv.morphologyEx(noise_img.astype(np.float32),cv.MORPH_CLOSE,kutu)
plt.figure(),plt.imshow(closing,cmap="gray"),plt.title("closing"),plt.axis("off") 



"""
Gradyan (Gradient),bir görüntünün genişlemesi ile erozyona uğraması arasındaki farktır. Bu sayede görüntünün kenarlarının bulunmasını sağlayabiliriz.
"""


gradient = cv.morphologyEx(Happy,cv.MORPH_GRADIENT,kutu)
plt.figure(),plt.imshow(gradient,cmap="gray"),plt.title("gradient"),plt.axis("off") 

plt.show()















