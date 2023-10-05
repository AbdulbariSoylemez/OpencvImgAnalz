import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


foto1=cv.imread("img/DSCF9068.JPG")
foto1=cv.cvtColor(foto1,cv.COLOR_BGR2RGB)
foto2=cv.imread("img/DSCF8078.JPG")
foto2=cv.cvtColor(foto2,cv.COLOR_BGR2RGB)

# Birleştirme işlemi yapılacağı için fotografların boyutu aynı olması lazım 


Fheight = foto1.shape[0]  # Yükseklik
Fwidth = foto1.shape[1]   # Genişlik


foto2=cv.resize(foto2,(Fwidth,Fheight))


plt.figure()
plt.imshow(foto1)

plt.figure()
plt.imshow(foto2)

# Fotografalrı üst üste (Fotografları aynı karede iç içe gösterme işlemi ) birleştirme işlemi

birleştirme=cv.addWeighted(src1=foto1,alpha=0.5,src2=foto2,beta=0.5,gamma=0.0)

plt.figure()
plt.imshow(birleştirme)
plt.show()


