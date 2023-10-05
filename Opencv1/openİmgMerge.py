import cv2 as cv
import numpy as np

foto=cv.imread("img/DSCF9068.JPG")
cv.imshow("foto",foto)

# Fotografları yatay olarak birleştirir

yatay=np.hstack((foto,foto))
yatay=cv.resize(yatay,(500,500))

# Fotoları dikey olarak birleştirir

dikey=np.vstack((foto,foto))
dikey=cv.resize(dikey,(500,500))


cv.imshow("yetay",yatay)
cv.imshow("Dikey",dikey)

cv.waitKey(0)
cv.destroyAllWindows("foto")