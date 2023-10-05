import cv2 as cv
import matplotlib.pyplot as plt

GrandFoto=cv.imread("img/Otopark.jpg",0)
plt.figure(),plt.imshow(GrandFoto,cmap="gray"),plt.title("GrandFoto"),plt.axis("off")


# x gradyan

Sobelx=cv.Sobel(GrandFoto,ddepth=cv.CV_16S,dx=1,dy=0,ksize=5)
plt.figure(),plt.imshow(Sobelx,cmap="gray"),plt.title("Sobel x"),plt.axis("off")



# y gradyan

Sobely=cv.Sobel(GrandFoto,ddepth=cv.CV_16S,dx=0,dy=1,ksize=5)
plt.figure(),plt.imshow(Sobely,cmap="gray"),plt.title("Sobel y"),plt.axis("off")


# laplacian 

laplacian = cv.Laplacian(GrandFoto, ddepth=cv.CV_64F)
plt.figure(),plt.imshow(laplacian),plt.title("laplacian"),plt.axis("off")

# Gradyanları Toplama
topla=Sobelx + Sobely
plt.figure(),plt.imshow(topla,cmap="gray"),plt.title("topla"),plt.axis("off")
plt.show()

