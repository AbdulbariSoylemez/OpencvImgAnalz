import cv2 as cv

cap=cv.VideoCapture(0)

width=int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
height=int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
print(width,height)


while True:
    ret , frame=cap.read()
    frame=cv.flip(frame,1)
    frame=cv.resize(frame,(500,500))
    if not ret:
        break

    cv.imshow("Frame",frame)

    if cv.waitKey(1) & 0xFF==ord("q"):
        break

cap.release()
cv.destroyAllWindows()

