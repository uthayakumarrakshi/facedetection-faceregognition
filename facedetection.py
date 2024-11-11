import cv2
trainedDataset=cv2.CascadeClassifier("model/haarcascade_frontalface_default.xml")
img=cv2.imread("E:/facedetection/80847bba637cd0a71df4dd15b916c17d.jpg")
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
face=trainedDataset.detectMultiScale(gray)
print(face)
for x,y,w,h in face:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),4)
cv2.imshow('image',img)
cv2.waitKey()
