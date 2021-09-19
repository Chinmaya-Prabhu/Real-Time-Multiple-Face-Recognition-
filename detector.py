import numpy as np
import cv2

detector= cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
rec=cv2.face.LBPHFaceRecognizer_create();
rec.read("trainingData.yml")
id=0
font=cv2.FONT_HERSHEY_COMPLEX_SMALL
while(True):
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        id,conf=rec.predict(gray[y:y+h,x:x+w])
        if(conf<50):    
            cv2.putText(img,str(id),(x+10,y+10),font,1,(255,0,0),2);
        else:
            cv2.putText(img,"unknown",(x+10,y+10),font,1,(255,0,0),2);
    cv2.imshow('Face',img)
    if (cv2.waitKey(1) & 0xFF == ord('q')):
        break;
cap.release()
cv2.destroyAllWindows()
