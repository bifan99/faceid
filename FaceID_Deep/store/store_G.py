# This script will detect faces via your webcam.
# Tested with OpenCV3

import cv2
import numpy as np

cap = cv2.VideoCapture(0)


net = cv2.dnn.readNetFromCaffe("../dnn/deploy.prototxt.txt", "../dnn/res10_300x300_ssd_iter_140000.caffemodel")

index = 0
while(True):
	
    null, frame = cap.read()

    cv2.imshow('Store FaceID data..', frame)
    if cv2.waitKey(1) & 0xFF == ord('g'):
        

        (h, w) = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300,300)), 1.0, (300, 300), (103.93, 116.77, 123.68))

	
        net.setInput(blob)
        detections = net.forward()
	
        confidence = detections[0, 0, 0, 2]
        
        if confidence > 0.50:
            box = detections[0, 0, 0, 3:7] * np.array([w, h, w, h]) #0,0,X,3:7
    
            (startX, startY, endX, endY) = box.astype("int")
   
        
            cv2.imwrite("./guglielmo/"+str(index)+".jpg", frame[startY:endY, startX:endX]) #Prima y, Poi x
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
            index = index + 1
            cv2.imshow('Stored Pic!', frame[startY:endY, startX:endX])
            cv2.waitKey(600)
    
            


	
	
    


cap.release()
cv2.destroyAllWindows()
