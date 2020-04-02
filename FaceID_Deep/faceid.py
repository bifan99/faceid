


import cv2
import numpy as np
from gi.repository import Gdk
import time
import keyboard
import os
import subprocess as sub

global cap2

num_read = open('./store/num.ini','r')
num = int(num_read.read())

def PixelAt(x, y):
     w = Gdk.get_default_root_window()
     pb = Gdk.pixbuf_get_from_window(w, x, y, 1, 1)
     return pb.get_pixels()



def face_detection(frame):
    if np.shape(frame) != ():
        image_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        haar_classifier = cv2.CascadeClassifier('./dnn/face_model.xml')
        face = haar_classifier.detectMultiScale(image_gray, scaleFactor=1.13, minNeighbors=3, minSize=(120,120))
        if len(face) == 0:
            return (1,1)
        (x,y,w,h) = face[0]
        return frame[y:y+w, x:x+h]
    else:
        return (1,1)


def embedding_detection(pic):
    blob = cv2.dnn.blobFromImage(cv2.resize(pic, (224,224)), 1.0, (224, 224), (104, 117, 123))
    embedding.setInput(blob)
    detections = embedding.forward("fc8")
    return detections


print("[Loading neural networks for deep learning..]")

embedding = cv2.dnn.readNetFromCaffe("./dnn/VGG_FACE_deploy.prototxt","./dnn/VGG_FACE.caffemodel")

print("[Loading embeddings of ID Faces..]")

embeddings = np.empty(num+1, dtype=object)
for i in range(0,num+1):
    embeddings[i] = np.loadtxt("./store/guglielmo_embeddings/"+str(i)+".emb", dtype=np.float)
    print(str(embeddings[i]))


print("[FaceID is active!]")
while(True):
  
    if (keyboard.is_pressed('f2')):
        print("Pressione rilevata!")
        cap2 = cv2.VideoCapture(0)
        null, web = cap2.read()
        cap2.release()
        yourface = face_detection(web)
        if(yourface != (1,1)):
            yourembeddings = embedding_detection(yourface)
            L2Norm_array = np.empty(num+1, dtype=object)
            for i in range(0,num+1):
                L2Norm_array[i] = np.linalg.norm(yourembeddings - embeddings[i].astype(float))
                print("Frame - "+str(i)+".jpg L2Norm: ",str(L2Norm_array[i]))

            L2NormMin = min(L2Norm_array)
            print("L2 Norm MINIMO: ",str(L2NormMin))       

            if (L2NormMin <= 70):
                p = sub.Popen(["sudo", "loginctl","unlock-sessions"], stdout = sub.PIPE, stderr = sub.PIPE)
                print("Sblocco eseguito!")
                   
                
          
    time.sleep(0.3)
        
        

    
    
    
