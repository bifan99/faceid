# This script will detect faces via your webcam.
# Tested with OpenCV3

import cv2
import numpy as np

num_read = open('num.ini','r')
num = int(num_read.read())

def embedding_detection(pic):
    if (np.shape(pic) != ()):
        blob = cv2.dnn.blobFromImage(cv2.resize(pic, (224,224)), 1.0, (224, 224), (104, 117, 123))
        embedding.setInput(blob)
        detections = embedding.forward("fc8")
        return detections
    else:
        return (1,1)

print("[Loading neural networks for deep learning..]")

embedding = cv2.dnn.readNetFromCaffe("../dnn/VGG_FACE_deploy.prototxt","../dnn/VGG_FACE.caffemodel")

print("[Storing embeddings of ID Faces..]")

embeddings = np.empty(num+1, dtype=object)
for i in range(0,num+1):
    store = cv2.imread("./guglielmo/"+str(i)+".jpg")
    embeddings[i] = embedding_detection(store)   
    print(str(embeddings[i]))
    if (embeddings[i] != (1,1)):
        np.savetxt('./guglielmo_embeddings/'+str(i)+'.emb', embeddings[i], fmt='%f') 

print("[Embeddings have bene saved successfully!]")


