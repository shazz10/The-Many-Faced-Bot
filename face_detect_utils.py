from keras import backend as K
import numpy as np
import os
import cv2
import tensorflow as tf
from multiprocessing.dummy import Pool
import glob

detect = True

pad = 5

def triplet_loss(y_true, y_pred, alpha = 0.3):
    
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis=-1)
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis=-1)
    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))
    
    return loss

def prepare_faces(MFmodel):
    faces = {}

    for file in glob.glob("faces/*"):
        identity = os.path.splitext(os.path.basename(file))[0]
        img = cv2.imread(file, 1)
        faces[identity] = img_to_encoding(img, MFmodel)


    return faces

def many_faced_bot(faces,MFmodel):

    global detect

    cv2.namedWindow("Many-Faced-Bot")
    vc = cv2.VideoCapture(0)

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    
    while vc.isOpened():
        _, frame = vc.read()
        img = frame

        if detect:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            boxes = face_cascade.detectMultiScale(gray, 1.3, 5)

            ids = []
            for (x, y, w, h) in boxes:
                x1 = x-pad
                y1 = y-pad
                x2 = x+w+pad
                y2 = y+h+pad

                img = cv2.rectangle(frame,(x1, y1),(x2, y2),(255,0,0),2)
                
                identity = cut_face(frame, x1, y1, x2, y2, faces, MFmodel)

                if identity is not None:
                    ids.append(identity)

            if ids != []:
                print("hi",ids)
                detect = False

        else :
            break
        key = cv2.waitKey(100)
        cv2.imshow("Many-Faced-Bot", img)

        if key == 27: # exit on ESC
            break
    cv2.destroyWindow("Many-Faced-Bot")

def cut_face(frame, x1, y1, x2, y2, faces, MFmodel):

    h, w, ch = frame.shape
    x1=x1+2
    y1=y1+2
    x2=x2-2
    y2=y2-2
    cropped_image = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
    return find_face(cropped_image, faces, MFmodel)

def find_face(image, faces, MFmodel):
    encoding = img_to_encoding(image, MFmodel)
    
    min_dist = 100
    identity = None
    
    for (name, embed) in faces.items():
        
        dist = np.linalg.norm(embed - encoding)

        print('distance for %s is %s' %(name, dist))

        if dist < min_dist:
            min_dist = dist
            identity = name
    
    if min_dist > 0.52:
        return None
    else:
        return str(identity)


def img_to_encoding(image, model):
    #cv2.imwrite("test1.jpg",image)
    image = cv2.resize(image, (96, 96))
    #cv2.imwrite("test2.jpg",image) 
    img = image[...,::-1]
    #cv2.imwrite("test3.jpg",img)
    img = np.around(np.transpose(img, (2,0,1))/255.0, decimals=12)
    x_train = np.array([img])
    embedding = model.predict_on_batch(x_train)
    return embedding