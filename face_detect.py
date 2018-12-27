from keras import backend as K
K.set_image_data_format('channels_first')
import cv2
import os
import glob
import numpy as np
import tensorflow as tf
from keras.models import model_from_yaml
from face_detect_utils import *


#Load Pre-Trained Model
def load_model():
	yaml_file = open('model/many_faced_model.yaml', 'r')
	loaded_model_yaml = yaml_file.read()
	yaml_file.close()
	MFmodel = model_from_yaml(loaded_model_yaml)
	MFmodel.load_weights("model/many_faced_weights.h5")
	print("Loaded model from disk")
	MFmodel.compile(optimizer = 'adam', loss = triplet_loss, metrics = ['accuracy'])
	return MFmodel

def recognise_face(MFmodel):
	faces=prepare_faces(MFmodel)  #faces are prepared from database
	many_faced_bot(faces,MFmodel) #here we pass the-many-faced-bot list of faces and the model to be used