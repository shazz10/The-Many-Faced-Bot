import add_face
import face_detect
import os
MFmodel=face_detect.load_model()
while(1):
	print("Enter 1 to add face, 2 to detect face, 3 to exit")
	f=int(input())
	if f==1:
		print("Enter name")
		name=input()
		add_face.add_face(name)
	elif f==2:
		face_detect.recognise_face(MFmodel)
	else:
		os.system("pkill -9 python3") #to kill all process note it will kill all python3 processes
		break