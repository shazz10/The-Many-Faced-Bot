import cv2

def add_face(name):
	pad = 5
	cv2.namedWindow("ADD_FACE")
	vc = cv2.VideoCapture(0)

	face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

	while vc.isOpened():
	    ret, frame = vc.read()

	    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	    img=frame
	    boxes = face_cascade.detectMultiScale(gray, 1.3, 5)
	    x1=0
	    y1=0
	    x2=0
	    y2=0
	    for (x, y, w, h) in boxes:
	        x1 = x-pad
	        y1 = y-pad
	        x2 = x+w+pad
	        y2 = y+h+pad
	        img = cv2.rectangle(frame,(x1, y1),(x2, y2),(255,0,0),2)
	    x1=x1+2
	    y1=y1+2
	    x2=x2-2
	    y2=y2-2
	    key = cv2.waitKey(100)
	    cv2.imshow("ADD_FACE", img)


	    if key == 27: # exit on ESC
	        h, w , ch = frame.shape
	        face = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
	        cv2.imwrite("faces/{0}.jpg".format(name),face)
	        break
	cv2.destroyWindow("ADD_FACE")