# Importing Required Libraries
from scipy.spatial import distance # library to find distance between two points using method like euclidean
from imutils import face_utils # image processing library (face_utils for getting eyes landmarks)
import imutils 
import dlib # dlib is used to estimate the location of 68 coordinates (x, y) that map the facial points/landmarks on a person's face 
import cv2 # opencv library used for real-time computer vision tasks
from pygame import mixer # pygame library for playing sound/alarm/alert

mixer.init() # intializing the sound player
file = 'assets/alert.ogg' # fetching sound file
sound = mixer.Sound(file) # assigning sound file to sound variable

# Function to get eye aspect ration
def aspect_ratio(eye):
	A = distance.euclidean(eye[1], eye[5]) # height: distance between upper eyelid and lower eyelid  
	B = distance.euclidean(eye[2], eye[4]) # height: distance between upper eyelid and lower eyelid
	C = distance.euclidean(eye[0], eye[3]) # width of the eye
	ear = (A + B) / (2.0 * C) # eye aspect ration
	return ear 
	
thresh = 0.20 # threshold of the eye aspect ration upto which alarm won't start
frame_check = 20 # frame rate checking
detect = dlib.get_frontal_face_detector() # initializing hog face detector : https://towardsdatascience.com/hog-histogram-of-oriented-gradients-67ecd887675f
predict = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat") # Dat file is the crux of the code, we get 68 landmarks on the face by this file

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"] # getting landmarks of left eye i.e from 42-47
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"] # getting landmarks of right eye i.e from 36-41

cap=cv2.VideoCapture(0) # turning on the web cam (0 is default value for web cam) using opencv videocapture
flag=0 # initializing flag as 0

while True: # Initializing while loop
	ret, frame=cap.read() # capturing frames from web cam
	frame = imutils.resize(frame, width=450) # resize the frame of face images
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # converting colorful images to grayscale images using opencv inorder to reduce unneccessary data
	subjects = detect(gray, 0) # putting grayscale face images in the hog face detector model
	for subject in subjects: 
		shape = predict(gray, subject) # getting 68 landmarks on the grayscale face images capturing from the web cam in each frame
		# Here shape variable is the image of 68 landmarks in the form of (x, y) coordinates
		shape = face_utils.shape_to_np(shape) # converting grayscale images to numpy array
		# Now shape variable is the list of 68 landmarks in the form of (x, y) coordinates
		leftEye = shape[lStart:lEnd] # selecting the left eye landmarks from the 68 landmarks list
		rightEye = shape[rStart:rEnd] # selecting the right eye landmarks from the 68 landmarks list
		leftEAR = aspect_ratio(leftEye) # getting aspect ration of left eye
		rightEAR = aspect_ratio(rightEye) # getting aspect ration of right eye
		ear = (leftEAR + rightEAR) / 2.0 # getting aspect ration of whole eye
		leftEyeHull = cv2.convexHull(leftEye) # generating convex hull/boundry for the left eye opening
		rightEyeHull = cv2.convexHull(rightEye) # generating convex hull/boundry for the right eye opening
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1) # generating convex hull/boundry for the left eye opening in green color
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1) # generating convex hull/boundry for the right eye opening in green color
		if ear < thresh: # if eye aspect ration goes down and lesser than threshold value the variable flag get increase by one.
			flag += 1
			if flag >= frame_check: # if eye aspect ration remains lesser than threshold value for the consecutive 20 frames an alarm will blow and will give the alert. 
				sound.play()
				cv2.putText(frame, "****************ALERT!****************", (10, 30),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2) # Displaying Alert 
				cv2.putText(frame, "****************ALERT!****************", (10,325),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2) # Displaying Alert 
		else:
			flag = 0 # if eye aspect ration remains greater thean threshold, nothing will happen
	cv2.imshow("Frame", frame) # showing the web camera capturings
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"): # press q to quit
		break # after pressing q we get out of the while loop and video processing stops
cv2.destroyAllWindows() # destroy the video capturing and stops the web cam
cap.release() 
