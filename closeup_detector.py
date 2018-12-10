# USAGE
# python facial_landmarks.py --shape-predictor shape_predictor_68_face_landmarks.dat --image images/example_01.jpg 

# import the necessary packages
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
import os
import glob
import sys


def split_video_in_frames(frameOutputUrl, videoUrl, frameRate) :
	cmd = 'ffmpeg -i ' + videoUrl + ' -r '+frameRate + ' ' + frameOutputUrl + 'frame%3d.png'
	os.system(cmd)
	temp_list=[]
	for image in glob.glob(os.path.join(frameOutputUrl, '*.png')):
		temp_list.append(image)
	print(temp_list)
	return temp_list



def detectFaces(shapePredictor, frameList, finalOutputUrl):
	index=0
	closeup=[]
	for image_url in frameList :
		
		# initialize dlib's face detector (HOG-based) and then create
		# the facial landmark predictor
		detector = dlib.get_frontal_face_detector()
		predictor = dlib.shape_predictor(shapePredictor)

		# load the input image, resize it, and convert it to grayscale
		image = cv2.imread(image_url)
		image = imutils.resize(image, width=500)
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

		# detect faces in the grayscale image
		rects = detector(gray, 1)

		# loop over the face detections
		for (i, rect) in enumerate(rects):
			# determine the facial landmarks for the face region, then
			# convert the facial landmark (x, y)-coordinates to a NumPy array
			shape = predictor(gray, rect)
			shape = face_utils.shape_to_np(shape)

			# convert dlib's rectangle to a OpenCV-style bounding box
			# [i.e., (x, y, w, h)], then draw the face bounding box
			(x, y, w, h) = face_utils.rect_to_bb(rect)
			# print(x,y,w,h)
			cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

			#call the closeupDetector function and pass the boolean value to the closeup list
			closeup.append(closeupDetector(image, x, y, w, h, index, finalOutputUrl))
			index+=1
		
		cv2.waitKey(0)
	print(closeup)
	return closeup

def closeupDetector(image, x, y, w , h, index, finalOutputUrl):

	height= np.size(image, 0)
	width= np.size(image, 1)
	areaOfFace = w*h
	AreaOfImage= width*height
	lLimit=0.1
	hLimit=0.7
	ratio= areaOfFace/AreaOfImage
	print(ratio)

	if ratio>=lLimit and ratio<=hLimit:
		score=(ratio-lLimit)/(hLimit-lLimit)*100
		g = float("{0:.1f}".format(score))
	# show the face number
		cv2.putText(image, "CloseUp- #{}".format(g), (x - 10, y - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
		
	elif ratio>hLimit: 
		cv2.putText(image, "Perfect CloseUp!", (x - 10, y - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
	else:
		cv2.putText(image, "Not a CloseUp!", (x - 10, y - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
	# Sace the image in a folder 
	cv2.imwrite(finalOutputUrl +'/output{}'.format(index) +'.jpg', image)
	if ratio>=lLimit: return 1
	else: return 0

if __name__ == '__main__':
	if(len(sys.argv)<6): 
		print("USAGE: python facial_landmarks.py (shape_predictor_file_name) (url of folder where frames are to be saved) (url of video to be splitted) (desired frame rate) (url of the final output folder)")
		exit()
	# sys.argv[0]= 'facial_landmarks.py
	shapePredictor = sys.argv[1]
	frameOutputUrl = sys.argv[2]
	videoUrl = sys.argv[3]
	finalOutputUrl= sys.argv[5]	
	frameRate = sys.argv[4]
	frameList = split_video_in_frames(frameOutputUrl, videoUrl, frameRate)
	detectFaces(shapePredictor, frameList, finalOutputUrl)




