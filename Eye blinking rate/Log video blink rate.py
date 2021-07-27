import csv
import argparse
import time
import cv2
import dlib
import os
import imutils
import numpy as np
from imutils import face_utils
from scipy.spatial import distance as dist

video_file_path = '..\\dfdc\\train_sample_videos' #video file path
EYE_AR_THRESH = 0.2
EYE_AR_CONSEC_FRAMES = 3
COUNTER = 0
TOTAL = 0

ap = argparse.ArgumentParser()
ap.add_argument(
    "-p", "--shape-predictor", required=True, help="path to facial landmark predictor"
)
#ap.add_argument("-v", "--video", type=str, default="", help="path to input video file")
args = vars(ap.parse_args())

def loadFacialLandmark():
	print("[INFO] loading facial landmark predictor...")
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor(args["shape_predictor"])
	# (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
	# (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
	return detector, predictor

def readVideo(COUNTER, TOTAL, video):
	start_time = time.time()
	cap = cv2.VideoCapture(video)

	if(cap.isOpened() == False):
		print("Error opening the video file")

	while(cap.isOpened()):
		ret, frame = cap.read()

		if ret == True:
			if time.time() - start_time >= 100:
				return 0,0
			frame = imutils.resize(frame, width=800)
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			rects = detector(gray, 0)

			for rect in rects:
				shape = predictor(gray, rect)
				shape = face_utils.shape_to_np(shape)
				(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
				(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
				leftEye = shape[lStart:lEnd]
				rightEye = shape[rStart:rEnd]
				leftEAR = eye_aspect_ratio(leftEye)
				rightEAR = eye_aspect_ratio(rightEye)

				ear = (leftEAR + rightEAR) / 2.0

				# leftEyeHull = cv2.convexHull(leftEye)
				# rightEyeHull = cv2.convexHull(rightEye)
				# cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
				# cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

				if ear < EYE_AR_THRESH:
					COUNTER += 1
				else:

					if COUNTER >= EYE_AR_CONSEC_FRAMES:
						TOTAL += 1

						COUNTER = 0

			# 	cv2.putText(frame,"Blinks: {}".format(TOTAL),(300, 50),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0, 0, 255),2,)
			# 	cv2.putText(
		 #            frame,
		 #            "EAR: {:.2f}".format(ear),
		 #            (300, 30),
		 #            cv2.FONT_HERSHEY_SIMPLEX,
		 #            0.7,
		 #            (0, 0, 255),
		 #            2,
		 #        )
			

			# //cv2.imshow('video Testing', frame)

			if cv2.waitKey(25) & 0xFF == ord('q'):
				break

		else:
			end_time = time.time()
			return TOTAL, (TOTAL*(60)) / (end_time - start_time), end_time - start_time

	cap.release()
	cv2.destroyAllWindows


def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

detector, predictor = loadFacialLandmark()
videosDone = []
print("Landmarks loaded")
with open("blinking_Test_celeb_df.csv", 'w', newline = '') as file:
	with open('logVideoFile.txt', 'w', newline='') as logFile: 
		writer = csv.writer(file)
		writer.writerow(["video name", "blinking rate ","total"])
		c = 1
		for video in os.listdir(video_file_path):
			if video[-4:] != '.mp4':
				continue
			videoFilePath = video_file_path + "\\"+video 
			tota_l, blinking_rate, totalVideotime = readVideo(COUNTER, TOTAL, videoFilePath)
			writer.writerow([video, str(blinking_rate),str(tota_l)])
			print(f'{c} video: {video}, blinking rate: {blinking_rate}, time: {totalVideotime}')
			videosDone.append(video)
			logFile.write(video + '\n')
			c += 1
