import argparse
import time
import json
import cv2
import os
import dlib
import imutils
import pandas as pd
import numpy as np
from imutils import face_utils
from imutils.video import FileVideoStream, VideoStream
from scipy.spatial import distance as dist


def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


ap = argparse.ArgumentParser()
ap.add_argument(
    "-p", "--shape-predictor", required=True, help="path to facial landmark predictor"
)
#ap.add_argument("-v", "--video", type=str, default="", help="path to input video file")
args = vars(ap.parse_args())




print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

def go(path):
    EYE_AR_THRESH = 0.2
    EYE_AR_CONSEC_FRAMES = 3


    COUNTER = 0
    TOTAL = 0

    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]


    print("[INFO] starting video stream thread...")
    vs = FileVideoStream(str(path)).start()
    fileStream = True
    #vs = VideoStream(src=0).start()
    #fileStream = False
    time.sleep(1)

    start = time.time()
    try:
        while True:
            
            if fileStream and not vs.more():
                break
            
            frame = vs.read()
            #cv2.imshow("Frame", frame)
            try:
                frame = imutils.resize(frame, width=800)
            except:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            rects = detector(gray, 0)

            for rect in rects:

                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)

                leftEye = shape[lStart:lEnd]
                rightEye = shape[rStart:rEnd]
                leftEAR = eye_aspect_ratio(leftEye)
                rightEAR = eye_aspect_ratio(rightEye)

                ear = (leftEAR + rightEAR) / 2.0

                leftEyeHull = cv2.convexHull(leftEye)
                rightEyeHull = cv2.convexHull(rightEye)
                cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

                if ear < EYE_AR_THRESH:
                    COUNTER += 1

                else:

                    if COUNTER >= EYE_AR_CONSEC_FRAMES:
                        TOTAL += 1

                    COUNTER = 0

                cv2.putText(
                    frame,
                    "Blinks: {}".format(TOTAL),
                    (300, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )
                cv2.putText(
                    frame,
                    "EAR: {:.2f}".format(ear),
                    (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )

            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

        end = time.time()

        print("ALL DONE")
        cv2.destroyAllWindows()
        vs.stop()
        if(TOTAL == 0):
            return -1
        else:
            return (end-start)/TOTAL
    except:
        return -1

# f = open('./train_sample_videos/metadata.json',)
# df = pd.DataFrame({'name':[],'info':[],'score':[]})  
# d = json.load(f)
# fake = []
# real = []
# for name in os.listdir("./train_sample_videos/"):
#     n = go("./videp/" + name)
#     if(d[str(name)]['label'] == "FAKE"):
#         df.loc[len(df.index)] = [name,0,n]
#         fake.append(n)
#     else:
#         df.loc[len(df.index)] = [name,1,n]
#         real.append(n)
# print(fake + "\n")
# print(real)
# df.to_csv('file1.csv')
go("video\\ahfazfbntc.mp4")    

    
