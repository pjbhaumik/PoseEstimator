import cv2
import mediapipe as mp
import time
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

# mpPose = mp.solutions.pose
# pose = mpPose.Pose()
# mpDraw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture('a.mp4')
pTime = 0

with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as holistic:
    
    while cap.isOpened():
        success, img = cap.read()
        if success:
            img.flags.writeable = False
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = holistic.process(img)
            
            img.flags.writeable = True
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            new_plm = {}
            for id, lm in enumerate(results.pose_landmarks.landmark):
                h, w,c = img.shape
                print(id, lm)
                cx, cy = int(lm.x*w), int(lm.y*h)
                cv2.circle(img, (cx-200, cy), 5, (255,0,0), cv2.FILLED)
                new_plm[id] = [cx-200, cy]

            new_pcxcn = []
            for pcxcn in mp_holistic.POSE_CONNECTIONS:
                new_pcxcn.append(np.array([[new_plm[pcxcn[0]][0],new_plm[pcxcn[0]][1]],[new_plm[pcxcn[1]][0],new_plm[pcxcn[1]][1]]], np.int32))
            rectangleImage =cv2.polylines(img, new_pcxcn, False, (0,255,0), thickness=2)
            
            # new_flm = {}
            # for id, lm in enumerate(results.face_landmarks):
            #     h, w,c = img.shape
            #     print(id, lm)
            #     cx, cy = int(lm.x*w), int(lm.y*h)
            #     cv2.circle(img, (cx-200, cy), 5, (255,0,0), cv2.FILLED)
            #     new_flm[id] = [cx-200, cy]

            # new_fcxcn = []
            # for fcxcn in mp_holistic.FACEMESH_CONTOURS:
            #     new_fcxcn.append(np.array([[new_flm[fcxcn[0]][0],new_flm[fcxcn[0]][1]],[new_flm[fcxcn[1]][0],new_flm[fcxcn[1]][1]]], np.int32))
            # rectangleImage =cv2.polylines(img, new_fcxcn, False, (0,255,0), thickness=1)


            


            cTime = time.time()
            fps = 1/(cTime-pTime)
            pTime = cTime

        #cv2.putText(img, str(int(fps)), (50,50), cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0), 3)
        cv2.imshow("Image", img)
        cv2.waitKey(1)