import cv2
import mediapipe as mp
import time 
import numpy as np
from google.protobuf.json_format import MessageToDict

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc('M','J','P','G'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH,1240)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
cap.set(cv2.CAP_PROP_FPS,30)

with mp_hands.Hands(
      min_detection_confidence=0.7,
      min_tracking_confidence=0.7) as hands:
      with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=0) as pose:
        while cap.isOpened():
          success, image = cap.read()

          image=cv2.flip(image, 1)

          start = time.time()

          image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

          image.flags.writeable = False

          results = hands.process(image)

          results2 = pose.process(image)

          image.flags.writeable = True

          image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

          flagLeftHand=False

          lmListL = []
          lmListR = []

          if results.multi_hand_landmarks:
            for num, hand_landmarks in enumerate(results.multi_hand_landmarks):
              mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

              lbl = (results.multi_handedness[num].classification[0].label if results.multi_handedness[num].classification[0].score >= 0.99 else None)

              if lbl == "Left":
                  for id, lm in enumerate(hand_landmarks.landmark):
                    # print(id, lm)
                    h, w, c = image.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    # print(id, cx, cy)
                    lmListL.append([id, cx, cy])
                    cv2.circle(image, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

                  coords = tuple(np.multiply(
                      np.array((hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x, hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y)),
                      [640,480]).astype(int))
                  cv2.putText(image, lbl, coords, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2, cv2.LINE_AA)
              if len(lmListL) != 0: 
                print(lmListL[0], lmListL[4], lmListL[8], lmListL[12], lmListL[16], lmListL[20])
              
              if lbl == "Right":
                  for id, lm in enumerate(hand_landmarks.landmark):
                    # print(id, lm)
                    h, w, c = image.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    # print(id, cx, cy)
                    lmListR.append([id, cx, cy])
                    cv2.circle(image, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

                  coords = tuple(np.multiply(
                    np.array((hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x, hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y)),
                    [640,480]).astype(int))
                  cv2.putText(image, lbl, coords, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
              
              if len(lmListR) != 0: 
                print(lmListR[0], lmListR[4], lmListR[8], lmListR[12], lmListR[16], lmListR[20])

              
            
            
          if results2.pose_landmarks:
            # for body_landmarks in results2.pose_landmarks:
            mp_drawing.draw_landmarks(image, results2.pose_landmarks, mp_pose.POSE_CONNECTIONS)

          end = time.time()
          totalTime = end-start
          fps = 1 / totalTime
          
          # image=cv2.resize(image,(1920,1080))

          cv2.putText(image, str(int(fps)), (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                          (255, 0, 255), 3)
                          
          cv2.imshow('MediaPipe Holistic', image)
          if cv2.waitKey(5) & 0xFF == 27:
            break
      cap.release()