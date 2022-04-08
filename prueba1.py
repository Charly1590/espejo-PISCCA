from unittest import result
import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic
import time
import threading



# For webcam input:
cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc('M','J','P','G'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH,1240)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
cap.set(cv2.CAP_PROP_FPS,30)


with mp_holistic.Holistic(
    model_complexity=0) as holistic:

  pTime = 0
  cTime = 0

  # img1=cv2.imread('cepillo.png')

  # results=holistic.process(img1)

  # def cargarRed(num_hilo,**datos):
  #   global results
  #   results = holistic.process(datos["image"])
  #   # cv2.imshow('MediaPipe Holistic', datos["image"])
  #   # print("soy el hilo")
    
  while cap.isOpened():
    success, image = cap.read()

    start = time.time()


    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image.flags.writeable = False

    results = holistic.process(image)

    image.flags.writeable = True

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_holistic.POSE_CONNECTIONS)

    lmListL = []
    lmListR = []

    if results.left_hand_landmarks:
        myHand = results.left_hand_landmarks
        for id, lm in enumerate(myHand.landmark):
            # print(id, lm)
            h, w, c = image.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            # print(id, cx, cy)
            lmListL.append([id, cx, cy])
            cv2.circle(image, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

    if results.right_hand_landmarks:
        myHand = results.right_hand_landmarks
        for id, lm in enumerate(myHand.landmark):
            # print(id, lm)
            h, w, c = image.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            # print(id, cx, cy)
            lmListR.append([id, cx, cy])
            cv2.circle(image, (cx, cy), 5, (0, 255, 0), cv2.FILLED)
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    if len(lmListR) != 0: 
      print(lmListR[0], lmListR[4], lmListR[8], lmListR[12], lmListR[16], lmListR[20])


    # mp_drawing.draw_landmarks(
    #     image,
    #     results.left_hand_landmarks,
    #     mp_holistic.HAND_CONNECTIONS,
    #     landmark_drawing_spec=mp_drawing_styles
    #     .get_default_hand_landmarks_style())

    # mp_drawing.draw_landmarks(
    #     image,
    #     results.right_hand_landmarks,
    #     mp_holistic.HAND_CONNECTIONS,
    #     landmark_drawing_spec=mp_drawing_styles
    #     .get_default_hand_landmarks_style())


    # Flip the image horizontally for a selfie-view display.
    end = time.time()
    totalTime = end-start
    fps = 1 / totalTime
    
    # image=cv2.resize(image,(1920,1080))
    image=cv2.flip(image, 1)

    cv2.putText(image, str(int(fps)), (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                    (255, 0, 255), 3)
    
    

    # image=cv2.resize(image,(1920,1080))
    # image=cv2.rotate(image,cv2.ROTATE_90_COUNTERCLOCKWISE)
                    
    cv2.imshow('MediaPipe Holistic', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()