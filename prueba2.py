import cv2
import mediapipe as mp
import time 
import numpy as np
from PIL import Image
from moduloPosicionarImgs import Posicionamiento as put_img
import screeninfo 

screen = screeninfo.get_monitors()[0] 
cv2.namedWindow('lavado_dientes', cv2.WND_PROP_FULLSCREEN)
cv2.moveWindow('lavado_dientes', screen.x - 1, screen.y - 1)
cv2.setWindowProperty('lavado_dientes', cv2.WND_PROP_FULLSCREEN,  cv2.WINDOW_FULLSCREEN) 


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc('M','J','P','G'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH,1080)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
cap.set(cv2.CAP_PROP_FPS,60)


img_boca = np.array(Image.open("Boca.png"))
img_boca=cv2.rotate(img_boca, cv2.ROTATE_90_COUNTERCLOCKWISE)
img_boca=cv2.resize(img_boca,(50,100))
alpha_mask_boca= img_boca[:, :, 3] / 255.0

img_cepillo = np.array(Image.open("cepillo1.png"))
img_cepillo=cv2.rotate(img_cepillo, cv2.ROTATE_90_COUNTERCLOCKWISE)
# imagimg_cepilloe = cv2.cvtColor(img_cepillo, cv2.COLOR_BGR2RGB)
img_cepillo=cv2.resize(img_cepillo,(50,170))
alpha_mask_cepillo = img_cepillo[:, :, 3] / 255.0

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

    results2 = pose.process(image)

    image.flags.writeable = True

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    flagLeftHand=False

    lmListL = []
    lmListR = []
      
    img_result=None  
    image_height, image_width, _ = image.shape
    if results2.pose_landmarks:
      # mp_drawing.draw_landmarks(image, results2.pose_landmarks, mp_pose.POSE_CONNECTIONS)

      r_mouth_position_x=int(results2.pose_landmarks.landmark[mp_pose.PoseLandmark.MOUTH_RIGHT].x*image_width)
      r_mouth_position_y=int(results2.pose_landmarks.landmark[mp_pose.PoseLandmark.MOUTH_RIGHT].y*image_height)
      x, y = r_mouth_position_x, r_mouth_position_y
      x-=10
      y-=60
      img_result=put_img.put_elements_in_viedo(x,y,image,img_boca)

      r_hand_position_x=int(results2.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_INDEX].x*image_width)
      r_hand_position_y=int(results2.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_INDEX].y*image_height)
      x, y = r_hand_position_x, r_hand_position_y
      y-=100
      img_result=put_img.put_elements_in_viedo(x,y,img_result,img_cepillo)

    end = time.time()
    totalTime = end-start
    fps = 1 / totalTime

    cv2.putText(img_result, str(int(fps)), (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                    (255, 0, 255), 3)
                    
    try:
      img_result=cv2.rotate(img_result, cv2.ROTATE_90_CLOCKWISE)
      cv2.imshow('lavado_dientes', img_result)
    except:
      image=cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
      cv2.imshow('lavado_dientes', image)

    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()