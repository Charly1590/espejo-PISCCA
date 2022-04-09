import cv2
import mediapipe as mp
import time 
import numpy as np
from PIL import Image

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc('M','J','P','G'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH,1080)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
cap.set(cv2.CAP_PROP_FPS,60)

def overlay_image_alpha(img, img_overlay, x, y, alpha_mask):
    """Overlay `img_overlay` onto `img` at (x, y) and blend using `alpha_mask`.

    `alpha_mask` must have same HxW as `img_overlay` and values in range [0, 1].
    """
    # Image ranges
    y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
    x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])

    # Overlay ranges
    y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
    x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)

    # Exit if nothing to do
    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
        return

    # Blend overlay within the determined ranges
    img_crop = img[y1:y2, x1:x2]
    img_overlay_crop = img_overlay[y1o:y2o, x1o:x2o]
    alpha = alpha_mask[y1o:y2o, x1o:x2o, np.newaxis]
    alpha_inv = 1.0 - alpha

    img_crop[:] = alpha * img_overlay_crop + alpha_inv * img_crop

img_boca = np.array(Image.open("Boca.png"))
img_boca=cv2.rotate(img_boca, cv2.ROTATE_90_COUNTERCLOCKWISE)
img_boca=cv2.resize(img_boca,(50,100))
alpha_mask_boca= img_boca[:, :, 3] / 255.0

img_cepillo = np.array(Image.open("cepillo1.png"))
img_cepillo=cv2.rotate(img_cepillo, cv2.ROTATE_90_COUNTERCLOCKWISE)
# imagimg_cepilloe = cv2.cvtColor(img_cepillo, cv2.COLOR_BGR2RGB)
img_cepillo=cv2.resize(img_cepillo,(50,170))
alpha_mask_cepillo = img_cepillo[:, :, 3] / 255.0

# with mp_hands.Hands(
#       min_detection_confidence=0.5,
#       min_tracking_confidence=0.5,
#       model_complexity=0) as hands:
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

    # results = hands.process(image)

    results2 = pose.process(image)

    image.flags.writeable = True

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    flagLeftHand=False

    lmListL = []
    lmListR = []

    # if results.multi_hand_landmarks:
    #   for num, hand_landmarks in enumerate(results.multi_hand_landmarks):
    #     # mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    #     lbl = (results.multi_handedness[num].classification[0].label if results.multi_handedness[num].classification[0].score >= 0.5 else None)

    #     if lbl == "Left":
    #         for id, lm in enumerate(hand_landmarks.landmark):
    #           # print(id, lm)
    #           h, w, c = image.shape
    #           cx, cy = int(lm.x * w), int(lm.y * h)
    #           # print(id, cx, cy)
    #           lmListL.append([id, cx, cy])
    #           # cv2.circle(image, (cx, cy), 8, (255, 0, 255), cv2.FILLED)

    #           # coords = tuple(np.multiply(
    #           #   np.array((hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x, hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y)),
    #           #   [640,480]).astype(int))
    #     # if len(lmListL) != 0: 
    #     #   print(lmListL[0], lmListL[4], lmListL[8], lmListL[12], lmListL[16], lmListL[20])
        
    #     if lbl == "Right":
    #         for id, lm in enumerate(hand_landmarks.landmark):
    #           # print(id, lm)
    #           h, w, c = image.shape
    #           cx, cy = int(lm.x * w), int(lm.y * h)
    #           # print(id, cx, cy)
    #           lmListR.append([id, cx, cy])
    #           # cv2.circle(image, (cx, cy), 8, (255, 0, 0), cv2.FILLED)

    #           # coords = tuple(np.multiply(
    #           # np.array((hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x, hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y)),
    #           # [640,480]).astype(int))
        
    #     # if len(lmListR) != 0: 
    #     #   print(lmListR[0], lmListR[4], lmListR[8], lmListR[12], lmListR[16], lmListR[20])

        
      
    img_result=None  
    image_height, image_width, _ = image.shape
    if results2.pose_landmarks:
      # for body_landmarks in results2.pose_landmarks:
      # mp_drawing.draw_landmarks(image, results2.pose_landmarks, mp_pose.POSE_CONNECTIONS)

      # image=cv2.resize(image,(1920,1080))

      r_mouth_position_x=int(results2.pose_landmarks.landmark[mp_pose.PoseLandmark.MOUTH_RIGHT].x*image_width)
      r_mouth_position_y=int(results2.pose_landmarks.landmark[mp_pose.PoseLandmark.MOUTH_RIGHT].y*image_height)
      x, y = r_mouth_position_x, r_mouth_position_y

      x-=10
      y-=60
      img = np.array(image)
      
      img_result = img[:, :, :3].copy()
      img_overlay = img_boca[:, :, :3]
      
      img_overlay = cv2.cvtColor(img_overlay, cv2.COLOR_BGR2RGB)

      overlay_image_alpha(img_result, img_overlay, x, y, alpha_mask_boca)

      r_hand_position_x=int(results2.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_INDEX].x*image_width)
      r_hand_position_y=int(results2.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_INDEX].y*image_height)
      x, y = r_hand_position_x, r_hand_position_y
      y-=100
      img = np.array(img_result)
      
      img_result = img[:, :, :3].copy()
      img_overlay = img_cepillo[:, :, :3]

      img_overlay = cv2.cvtColor(img_overlay, cv2.COLOR_BGR2RGB)

      overlay_image_alpha(img_result, img_overlay, x, y, alpha_mask_cepillo)

    end = time.time()
    totalTime = end-start
    fps = 1 / totalTime

    cv2.putText(img_result, str(int(fps)), (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                    (255, 0, 255), 3)
                    
    try:
      img_result=cv2.rotate(img_result, cv2.ROTATE_90_CLOCKWISE)
      cv2.imshow('MediaPipe Holistic', img_result)
    except:
      image=cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
      cv2.imshow('MediaPipe Holistic', image)

    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()