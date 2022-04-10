import cv2
import mediapipe as mp
import time 
import numpy as np
from PIL import Image
from modulos.moduloPosicionarImgs import Posicionamiento as put_img
import screeninfo

#Definicion de Pantalla para Pantalla completa
screen = screeninfo.get_monitors()[0]
cv2.namedWindow('MediaPipe Holistic', cv2.WND_PROP_FULLSCREEN)
cv2.moveWindow('MediaPipe Holistic', screen.x - 1, screen.y - 1)
cv2.setWindowProperty('MediaPipe Holistic', cv2.WND_PROP_FULLSCREEN,  cv2.WINDOW_FULLSCREEN)




#Red Neuronal
mp_pose = mp.solutions.pose



#Captura de Video
cap = cv2.VideoCapture(2)
cap.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc('M','J','P','G'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH,1360)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,768)
cap.set(cv2.CAP_PROP_FPS,60)




#Graficos añadidos

#Mano derecha
img_mano_izquierda = np.array(Image.open("manosucia.png"))
img_mano_izquierda=cv2.rotate(img_mano_izquierda, cv2.ROTATE_90_COUNTERCLOCKWISE)
img_mano_izquierda=cv2.resize(img_mano_izquierda,(50,50))
#Creacion de Alpha
alpha_mask_mano_izquierda= img_mano_izquierda[:, :, 3] / 255.0





#Mano izquierda
img_mano_derecha=cv2.flip(img_mano_izquierda, 0)
#img_mano_izquierda=cv2.rotate(img_mano_derecha, cv2.ROTATE_90_COUNTERCLOCKWISE)
img_mano_derecha=cv2.resize(img_mano_derecha,(50,50))
#Creacion de Alpha
alpha_mask_mano_derecha = img_mano_derecha[:, :, 3] / 255.0



with mp_pose.Pose(
  
  #Ajustes de la red neuronal
  min_detection_confidence=0.5,
  min_tracking_confidence=0.5,
  model_complexity=0) as pose:
  


  while cap.isOpened():

    #Lectura y volteado de imagen
    success, image = cap.read()
    image=cv2.flip(image, 1)


    #Contador FPS
    start = time.time()


    #Convercion de colores de BGR a RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results2 = pose.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)



    #Mat para guardar la imagen final y variables para guardar su tamaño
    img_result=None  
    image_height, image_width, _ = image.shape



    #Guardado de puntos de interes
    if results2.pose_landmarks:

      #Guardado de posicion Mano Derecha
      l_wrist_position_x=int(results2.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].x*image_width)
      l_wrist_position_y=int(results2.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y*image_height)
      x, y =  l_wrist_position_x, l_wrist_position_y


      #Reajuste de la posicion 
      x-=100
      y-=0

      #Guardamos el Frame actual y dibujamos sobre la imagen
      img = np.array(image)
      img_result = img[:, :, :3].copy()
      img_overlay = img_mano_izquierda[:, :, :3]
      img_overlay = cv2.cvtColor(img_overlay, cv2.COLOR_BGR2RGB)
      #Metodo que recive la imagen base, la imagen a dibuja su posicion y alpha 
      put_img.overlay_image_alpha(img_result, img_overlay, x, y, alpha_mask_mano_izquierda)



      #Guardado de posicion Mano Izquierda
      r_wrist_position_x=int(results2.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].x*image_width)
      r_wrist_position_y=int(results2.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y*image_height)
      x, y = r_wrist_position_x, r_wrist_position_y

      #Reajuste de la posicion
      x-=100
      y-=0

      #Dibujado
      img = np.array(img_result)
      img_result = img[:, :, :3].copy()
      img_overlay = img_mano_derecha[:, :, :3]
      img_overlay = cv2.cvtColor(img_overlay, cv2.COLOR_BGR2RGB)
      put_img.overlay_image_alpha(img_result, img_overlay, x, y, alpha_mask_mano_derecha)



    #Fin del contador de FPS
    end = time.time()
    totalTime = end-start
    fps = 1 / totalTime

    #Dibujado de FPS
    cv2.putText(img_result, str(int(fps)), (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                    (255, 0, 255), 3)
                    

    #Display De imagenes
    try:
      img_result=cv2.rotate(img_result, cv2.ROTATE_90_CLOCKWISE)
      cv2.imshow('MediaPipe Holistic', img_result)
    except:
      image=cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
      cv2.imshow('MediaPipe Holistic', image)

    #Tecla de salida ESC
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()