import cv2
from cv2 import log
import mediapipe as mp
import time 
import numpy as np
from PIL import Image
from modulos.moduloPosicionarImgs import Posicionamiento as put_img
import screeninfo
import random



mp_drawing = mp.solutions.drawing_utils

#Banderas globales




jabMano = False
espuma=False
manoR=False

#Definicion de Pantalla para Pantalla completa
screen = screeninfo.get_monitors()[0]
cv2.namedWindow('MediaPipe Holistic', cv2.WND_PROP_FULLSCREEN)
cv2.moveWindow('MediaPipe Holistic', screen.x - 1, screen.y - 1)
cv2.setWindowProperty('MediaPipe Holistic', cv2.WND_PROP_FULLSCREEN,  cv2.WINDOW_FULLSCREEN)




#Red Neuronal
mp_pose = mp.solutions.pose



#Captura de Video
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc('M','J','P','G'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH,1360)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,768)
cap.set(cv2.CAP_PROP_FPS,60)




#Graficos añadidos

#Mano derecha
img_mano_izquierda = np.array(Image.open("recursos/autoc/lavmanos/las-bacterias.png"))
img_mano_izquierda=cv2.rotate(img_mano_izquierda, cv2.ROTATE_90_COUNTERCLOCKWISE)
img_mano_izquierda=cv2.resize(img_mano_izquierda,(100,120))
#Creacion de Alpha
alpha_mask_mano_izquierda= img_mano_izquierda[:, :, 3] / 255.0





#Mano izquierda
img_mano_derecha=cv2.flip(img_mano_izquierda, 0)
#img_mano_izquierda=cv2.rotate(img_mano_derecha, cv2.ROTATE_90_COUNTERCLOCKWISE)
img_mano_derecha=cv2.resize(img_mano_derecha,(120,100))
#Creacion de Alpha
alpha_mask_mano_derecha = img_mano_derecha[:, :, 3] / 255.0


#Jabon
tam = 2
img_jabon = np.array(Image.open("recursos/autoc/lavmanos/jabon.png"))
img_jabon=cv2.rotate(img_jabon, cv2.ROTATE_90_COUNTERCLOCKWISE)
img_jabon=cv2.resize(img_jabon,(60*tam,80*tam))
#Creacion de Alpha
alpha_mask_jabon= img_jabon[:, :, 3] / 255.0

#Burbujas
img_burbuja = np.array(Image.open("recursos/autoc/lavmanos/burbuja.png"))
img_burbuja=cv2.rotate(img_burbuja, cv2.ROTATE_90_COUNTERCLOCKWISE)
img_burbuja=cv2.resize(img_burbuja,(100 ,100 ))
#Creacion de Alpha
alpha_mask_burbuja= img_burbuja[:, :, 3] / 255.0



#Funcion para detectar el jabon 
#Cuidado, aqui perdi el trazo de cual es izquierda o derecha por que el video esta invertido, pero la red neuronal no xd
def tomar_jabon(xl, yl, xr, yr ):
  global jabMano
  global manoR
  global espuma
  
  # 'Hitbox' del jabon 
  if (xl>=1050 and xl<=1150) and (yl>=450  and yl<=550):
    print('jabon der')
    
    jabMano=True
    espuma=True

  elif (xr>=1050 and xr<=1150) and (yr>=450  and yr<=550):
    print('jabon izq')
    
    jabMano=True
    manoR=True
    espuma=True






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

      #Guardado calculado de la posicion central Mano Derecha
      l_position_x=int(results2.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].x*image_width)
      l_position_y=int(results2.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y*image_height)
      xwl, ywl =  l_position_x, l_position_y

      l_position_x=int(results2.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_INDEX].x*image_width)
      l_position_y=int(results2.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_INDEX].y*image_height)
      xil, yil =  l_position_x, l_position_y

      l_position_x=int(results2.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_PINKY].x*image_width)
      l_position_y=int(results2.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_PINKY].y*image_height)
      xpl, ypl =  l_position_x, l_position_y

      xl = (xwl + xil + xpl) / 3
      yl = (ywl + yil + ypl) / 3

      #Reajuste de valores y la posicion 
      xl=int(xl)-100
      yl=int(yl)

      #Guardamos el Frame actual y dibujamos sobre la imagen
      img = np.array(image)
      img_result = img[:, :, :3].copy()
      img_overlay = img_mano_izquierda[:, :, :3]
      img_overlay = cv2.cvtColor(img_overlay, cv2.COLOR_BGR2RGB)
      #Metodo que recive la imagen base, la imagen a dibuja su posicion y alpha 
      put_img.overlay_image_alpha(img_result, img_overlay, xl, yl, alpha_mask_mano_izquierda)



     #Guardado calculado de la posicion central Mano Derecha
      r_position_x=int(results2.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].x*image_width)
      r_position_y=int(results2.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y*image_height)
      xwr, ywr =  r_position_x, r_position_y

      r_position_x=int(results2.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_INDEX].x*image_width)
      r_position_y=int(results2.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_INDEX].y*image_height)
      xir, yir =  r_position_x, r_position_y

      r_position_x=int(results2.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_PINKY].x*image_width)
      r_position_y=int(results2.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_PINKY].y*image_height)
      xpr, ypr =  r_position_x, r_position_y

      xr = (xwr + xir + xpr) / 3
      yr = (ywr + yir + ypr) / 3

      #Reajuste de valores y la posicion 
      xr=int(xr)-100
      yr=int(yr)
      
      #Dibujado
      img = np.array(img_result)
      #img_result = img[:, :, :3].copy()
      img_overlay = img_mano_derecha[:, :, :3]
      img_overlay = cv2.cvtColor(img_overlay, cv2.COLOR_BGR2RGB)
      put_img.overlay_image_alpha(img_result, img_overlay, xr, yr, alpha_mask_mano_derecha)


      #Ajustes de la posicion del jabon y funcion
      #Funciones
      if(jabMano):
        if(manoR):
          xj = xr
          yj = yr

        else:
          xj = xl
          yj = yl

      else:
        xj = 1050
        yj = 450 
        tomar_jabon(xl, yl, xr, yr)
          
      if(espuma):
        if (xl+150>=xr and xl-150<=xr) and (yl+150>=yr  and yl-150<=yr):
          img = np.array(img_result)
          #img_result = img[:, :, :3].copy()
          img_overlay = img_burbuja[:, :, :3]
          img_overlay = cv2.cvtColor(img_overlay, cv2.COLOR_BGR2RGB)
          xbl = random.randint(xl-150, xl+150)
          ybl = random.randint(yl-150, yl+150)
          xbr = random.randint(xr-150, xr+150)
          ybr = random.randint(yr-150, yr+150)
          xb = int((xbl+xbr)/2)  
          yb = int((ybl+ybr)/2) 

          put_img.overlay_image_alpha(img_result, img_overlay, xb, yb, alpha_mask_burbuja)



      #Dibujado
      img = np.array(img_result)
      #img_result = img[:, :, :3].copy()
      img_overlay = img_jabon[:, :, :3]
      img_overlay = cv2.cvtColor(img_overlay, cv2.COLOR_BGR2RGB)
      put_img.overlay_image_alpha(img_result, img_overlay, xj, yj, alpha_mask_jabon) 


      mp_drawing.draw_landmarks(img_result, results2.pose_landmarks, mp_pose.POSE_CONNECTIONS)
      


    


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