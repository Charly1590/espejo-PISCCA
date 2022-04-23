import cv2
from cv2 import log
import mediapipe as mp
import time 
import numpy as np
from PIL import Image
from modulos.moduloPosicionarImgs import Posicionamiento as put_img
import screeninfo
import random


class lavado_manos():
  
  def actividad():
    #mp_drawing = mp.solutions.drawing_utils

    #Banderas globales
    jabMano=False
    espuma=False
    manoR=False

    #Tiempo de lavado de manos
    cont_bact= 0

    #Definicion de Pantalla para Pantalla completa
    screen = screeninfo.get_monitors()[0]
    cv2.namedWindow('lavado_manos', cv2.WND_PROP_FULLSCREEN)
    cv2.moveWindow('lavado_manos', screen.x - 1, screen.y - 1)
    cv2.setWindowProperty('lavado_manos', cv2.WND_PROP_FULLSCREEN,  cv2.WINDOW_FULLSCREEN)


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



    def dibujar_burbujas (n, img_result):

      #img = np.array(img_result)
      #img_result = img[:, :, :3].copy()
      
      while (n > 0):
        img_overlay = img_burbuja[:, :, :3]
        img_overlay = cv2.cvtColor(img_overlay, cv2.COLOR_BGR2RGB)
        xbl = random.randint(xl-150, xl+150)
        ybl = random.randint(yl-150, yl+150)
        xbr = random.randint(xr-150, xr+150)
        ybr = random.randint(yr-150, yr+150)
        xb = int((xbl+xbr)/2)  
        yb = int((ybl+ybr)/2) 

        put_img.overlay_image_alpha(img_result, img_overlay, xb, yb, alpha_mask_burbuja)
        n= n-1
        
        
      


    with mp_pose.Pose(
      
      #Ajustes de la red neuronal
      min_detection_confidence=0.5,
      min_tracking_confidence=0.5,
      model_complexity=0) as pose:
      


      while cap.isOpened():

        #Lectura y volteado de imagen
        succes, image = cap.read()
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


          #Guardamos el Frame actual y dibujamos sobre la imagen
          img = np.array(image)
          img_result = img[:, :, :3].copy()

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


          #Tiempo maximo de las baterias
          if (cont_bact < 50):
            img_overlay = img_mano_izquierda[:, :, :3]
            img_overlay = cv2.cvtColor(img_overlay, cv2.COLOR_BGR2RGB)
            #Metodo que recive la imagen base, la imagen a dibuja su posicion y alpha 
            put_img.overlay_image_alpha(img_result, img_overlay, xl, yl, alpha_mask_mano_izquierda)

            #img = np.array(img_result)
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
            # 'Hitbox' del jabon 
            if (xl>=1040 and xl<=1160) and (yl>=440  and yl<=550):
              print('jabon der')
              
              jabMano=True
              espuma=True

            elif (xr>=1040 and xr<=1160) and (yr>=440  and yr<=560):
              print('jabon izq')
              
              jabMano=True
              manoR=True
              espuma=True



              
          if(espuma):
            if (xl+160>=xr and xl-140<=xr) and (yl+160>=yr  and yl-140<=yr):
              dibujar_burbujas(12,img_result)
              #Registro del tiempo para que desaparescan las bacterias
              cont_bact = cont_bact +1



          #Dibujado
          #img = np.array(img_result)
          #img_result = img[:, :, :3].copy()
          img_overlay = img_jabon[:, :, :3]
          img_overlay = cv2.cvtColor(img_overlay, cv2.COLOR_BGR2RGB)
          put_img.overlay_image_alpha(img_result, img_overlay, xj, yj, alpha_mask_jabon) 

          #Esta linea se encarga de dibujar el esqueleto sobre la persona 
          #mp_drawing.draw_landmarks(img_result, results2.pose_landmarks, mp_pose.POSE_CONNECTIONS)
          


        


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
          cv2.imshow('lavado_manos', img_result)
        except:
          image=cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
          cv2.imshow('lavado_manos', image)

        #Tecla de salida ESC
        if cv2.waitKey(5) & 0xFF == 27:
          cv2.destroyWindow('lavado_manos')
          break
    cap.release()