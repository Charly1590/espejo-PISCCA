import cv2
import mediapipe as mp
import time 
import numpy as np
from PIL import Image
from modulos.moduloPosicionarImgs import Posicionamiento as put_img
import screeninfo
import random
import multiprocessing
from playsound import playsound

class prevencion_nina():

  return_action=False
  
  def click_event(event, x, y, flags, params):
    global return_action
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x," ",y, "Lavadomanos")
        if (x>=540 and x<=720) and (y>=30 and y<=150):
          return_action=True

  def actividad():
    #mp_drawing = mp.solutions.drawing_utils

    #Banderas globales
    global return_action
    return_action=False
    
    
    animdir=False
    cicloanim=1.0
    posianim=0
    manodist= -1200
    actividadActual=0
    alerta=False

    #Definicion de Pantalla para Pantalla completa
    screen = screeninfo.get_monitors()[0]
    cv2.namedWindow('prev_nina', cv2.WND_PROP_FULLSCREEN)
    cv2.moveWindow('prev_nina', screen.x - 1, screen.y - 1)
    cv2.setWindowProperty('prev_nina', cv2.WND_PROP_FULLSCREEN,  cv2.WINDOW_FULLSCREEN)


    #Red Neuronal
    mp_pose = mp.solutions.pose



    #Captura de Video
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc('M','J','P','G'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,1360)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,768)
    cap.set(cv2.CAP_PROP_FPS,60)




    #Graficos añadidos
    #Para no complicarse con el cambio de tamaño e algunas imagenes (Bacterias, Burbujas)
    #Se cambio el tamaño de estas con un editor de imagenes (GIMP)

    img_focus = np.array(Image.open("recursos/edsex/focus.png"))
    img_focus=cv2.rotate(img_focus, cv2.ROTATE_90_COUNTERCLOCKWISE)
    #Creacion de Alpha
    alpha_mask_focus= img_focus[:, :, 3] / 255.0

    img_focus2 = np.array(Image.open("recursos/edsex/focus2.png"))
    img_focus2=cv2.rotate(img_focus2, cv2.ROTATE_90_COUNTERCLOCKWISE)
    #Creacion de Alpha
    alpha_mask_focus2= img_focus2[:, :, 3] / 255.0

    img_garra = np.array(Image.open("recursos/edsex/Sombra.png"))
    img_garra=cv2.rotate(img_garra, cv2.ROTATE_90_COUNTERCLOCKWISE)
    #Creacion de Alpha
    alpha_mask_garra= img_garra[:, :, 3] / 255.0

    #Atras
    img_return = np.array(Image.open("recursos/autoc/cepilladodientes/volver.png"))
    img_return=cv2.rotate(img_return, cv2.ROTATE_90_COUNTERCLOCKWISE)

    with mp_pose.Pose(
      
      #Ajustes de la red neuronal
      min_detection_confidence=0.6,
      min_tracking_confidence=0.6,
      model_complexity=0) as pose:
      


      while cap.isOpened():

        #Lectura y volteado de imagen
        succes, image = cap.read()
        image=cv2.flip(image, 1)


        #Contador FPS
        #start = time.time()


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


            #Esta linea se encarga de dibujar el esqueleto sobre la persona 
            #mp_drawing.draw_landmarks(img_result, results2.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            



            if actividadActual == 1:

              #Obtencion del pixcel central para el pecho
              l_position_x=int(results2.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x*image_width)
              l_position_y=int(results2.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y*image_height)
              xsl, ysl =  l_position_x, l_position_y

              r_position_x=int(results2.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x*image_width)
              r_position_y=int(results2.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y*image_height)
              xsr, ysr =  r_position_x, r_position_y

              xp = int((xsr + xsl) / 2)
              yp = int((ysl + ysr) / 2)

              if manodist >= -900:
                img_overlay = img_focus[:, :, :3]
                img_overlay = cv2.cvtColor(img_overlay, cv2.COLOR_BGR2RGB)
                put_img.overlay_image_alpha(img_result, img_overlay, xp-25, yp-100, alpha_mask_focus)

                #Animacion de la mano acercandose
                
                img_overlay = img_garra[:, :, :3]
                img_overlay = cv2.cvtColor(img_overlay, cv2.COLOR_BGR2RGB)
                put_img.overlay_image_alpha(img_result, img_overlay, xp-400, yp+manodist, alpha_mask_garra)
                
              #Control de velocidad
              if manodist <= -750:
                manodist+=6
              elif manodist <= -400:
                manodist+=1

              
              if manodist >= -800:
                alerta=True

            elif actividadActual == 2:

              #Obtencion del pixcel central para nalgas/genitales
              l_position_x=int(results2.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].x*image_width)
              l_position_y=int(results2.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y*image_height)
              xsl, ysl =  l_position_x, l_position_y

              r_position_x=int(results2.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].x*image_width)
              r_position_y=int(results2.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].y*image_height)
              xsr, ysr =  r_position_x, r_position_y

              xp = int((xsr + xsl) / 2)
              yp = int((ysl + ysr) / 2)

              if manodist >= -900:
                img_overlay = img_focus[:, :, :3]
                img_overlay = cv2.cvtColor(img_overlay, cv2.COLOR_BGR2RGB)
                put_img.overlay_image_alpha(img_result, img_overlay, xp-25, yp-100, alpha_mask_focus)

                #Animacion de la mano acercandose
                
                img_overlay = img_garra[:, :, :3]
                img_overlay = cv2.cvtColor(img_overlay, cv2.COLOR_BGR2RGB)
                put_img.overlay_image_alpha(img_result, img_overlay, xp-400, yp+manodist, alpha_mask_garra)
              
              #Control de velocidad
              if manodist <= -750:
                manodist+=6
              elif manodist <= -400:
                manodist+=1


              
              if manodist >= -800:
                alerta=True

            elif actividadActual == 3:

              #Obtencion del pixcel central para la boca
              l_position_x=int(results2.pose_landmarks.landmark[mp_pose.PoseLandmark.MOUTH_LEFT].x*image_width)
              l_position_y=int(results2.pose_landmarks.landmark[mp_pose.PoseLandmark.MOUTH_LEFT].y*image_height)
              xsl, ysl =  l_position_x, l_position_y

              r_position_x=int(results2.pose_landmarks.landmark[mp_pose.PoseLandmark.MOUTH_RIGHT].x*image_width)
              r_position_y=int(results2.pose_landmarks.landmark[mp_pose.PoseLandmark.MOUTH_RIGHT].y*image_height)
              xsr, ysr =  r_position_x, r_position_y

              xp = int((xsr + xsl) / 2)
              yp = int((ysl + ysr) / 2)

              if manodist >= -900:
                img_overlay = img_focus2[:, :, :3]
                img_overlay = cv2.cvtColor(img_overlay, cv2.COLOR_BGR2RGB)
                put_img.overlay_image_alpha(img_result, img_overlay, xp-30, yp-30, alpha_mask_focus2)

                #Animacion de la mano acercandose
                
                img_overlay = img_garra[:, :, :3]
                img_overlay = cv2.cvtColor(img_overlay, cv2.COLOR_BGR2RGB)
                put_img.overlay_image_alpha(img_result, img_overlay, xp-440, yp+manodist, alpha_mask_garra)
                
              #Control de velocidad
              if manodist <= -750:
                manodist+=6
              elif manodist <= -400:
                manodist+=1
              

              if manodist >= -800:
                alerta=True

            else:
              actividadActual = random.randint(1, 3)
              #actividadActual = 1
            


            #Guardado calculado de la posicion central Mano Derecha
            lm_position_x=int(results2.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].x*image_width)
            lm_position_y=int(results2.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y*image_height)
            xwl, ywl =  lm_position_x, lm_position_y

            lm_position_x=int(results2.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_INDEX].x*image_width)
            lm_position_y=int(results2.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_INDEX].y*image_height)
            xil, yil =  lm_position_x, lm_position_y

            lm_position_x=int(results2.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_PINKY].x*image_width)
            lm_position_y=int(results2.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_PINKY].y*image_height)
            xpl, ypl =  lm_position_x, lm_position_y

            xl = (xwl + xil + xpl) / 3
            yl = (ywl + yil + ypl) / 3

            #Reajuste de valores y la posicion 
            xl=int(xl)
            yl=int(yl)
            

            #Guardado calculado de la posicion central Mano Derecha
            rm_position_x=int(results2.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].x*image_width)
            rm_position_y=int(results2.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y*image_height)
            xwr, ywr =  rm_position_x, rm_position_y

            rm_position_x=int(results2.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_INDEX].x*image_width)
            rm_position_y=int(results2.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_INDEX].y*image_height)
            xir, yir =  rm_position_x, rm_position_y

            rm_position_x=int(results2.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_PINKY].x*image_width)
            rm_position_y=int(results2.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_PINKY].y*image_height)
            xpr, ypr =  rm_position_x, rm_position_y

            xr = (xwr + xir + xpr) / 3
            yr = (ywr + yir + ypr) / 3

            #Reajuste de valores y la posicion 
            xr=int(xr)
            yr=int(yr)


            #Creacion de la animacion de llamado de atencion 
            if cicloanim >= 1.0:
              animdir=False
            elif cicloanim <= 0.86:
              animdir=True

            if animdir:
              cicloanim+=0.03
              posianim-=3
            else:
              cicloanim-=0.03
              posianim+=3

            #print(cicloanim)

            if alerta==True:
              #Posicion "No Tocar"
              xn=170
              yn=400
              #Tamaño de la imagen
              xnf=xn+229
              ynf=yn+200
              


              #"No Tocar" 
              img_notoc = np.array(Image.open("recursos/edsex/Manono.png"))
              img_notoc=cv2.rotate(img_notoc, cv2.ROTATE_90_COUNTERCLOCKWISE)
              #Alpha
              alpha_mask_notoc= img_notoc[:, :, 3] / 255.0

              #Dibujado de "No tocar"
              img_notoc = cv2.resize(img_notoc, (int(229*cicloanim), int(200*cicloanim)) , interpolation= cv2.INTER_LINEAR)
              alpha_mask_notoc = cv2.resize(alpha_mask_notoc, (int(229*cicloanim), int(200*cicloanim)) , interpolation= cv2.INTER_LINEAR)
              
              
              
              
              img_overlay = img_notoc[:, :, :3]
              img_overlay = cv2.cvtColor(img_overlay, cv2.COLOR_BGR2RGB)
              
              put_img.overlay_image_alpha(img_result, img_overlay, xn+posianim, yn+posianim, alpha_mask_notoc)


              
              #Dibujar rectangulo de "No tocar"
              # start_jabon = (xn, yn) 
              # end_jabon = (xnf, ynf)
              # color = (255, 0, 255) 
              # thickness = 2
              # img_result = cv2.rectangle(img_result, start_jabon, end_jabon, color, thickness)

              if (xl>=xn and xl<=xnf) and (yl>=yn and yl<=ynf):
                actividadActual = 0
                manodist= -1200
                alerta=False
              if (xr>=xn and xr<=xnf) and (yr>=yn and yr<=ynf):
                actividadActual = 0
                manodist= -1200
                alerta=False



        #Fin del contador de FPS
        # end = time.time()
        # totalTime = end-start 
        # fps = 1 / totalTime

        #Dibujado de FPS
        # cv2.putText(img_result, str(int(fps)), (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
        #                 (255, 0, 255), 3)




        #Display De imagenes
        try:
          img_result=put_img.put_elements_in_viedo(20,10,img_result,img_return)
          img_result=cv2.rotate(img_result, cv2.ROTATE_90_CLOCKWISE)
          cv2.imshow('prev_nina', img_result)
          cv2.setMouseCallback('prev_nina', prevencion_nina.click_event)
        except:
          image=put_img.put_elements_in_viedo(20,10,image,img_return)
          image=cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
          cv2.imshow('prev_nina', image)
          cv2.setMouseCallback('prev_nina', prevencion_nina.click_event)

        #Tecla de salida ESC
        try:
          if return_action or (cv2.waitKey(5) & 0xFF == 27):
            cv2.destroyWindow('prev_nina')
            break
        except Exception as e:
          print(e)

    cap.release()