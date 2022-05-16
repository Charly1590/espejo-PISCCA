import cv2
from defer import return_value
import mediapipe as mp
import time 
import numpy as np
from PIL import Image
from modulos.moduloPosicionarImgs import Posicionamiento as put_img
import screeninfo
import random
import math
import threading

import multiprocessing
from playsound import playsound
from pygame import mixer

class lavado_dientes():
  
  return_action=False

  def click_event(event, x, y, flags, params):
    global return_action
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x," ",y, "sdahsjk")
        if (x>=540 and x<=720) and (y>=30 and y<=150):
          return_action=True

  def dibujar_burbujas (n, img_result, mx, my):
      img_burbuja = np.array(Image.open("recursos/autoc/cepilladodientes/Espuma.png"))
      img_burbuja=cv2.rotate(img_burbuja, cv2.ROTATE_90_COUNTERCLOCKWISE)
      # img_burbuja=cv2.resize(img_burbuja,(30 ,40))
      alpha_mask_burbuja= img_burbuja[:, :, 3] / 255.0

      while (n > 0):
        img_overlay = img_burbuja[:, :, :3]
        img_overlay = cv2.cvtColor(img_overlay, cv2.COLOR_BGR2RGB)
        x_random = int(random.randint(mx-10, mx+40))
        y_random = int(random.randint(my-20, my+40))

        put_img.overlay_image_alpha(img_result, img_overlay, x_random, y_random, alpha_mask_burbuja)
        n= n-1
      
      return img_result
  
  def dibujar_brillos (n, img_result, mx, my):
      img_brillo = np.array(Image.open("recursos/autoc/cepilladodientes/Brillo1.png"))
      img_brillo=cv2.rotate(img_brillo, cv2.ROTATE_90_COUNTERCLOCKWISE)
      # img_brillo=cv2.resize(img_brillo,(30 ,130))
      alpha_mask_burbuja= img_brillo[:, :, 3] / 255.0

      while (n > 0):
        img_overlay = img_brillo[:, :, :3]
        img_overlay = cv2.cvtColor(img_overlay, cv2.COLOR_BGR2RGB)
        x_random = int(random.randint(mx-20, mx+40))
        y_random = int(random.randint(my-30, my+10))
        
        put_img.overlay_image_alpha(img_result, img_overlay, x_random, y_random, alpha_mask_burbuja)
        n= n-1
      
      return img_result
  
  def checkSound():
    playsound('recursos/autoc/cepilladodientes/check.mp3')
  
  def brushingSound():
    playsound('recursos/autoc/cepilladodientes/brushing.mp3')

  def actividad():
    
    # Variable globales
    global return_action
    return_action=False
    cepillo_mano=False
    cepillo_mano_derecha=False
    cepillo_mano_izquierda=False
    soundBrushin=True
    soundCheck=True
    pararCepillado=True

    introduccion=True
    primera_indicacion=True
    agarra_cepillo=True
    agarra_pasta=True
    poner_pasta=True
    cepillarse_dientes=True
    arriba_abajo=True
    dientes_limpios=True

    t_fin=0
    t_ini_chek=0
    t_fin_check=0
    pasta_mano_derecha=1
    pasta_mano_izquierda=1
    bacteria=0

    mx=0
    zhand=0
    zmouth=0
    my=0
    rhx=0
    rhy=0
    
    #Esta es la imagen que se muestra como guia durante el cepillado de dientes y su alpha
    imagenGuia="recursos/autoc/cepilladodientes/CogerCepillo.png"
    imagenGuiaAlpha="recursos/autoc/cepilladodientes/Alphas/CogerCepillo.png"

    mixer.init()

    """
      Se pondra a la ventana en pantalla completa para evitar
      los bordes de la interfaz del sistema
    """
    screen = screeninfo.get_monitors()[0] 
    cv2.namedWindow('lavado_dientes', cv2.WND_PROP_FULLSCREEN)
    cv2.moveWindow('lavado_dientes', screen.x - 1, screen.y - 1)
    cv2.setWindowProperty('lavado_dientes', cv2.WND_PROP_FULLSCREEN,  cv2.WINDOW_FULLSCREEN) 

    mp_pose = mp.solutions.pose

    cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc('M','J','P','G'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,1360)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,768)
    cap.set(cv2.CAP_PROP_FPS,60)

    
    # Carga de imagenes
    img_boca = np.array(Image.open("recursos/autoc/cepilladodientes/Boca.png"))
    img_boca=cv2.rotate(img_boca, cv2.ROTATE_90_COUNTERCLOCKWISE)
    # img_boca=cv2.resize(img_boca,(50,100))

    img_cepillo = np.array(Image.open("recursos/autoc/cepilladodientes/cepillo1.png"))
    img_cepillo=cv2.rotate(img_cepillo, cv2.ROTATE_90_COUNTERCLOCKWISE)
    img_cepillo=cv2.rotate(img_cepillo, cv2.ROTATE_90_COUNTERCLOCKWISE)
    # img_cepillo=cv2.resize(img_cepillo,(200,50))

    img_pasta = np.array(Image.open("recursos/autoc/cepilladodientes/Pasta.png"))
    img_pasta = cv2.rotate(img_pasta, cv2.ROTATE_90_COUNTERCLOCKWISE)
    img_pasta = cv2.rotate(img_pasta, cv2.ROTATE_90_COUNTERCLOCKWISE)
    # img_pasta=cv2.resize(img_pasta,(200,200))

    img_cepillo_right= np.array(Image.open("recursos/autoc/cepilladodientes/cepillo1.png"))
    img_cepillo_right=cv2.rotate(img_cepillo_right, cv2.ROTATE_90_CLOCKWISE)
    # img_cepillo_right=cv2.resize(img_cepillo_right,(50,200))
    
    img_cepilloPasta_right= np.array(Image.open("recursos/autoc/cepilladodientes/CepilloPasta.png"))
    img_cepilloPasta_right=cv2.rotate(img_cepilloPasta_right, cv2.ROTATE_90_COUNTERCLOCKWISE)
    # img_cepilloPasta_right=cv2.resize(img_cepilloPasta_right,(50,200))

    img_pasta_right= np.array(Image.open("recursos/autoc/cepilladodientes/Pasta.png"))
    img_pasta_right=cv2.rotate(img_pasta_right, cv2.ROTATE_90_COUNTERCLOCKWISE)
    # img_pasta_right=cv2.resize(img_pasta_right,(200,200))

    img_cepillo_left= np.array(Image.open("recursos/autoc/cepilladodientes/cepillo1.png"))
    img_cepillo_left=cv2.rotate(img_cepillo_left, cv2.ROTATE_90_COUNTERCLOCKWISE)
    # img_cepillo_left=cv2.resize(img_cepillo_left,(50,200))

    img_cepilloPasta_left= np.array(Image.open("recursos/autoc/cepilladodientes/CepilloPasta.png"))
    img_cepilloPasta_left=cv2.rotate(img_cepilloPasta_left, cv2.ROTATE_90_CLOCKWISE)
    # img_cepilloPasta_left=cv2.resize(img_cepilloPasta_left,(50,200))

    img_pasta_left= np.array(Image.open("recursos/autoc/cepilladodientes/Pasta.png"))
    img_pasta_left=cv2.rotate(img_pasta_left, cv2.ROTATE_90_CLOCKWISE)
    # img_pasta_left=cv2.resize(img_pasta_left,(200,200))

    img_return = np.array(Image.open("recursos/autoc/cepilladodientes/volver.png"))
    img_return=cv2.rotate(img_return, cv2.ROTATE_90_COUNTERCLOCKWISE)
    # img_return=cv2.resize(img_return,(150,150))

    img_bacteria1= np.array(Image.open("recursos/autoc/cepilladodientes/bacteria1.png"))
    img_bacteria1=cv2.rotate(img_bacteria1, cv2.ROTATE_90_COUNTERCLOCKWISE)
    # img_bacteria1=cv2.resize(img_bacteria1,(30,30))
    
    img_bacteria2= np.array(Image.open("recursos/autoc/cepilladodientes/bacteria3.png"))
    img_bacteria2=cv2.rotate(img_bacteria2, cv2.ROTATE_90_COUNTERCLOCKWISE)
    # img_bacteria2=cv2.resize(img_bacteria2,(30,30))

    img_bacteria3= np.array(Image.open("recursos/autoc/cepilladodientes/bacterias7.png"))
    img_bacteria3=cv2.rotate(img_bacteria3, cv2.ROTATE_90_COUNTERCLOCKWISE)
    # img_bacteria3=cv2.resize(img_bacteria3,(30,30))

    img_bacteria4= np.array(Image.open("recursos/autoc/cepilladodientes/bacterias9.png"))
    img_bacteria4=cv2.rotate(img_bacteria4, cv2.ROTATE_90_COUNTERCLOCKWISE)
    # img_bacteria4=cv2.resize(img_bacteria4,(30,30))
    
    img_bacteria5= np.array(Image.open("recursos/autoc/cepilladodientes/bacterias12.png"))
    img_bacteria5=cv2.rotate(img_bacteria5, cv2.ROTATE_90_COUNTERCLOCKWISE)
    # img_bacteria5=cv2.resize(img_bacteria5,(30,30))

    """
      Se carga la red neuronal con un modelo de complegidad
      basico para evitar una carga mayor al dispositivo,
      si se decea aumentar la complejidad los valores son:
      0 = complejidad baja
      1 = complejidad media
      2 = complejidad alta
      Mientras mas alta la complejidad sera mejor la prediccion
      pero sera una mayor carga de procesamiento
    """
    with mp_pose.Pose(
      min_detection_confidence=0.5,
      min_tracking_confidence=0.5,
      model_complexity=0) as pose:
      while cap.isOpened():
        
        t_ini=time.time()
        success, image = cap.read()

        image=cv2.flip(image, 1)

        # start = time.time()

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image.flags.writeable = False

        results2 = pose.process(image)

        image.flags.writeable = True

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
          
        img_result=None  
        img_result2=None
        image_height, image_width, _ = image.shape

        if introduccion:
              mixer.music.load('recursos/audios/lavadoDientes/inicio.ogg')
              mixer.music.play()
              introduccion=False

        if not introduccion and t_fin>=7.5 :
          if primera_indicacion: 
            mixer.music.load('recursos/audios/lavadoDientes/indicacionInicial.ogg')
            mixer.music.play()
            primera_indicacion=False
            t_fin=0

        if results2.pose_landmarks and not primera_indicacion:
          
          if agarra_cepillo and t_fin>=6.5:
              mixer.music.load('recursos/audios/lavadoDientes/cogerCepillo.ogg')
              mixer.music.play()
              agarra_cepillo=False
              t_fin=0

          """
            Se obtendran las coordenadas de la boca en la parte derecha
            se utiliza la misma logica para las otras partes del cuerpo
            si se decea obtener mas informacion de como llamar a otras 
            partes del cuerpo se puede encontrar en: https://google.github.io/mediapipe/solutions/pose
          """
          r_mouth_position_x=int(results2.pose_landmarks.landmark[mp_pose.PoseLandmark.MOUTH_RIGHT].x*image_width)
          r_mouth_position_y=int(results2.pose_landmarks.landmark[mp_pose.PoseLandmark.MOUTH_RIGHT].y*image_height)
          zmouth=round(results2.pose_landmarks.landmark[mp_pose.PoseLandmark.MOUTH_RIGHT].z,2)
          mx, my = r_mouth_position_x, r_mouth_position_y
          mx-=25
          my-=70

          img_result=put_img.put_elements_in_viedo(mx,my,image,img_boca)
          img_result2=put_img.put_elements_in_viedo(mx,my,image,img_boca)
          # cv2.putText(img_result, str(zmouth), (mx, my), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
          #               (255, 0, 255), 3)

          # se grafica el cepillo a para tomarlo
          if cepillo_mano == False:
            l_hand_position_x=int(results2.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_INDEX].x*image_width)
            l_hand_position_y=int(results2.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_INDEX].y*image_height)
            lhx, lhy = l_hand_position_x, l_hand_position_y

            r_hand_position_x=int(results2.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_INDEX].x*image_width)
            r_hand_position_y=int(results2.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_INDEX].y*image_height)
            rhx, rhy = r_hand_position_x, r_hand_position_y

            #if para saber si se agarra con la mano derecha
            if (rhx>=480 and rhx<=516) and (rhy>=300 and rhy<=496) and not agarra_cepillo and t_fin>=4:
              cepillo_mano=True
              cepillo_mano_derecha=True
              mixer.music.load('recursos/autoc/cepilladodientes/check.ogg')
              mixer.music.play()
              t_ini_check=time.time()
              #Esta es la imagen que se muestra como guia durante el cepillado de dientes y su alpha
              imagenGuia="recursos/autoc/cepilladodientes/AgarrarPasta.png"
              imagenGuiaAlpha="recursos/autoc/cepilladodientes/Alphas/AgarrarPasta.png"
              
            #if para saber si se agarra con la mano izquierda  
            if (lhx>=480 and lhx<=516) and (lhy>=300 and lhy<=496) and not agarra_cepillo and t_fin>=4:
              cepillo_mano=True
              cepillo_mano_izquierda=True
              mixer.music.load('recursos/autoc/cepilladodientes/check.ogg')
              mixer.music.play()
              t_ini_check=time.time()
              #Esta es la imagen que se muestra como guia durante el cepillado de dientes y su alpha
              imagenGuia="recursos/autoc/cepilladodientes/AgarrarPasta.png"
              imagenGuiaAlpha="recursos/autoc/cepilladodientes/Alphas/AgarrarPasta.png"
              
              
            img_result=put_img.put_elements_in_viedo(350,300,img_result,img_cepillo)
          
          if bacteria<=75:
            img_result=put_img.put_elements_in_viedo(mx+10,my+10,img_result,img_bacteria1)  
            img_result2=img_result
          if bacteria<=105:
            img_result=put_img.put_elements_in_viedo(mx+12,my+20,img_result,img_bacteria2)  
            img_result2=img_result
          if bacteria<=125:
            img_result=put_img.put_elements_in_viedo(mx+7,my+30,img_result,img_bacteria3)  
            img_result2=img_result
          if bacteria<=160:
            img_result=put_img.put_elements_in_viedo(mx+10,my+40,img_result,img_bacteria4)  
            img_result2=img_result
          if bacteria<=185:
            img_result=put_img.put_elements_in_viedo(mx+7,my+50,img_result,img_bacteria5)  
            img_result2=img_result
          if bacteria>185:
            if soundCheck:
              mixer.music.load('recursos/autoc/cepilladodientes/check.ogg')
              mixer.music.play()
              t_ini_check=time.time()
              soundCheck=False
            if dientes_limpios and t_fin_check>=2.5:
              mixer.music.load('recursos/audios/lavadoDientes/dientesLimpios.ogg')
              mixer.music.play()
              dientes_limpios=False
              t_fin_check=0
            img_result=lavado_dientes.dibujar_brillos(3,img_result,mx,my)
            img_result2=img_result

          if bacteria<=185:  
            
            if cepillo_mano_derecha:

              if agarra_pasta and t_fin_check>=2.5:
                mixer.music.load('recursos/audios/lavadoDientes/aggarraLaPasta.ogg')
                mixer.music.play()
                agarra_pasta=False
                t_fin=0
                t_fin_check=0

              r_hand_position_x=int(results2.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_INDEX].x*image_width)
              r_hand_position_y=int(results2.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_INDEX].y*image_height)
              zhand=round(results2.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_INDEX].z,2)
              rhx, rhy = r_hand_position_x, r_hand_position_y
              rhx-=30
              rhy-=80
              img_result=put_img.put_elements_in_viedo(rhx,rhy,img_result,img_cepillo_right)

              distance_mouth_hand = round(math.sqrt((mx-rhx)**2+(my-rhy)**2),2)   

              l_hand_position_x=int(results2.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_INDEX].x*image_width)
              l_hand_position_y=int(results2.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_INDEX].y*image_height)
              lhx, lhy = l_hand_position_x, l_hand_position_y
              lhx-=100
              lhy-=60

              # se grafica la pasta para tomarla
              if pasta_mano_izquierda == 1:

                # #Dibujar rectangulo Toalla
                # start_agua = (350, 300) 
                # end_agua = (516, 400)
                # color = (0, 255, 0) 
                # thickness = 2
                # img_result = cv2.rectangle(img_result, start_agua, end_agua, color, thickness) 


                img_result=put_img.put_elements_in_viedo(350,300,img_result,img_pasta)
                if (lhx>=350 and lhx<=516) and (lhy>=300 and lhy<=400):
                  pasta_mano_izquierda=2
                  imagenGuia="recursos/autoc/cepilladodientes/PonersePasta.png"
                  imagenGuiaAlpha="recursos/autoc/cepilladodientes/Alphas/PonersePasta.png"
                  mixer.music.load('recursos/autoc/cepilladodientes/check.ogg')
                  mixer.music.play()
                  t_ini_check=time.time()
              else:
                img_result=put_img.put_elements_in_viedo(lhx,lhy,img_result,img_pasta_left)
              
              
              if pasta_mano_izquierda == 2 and not agarra_pasta and t_fin>=4 and t_fin_check>=2.5:

                if poner_pasta and t_fin_check>=4.5:
                    mixer.music.load('recursos/audios/lavadoDientes/ponerPastaCepillo.ogg')
                    mixer.music.play()
                    poner_pasta=False
                    t_fin=0
                    t_fin_check=0
                distance_hands=round(math.sqrt((lhx-rhx)**2+(lhy-rhy)**2),2)   
                # cv2.putText(img_result, str(int(distance_hands)), (150, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                #           (255, 0, 255), 3)
                if distance_hands>=85 and distance_hands<=200:

                  imagenGuia="recursos/autoc/cepilladodientes/Cepillarse.png"
                  imagenGuiaAlpha="recursos/autoc/cepilladodientes/Alphas/Cepillarse.png"
                  mixer.music.load('recursos/autoc/cepilladodientes/check.ogg')
                  mixer.music.play()
                  t_ini_check=time.time()
                  pasta_mano_izquierda = 3
                  
              # se grafica el cepillo con pasta
              if pasta_mano_izquierda == 3:
                img_result=put_img.put_elements_in_viedo(rhx,rhy,img_result2,img_cepilloPasta_right)
                diferencia_en_x=mx-rhx
                if not poner_pasta and t_fin>=4 and t_fin_check>=3.5:
                  if cepillarse_dientes:
                      mixer.music.load('recursos/audios/lavadoDientes/cepillarseDientes.ogg')
                      mixer.music.play()
                      cepillarse_dientes=False
                      t_fin=0

                if not cepillarse_dientes and t_fin>=3:
                  if arriba_abajo:
                    mixer.music.load('recursos/audios/lavadoDientes/arribaAbajoCepillado.ogg')
                    mixer.music.play()
                    arriba_abajo=False
                    t_fin=0

                if not arriba_abajo and t_fin>=4.5:
                  try:
                    # se grafica las burbujas del cepillado
                    if distance_mouth_hand >= 80 and distance_mouth_hand <= 200 and diferencia_en_x>=-105 and diferencia_en_x<=105:
                      bacteria+=1
                      img_result=lavado_dientes.dibujar_burbujas(7,img_result,mx,my)
                      if soundBrushin:
                        mixer.music.load('recursos/autoc/cepilladodientes/brushing.ogg')
                        mixer.music.play()
                        soundBrushin=False
                    elif not soundBrushin:
                      mixer.music.stop()
                      soundBrushin=True
                  except:
                    img_result=img_result
                

                   
            if cepillo_mano_izquierda:

              if agarra_pasta and t_fin_check>=2.5:
                mixer.music.load('recursos/audios/lavadoDientes/aggarraLaPasta.ogg')
                mixer.music.play()
                agarra_pasta=False
                t_fin=0
                t_fin_check=0

              l_hand_position_x=int(results2.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_INDEX].x*image_width)
              l_hand_position_y=int(results2.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_INDEX].y*image_height)
              lhx, lhy = l_hand_position_x, l_hand_position_y
              lhx-=30
              lhy-=80
              img_result=put_img.put_elements_in_viedo(lhx,lhy,img_result,img_cepillo_left)
            
              distance_mouth_hand = round(math.sqrt((mx-lhx)**2+(my-lhy)**2),2)   

              # cv2.putText(img_result, str(distance_mouth_hand), (70, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
              #             (0, 255, 0), 3)
              
              r_hand_position_x=int(results2.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_INDEX].x*image_width)
              r_hand_position_y=int(results2.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_INDEX].y*image_height)
              zhand=round(results2.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_INDEX].z,2)
              rhx, rhy = r_hand_position_x, r_hand_position_y
              rhx-=100
              rhy-=60

              if pasta_mano_derecha == 1:
                img_result=put_img.put_elements_in_viedo(350,300,img_result,img_pasta)

                #Dibujar rectangulo Toalla
                # start_agua = (350, 300) 
                # end_agua = (500, 400)
                # color = (0, 255, 0) 
                # thickness = 2
                # img_result = cv2.rectangle(img_result, start_agua, end_agua, color, thickness) 


                if (rhx>=350 and rhx<=500) and (rhy>=300 and rhy<=400):
                  
                  imagenGuia="recursos/autoc/cepilladodientes/PonersePasta.png"
                  imagenGuiaAlpha="recursos/autoc/cepilladodientes/Alphas/PonersePasta.png"
                  pasta_mano_derecha=2
                  mixer.music.load('recursos/autoc/cepilladodientes/check.ogg')
                  mixer.music.play()
                  t_ini_check=time.time()
              else:
                img_result=put_img.put_elements_in_viedo(rhx,rhy,img_result,img_pasta_right)
          
              if pasta_mano_derecha == 2 and not agarra_pasta and t_fin>=3:

                if poner_pasta and t_fin>=4 and t_fin_check>=3.5:
                    mixer.music.load('recursos/audios/lavadoDientes/ponerPastaCepillo.ogg')
                    mixer.music.play()
                    poner_pasta=False
                    t_fin=0
                    t_fin_check=0

                distance_hands=round(math.sqrt((lhx-rhx)**2+(lhy-rhy)**2),2) 
                if distance_hands>=85 and distance_hands<=200:
                  pasta_mano_derecha=3
                  imagenGuia="recursos/autoc/cepilladodientes/Cepillarse.png"
                  imagenGuiaAlpha="recursos/autoc/cepilladodientes/Alphas/Cepillarse.png"
                  mixer.music.load('recursos/autoc/cepilladodientes/check.ogg')
                  mixer.music.play()
                  t_ini_check=time.time()
                  
                  

              if pasta_mano_derecha == 3:
                img_result=put_img.put_elements_in_viedo(lhx,lhy,img_result2,img_cepilloPasta_left)
                diferencia_en_x=mx-lhx

                if not poner_pasta and t_fin>=4 and t_fin_check>=5.5:
                  if cepillarse_dientes:
                      mixer.music.load('recursos/audios/lavadoDientes/cepillarseDientes.ogg')
                      mixer.music.play()
                      cepillarse_dientes=False
                      t_fin=0

                if not cepillarse_dientes and t_fin>=3:
                  if arriba_abajo:
                    mixer.music.load('recursos/audios/lavadoDientes/arribaAbajoCepillado.ogg')
                    mixer.music.play()
                    arriba_abajo=False
                    t_fin=0

                if not arriba_abajo and t_fin>=4.5:
                  try:
                    if distance_mouth_hand >= 5 and distance_mouth_hand <= 200 and diferencia_en_x>=-105 and diferencia_en_x<=105:
                      bacteria+=1
                      img_result=lavado_dientes.dibujar_burbujas(7,img_result,mx,my)

                      if soundBrushin:
                          mixer.music.load('recursos/autoc/cepilladodientes/brushing.ogg')
                          mixer.music.play()
                          soundBrushin=False
                    elif not soundBrushin:
                      mixer.music.stop()
                      soundBrushin=True
                  except:
                    img_result=img_result
                

          #Dibujado y pocicionado de los pictogramas Guia (Estos cambian progresivamente)
          img_guia = np.array(Image.open(imagenGuia))
          img_guia=cv2.rotate(img_guia, cv2.ROTATE_90_COUNTERCLOCKWISE)
          alpha_guia= np.array(Image.open(imagenGuiaAlpha))
          img_overlay = img_guia[:, :, :3]
          img_overlay = cv2.cvtColor(img_guia, cv2.COLOR_BGR2RGB)
          put_img.overlay_image_alpha(img_result, img_overlay, 50, 240, alpha_guia)

        # end = time.time()
        # totalTime = end-start
        # fps = 1 / totalTime

        # cv2.putText(img_result, str(int(fps)), (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
        #                 (255, 0, 255), 3)

        try:
          img_result=put_img.put_elements_in_viedo(20,20,img_result,img_return)
          img_result=cv2.rotate(img_result, cv2.ROTATE_90_CLOCKWISE)
          cv2.imshow('lavado_dientes', img_result)
          cv2.setMouseCallback('lavado_dientes', lavado_dientes.click_event)
        except:
          image=put_img.put_elements_in_viedo(20,20,image,img_return)
          image=cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
          cv2.imshow('lavado_dientes', image)
          cv2.setMouseCallback('lavado_dientes', lavado_dientes.click_event)
        
        try:
          if return_action or (cv2.waitKey(5) & 0xFF == 27):
            mixer.music.load('recursos/audios/Bubble.ogg')
            mixer.music.play()
            cv2.destroyWindow('lavado_dientes')
            break
        except Exception as e:
          print(e)
        
        t_fin+=time.time()-t_ini
        try:
          t_fin_check+=time.time()-t_ini_check
          print(t_fin_check)
        except Exception as e:
          print(e)

        
        
    cap.release()
