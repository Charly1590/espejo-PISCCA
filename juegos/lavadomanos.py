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
from pygame import mixer

class lavado_manos():

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

    jabMano=False
    espuma=False
    manoR=False
    aguader=False
    aguaizq=False
    sucio=True
    brilloder=False
    brilloizq=False
    

    paso1=True
    paso2=False
    paso3=False
    paso4=False

    showtuto=False

    narrapaso0=True
    narrapaso1=False
    narrapaso2=False
    narrapaso3=False
    narrapaso4=False
    narrapaso5=False
    narrapaso6=False

    t_fin=0

    #Esta es la imagen que se muestra como guia durante el lavado de manos y su alpha
    imagenGuia="recursos/autoc/lavmanos/MojarseLasManos.png"
    imagenGuiaAlpha="recursos/autoc/lavmanos/Alphas/MojarseLasManos.png"


    
    

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
    #Para no complicarse con el cambio de tamaño e algunas imagenes (Bacterias, Burbujas)
    #Se cambio el tamaño de estas con un editor de imagenes (GIMP)


    #Bacterias 
    img_bacteria4 = np.array(Image.open("recursos/autoc/lavmanos/bacteria4.png"))
    img_bacteria4=cv2.rotate(img_bacteria4, cv2.ROTATE_90_COUNTERCLOCKWISE)
    #Alpha
    #cv2.imwrite("recursos/autoc/lavmanos/Alphas/bacteria4.png", alpha_mask_bacteria4)
    #alpha_mask_bacteria4= np.array(Image.open("recursos/autoc/lavmanos/Alphas/bacteria4.png"))

    alpha_mask_bacteria4= img_bacteria4[:, :, 3] / 255.0




    img_bacteria5 = np.array(Image.open("recursos/autoc/lavmanos/bacteria5.png"))
    img_bacteria5=cv2.rotate(img_bacteria5, cv2.ROTATE_90_COUNTERCLOCKWISE)
    #Creacion de Alpha
    alpha_mask_bacteria5= img_bacteria5[:, :, 3] / 255.0
    

    img_bacteria6 = np.array(Image.open("recursos/autoc/lavmanos/bacteria6.png"))
    img_bacteria6=cv2.rotate(img_bacteria6, cv2.ROTATE_90_COUNTERCLOCKWISE)
    #Creacion de Alpha
    alpha_mask_bacteria6= img_bacteria6[:, :, 3] / 255.0

    img_bacteria7 = np.array(Image.open("recursos/autoc/lavmanos/bacterias7.png"))
    img_bacteria7=cv2.rotate(img_bacteria7, cv2.ROTATE_90_COUNTERCLOCKWISE)
    #Creacion de Alpha
    alpha_mask_bacteria7= img_bacteria7[:, :, 3] / 255.0

    img_bacteria8 = np.array(Image.open("recursos/autoc/lavmanos/bacterias8.png"))
    img_bacteria8=cv2.rotate(img_bacteria8, cv2.ROTATE_90_COUNTERCLOCKWISE)
    #Creacion de Alpha
    alpha_mask_bacteria8= img_bacteria8[:, :, 3] / 255.0

    img_bacteria11 = np.array(Image.open("recursos/autoc/lavmanos/bacterias11.png"))
    img_bacteria11=cv2.rotate(img_bacteria11, cv2.ROTATE_90_COUNTERCLOCKWISE)
    #Creacion de Alpha
    alpha_mask_bacteria11= img_bacteria11[:, :, 3] / 255.0

    img_bacteria12 = np.array(Image.open("recursos/autoc/lavmanos/bacterias12.png"))
    img_bacteria12=cv2.rotate(img_bacteria12, cv2.ROTATE_90_COUNTERCLOCKWISE)
    #Creacion de Alpha
    alpha_mask_bacteria12= img_bacteria12[:, :, 3] / 255.0

    #Gota
    img_gota=np.array(Image.open("recursos/autoc/lavmanos/gota.png"))
    img_gota=cv2.rotate(img_gota, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    #Creacion de Alpha
    alpha_mask_gota= img_gota[:, :, 3] / 255.0


    #Jabon
    img_jabon = np.array(Image.open("recursos/autoc/lavmanos/jabon.png"))
    img_jabon=cv2.rotate(img_jabon, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    #Creacion de Alpha
    alpha_mask_jabon= img_jabon[:, :, 3] / 255.0

    #Burbujas
    img_burbuja1 = np.array(Image.open("recursos/autoc/lavmanos/Burbuja1.png"))
    img_burbuja1=cv2.rotate(img_burbuja1, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    #Creacion de Alpha
    alpha_mask_burbuja1= img_burbuja1[:, :, 3] / 255.0

    img_burbuja2 = np.array(Image.open("recursos/autoc/lavmanos/Burbuja2.png"))
    img_burbuja2=cv2.rotate(img_burbuja2, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    #Creacion de Alpha
    alpha_mask_burbuja2= img_burbuja2[:, :, 3] / 255.0
  
    img_burbuja3 = np.array(Image.open("recursos/autoc/lavmanos/Burbuja3.png"))
    img_burbuja3=cv2.rotate(img_burbuja3, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    #Creacion de Alpha
    alpha_mask_burbuja3= img_burbuja3[:, :, 3] / 255.0

    img_burbuja4 = np.array(Image.open("recursos/autoc/lavmanos/Burbuja4.png"))
    img_burbuja4=cv2.rotate(img_burbuja4, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    #Creacion de Alpha
    alpha_mask_burbuja4= img_burbuja4[:, :, 3] / 255.0


    #Toalla
    img_toalla= np.array(Image.open("recursos/autoc/lavmanos/toalla.png"))
    img_toalla=cv2.rotate(img_toalla, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    #Creacion de Alpha
    alpha_mask_toalla = img_toalla[:, :, 3] / 255.0


    #Espuma
    img_espuma= np.array(Image.open("recursos/autoc/lavmanos/espuma.png"))
    img_espuma=cv2.rotate(img_espuma, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    #Creacion de Alpha
    alpha_mask_espuma = img_espuma[:, :, 3] / 255.0

    #Grifo
    img_grifo= np.array(Image.open("recursos/autoc/lavmanos/grifo.png"))
    img_grifo=cv2.rotate(img_grifo, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    #Creacion de Alpha
    alpha_mask_grifo = img_grifo[:, :, 3] / 255.0

    #Brillo
    img_brillo= np.array(Image.open("recursos/autoc/lavmanos/brillo.png"))
    img_brillo=cv2.rotate(img_brillo, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    #Creacion de Alpha
    alpha_mask_brillo = img_brillo[:, :, 3] / 255.0

    #Atras
    img_return = np.array(Image.open("recursos/autoc/cepilladodientes/volver.png"))
    img_return=cv2.rotate(img_return, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    mixer.init() 
    t_fin=0
    soundWashingHands=True

    def dibujar_brilloizq (n, img_result):
      while (n > 0):
        xbl = random.randint(xr-75, xr+75)
        ybl = random.randint(yr-75, yr+75)
        xb = int(xbl)-80  
        yb = int(ybl)-50

        img_overlay = img_brillo[:, :, :3]
        img_overlay = cv2.cvtColor(img_overlay, cv2.COLOR_BGR2RGB)
        put_img.overlay_image_alpha(img_result, img_overlay, xb, yb, alpha_mask_brillo)

        n= n-1

    def dibujar_brilloder (n, img_result):
      while (n > 0):
        xbr = random.randint(xl-75, xl+75)
        ybr = random.randint(yl-75, yl+75)
        xb = int(xbr)-80
        yb = int(ybr)-50

        img_overlay = img_brillo[:, :, :3]
        img_overlay = cv2.cvtColor(img_overlay, cv2.COLOR_BGR2RGB)
        put_img.overlay_image_alpha(img_result, img_overlay, xb, yb, alpha_mask_brillo)

        n= n-1

    def dibujar_gotas (n, img_result):

      #img = np.array(img_result)
      #img_result = img[:, :, :3].copy()
      
      while (n > 0):
        
        xbl = random.randint(xl-150, xl+150)
        ybl = random.randint(yl-150, yl+150)
        xbr = random.randint(xr-150, xr+150)
        ybr = random.randint(yr-150, yr+150)
        xb = int((xbl+xbr)/2)  
        yb = int((ybl+ybr)/2) 

        
        img_overlay = img_gota[:, :, :3]
        img_overlay = cv2.cvtColor(img_overlay, cv2.COLOR_BGR2RGB)
        put_img.overlay_image_alpha(img_result, img_overlay, xb, yb, alpha_mask_gota)

        n= n-1




    def dibujar_burbujas (n, img_result):

      #img = np.array(img_result)
      #img_result = img[:, :, :3].copy()
      
      while (n > 0):
        
        xbl = random.randint(xl-150, xl+150)
        ybl = random.randint(yl-150, yl+150)
        xbr = random.randint(xr-150, xr+150)
        ybr = random.randint(yr-150, yr+150)
        xb = int((xbl+xbr)/2)  
        yb = int((ybl+ybr)/2) 

        #Random para definir el tipo de burbuja a cargar

        tp = random.randint(1,4)
        
        if tp == 1:
          img_overlay = img_burbuja1[:, :, :3]
          img_overlay = cv2.cvtColor(img_overlay, cv2.COLOR_BGR2RGB)
          put_img.overlay_image_alpha(img_result, img_overlay, xb, yb, alpha_mask_burbuja1)
        elif tp == 2:
          img_overlay = img_burbuja2[:, :, :3]
          img_overlay = cv2.cvtColor(img_overlay, cv2.COLOR_BGR2RGB)
          put_img.overlay_image_alpha(img_result, img_overlay, xb, yb, alpha_mask_burbuja2)
        elif tp == 3:
          img_overlay = img_burbuja3[:, :, :3]
          img_overlay = cv2.cvtColor(img_overlay, cv2.COLOR_BGR2RGB)
          put_img.overlay_image_alpha(img_result, img_overlay, xb, yb, alpha_mask_burbuja3)
        elif tp == 4:
          img_overlay = img_burbuja4[:, :, :3]
          img_overlay = cv2.cvtColor(img_overlay, cv2.COLOR_BGR2RGB)
          put_img.overlay_image_alpha(img_result, img_overlay, xb, yb, alpha_mask_burbuja4)
        
        n= n-1
        
    #Saludo inicial
    mixer.music.load('recursos/audios/lavadoManos/inicio.ogg')
    mixer.music.play()


    with mp_pose.Pose(
      
      #Ajustes de la red neuronal
      min_detection_confidence=0.6,
      min_tracking_confidence=0.6,
      model_complexity=0) as pose:

      while cap.isOpened():
        

        t_ini=time.time()

        #Narrado de Pasos
        if narrapaso0 and t_fin>=6.5:
          mixer.music.load('recursos/audios/lavadoManos/indicacionInicial.ogg')
          mixer.music.play()
          narrapaso0=False
          narrapaso1=True
          t_fin=0

        if narrapaso1 and t_fin>=6.5:
          mixer.music.load('recursos/audios/lavadoManos/mojarseManos.ogg')
          mixer.music.play()
          showtuto=True
          narrapaso1=False
          t_fin=0

        if narrapaso2 and t_fin>=1.0:
          mixer.music.load('recursos/audios/lavadoManos/cogerJabon.ogg')
          mixer.music.play()
          narrapaso2=False
          t_fin=0

        if narrapaso3 and t_fin>=1.0:
          mixer.music.load('recursos/audios/lavadoManos/fregarseManos.ogg')
          mixer.music.play()
          narrapaso3=False
          t_fin=0

        if narrapaso4 and t_fin>=1.0:
          mixer.music.load('recursos/audios/lavadoManos/quitarJAbon.ogg')
          mixer.music.play()
          narrapaso4=False
          t_fin=0

        if narrapaso5 and t_fin>=1.0:
          mixer.music.load('recursos/audios/lavadoManos/secarManos.ogg')
          mixer.music.play()
          narrapaso5=False
          t_fin=0

        if narrapaso6 and t_fin>=1.0:
          mixer.music.load('recursos/audios/lavadoManos/manosLimpias.ogg')
          mixer.music.play()
          narrapaso6=False
          

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
          xl=int(xl)
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
          xr=int(xr)
          yr=int(yr)
          


          #Dibujado
          xld=xl-100
          yld=yl-50

          xrd=xr-100
          yrd=yr-50

          #Tiempo maximo de las baterias
          if (cont_bact < 8):
            if sucio:
              img_overlay = img_bacteria4[:, :, :3]
              img_overlay = cv2.cvtColor(img_overlay, cv2.COLOR_BGR2RGB)
              #Metodo que recibe la imagen base, la imagen a dibuja su posicion y alpha 
              put_img.overlay_image_alpha(img_result, img_overlay, xld, yld, alpha_mask_bacteria4)
          else:
            img_overlay = img_espuma[:, :, :3]
            img_overlay = cv2.cvtColor(img_overlay, cv2.COLOR_BGR2RGB)
            #Metodo que recibe la imagen base, la imagen a dibuja su posicion y alpha 
            put_img.overlay_image_alpha(img_result, img_overlay, xld, yld, alpha_mask_espuma)


          if (cont_bact < 16):
            if sucio:
              img_overlay = img_bacteria4[:, :, :3]
              img_overlay = cv2.cvtColor(img_overlay, cv2.COLOR_BGR2RGB)
              put_img.overlay_image_alpha(img_result, img_overlay, xrd+45, yrd+45, alpha_mask_bacteria4)
          else:
            img_overlay = img_espuma[:, :, :3]
            img_overlay = cv2.cvtColor(img_overlay, cv2.COLOR_BGR2RGB)
            #Metodo que recibe la imagen base, la imagen a dibuja su posicion y alpha 
            put_img.overlay_image_alpha(img_result, img_overlay, xrd+45, yrd+45, alpha_mask_espuma)

          
          if (cont_bact < 24):
            if sucio:
              img_overlay = img_bacteria5[:, :, :3]
              img_overlay = cv2.cvtColor(img_overlay, cv2.COLOR_BGR2RGB)
              put_img.overlay_image_alpha(img_result, img_overlay, xld+45, yld+45, alpha_mask_bacteria5)
          else:
            img_overlay = img_espuma[:, :, :3]
            img_overlay = cv2.cvtColor(img_overlay, cv2.COLOR_BGR2RGB)
            #Metodo que recibe la imagen base, la imagen a dibuja su posicion y alpha 
            put_img.overlay_image_alpha(img_result, img_overlay, xld+45, yld+45, alpha_mask_espuma)



          if (cont_bact < 32):
            if sucio:
              img_overlay = img_bacteria6[:, :, :3]
              img_overlay = cv2.cvtColor(img_overlay, cv2.COLOR_BGR2RGB)
              put_img.overlay_image_alpha(img_result, img_overlay, xrd-45, yrd-45, alpha_mask_bacteria6)
          else:
            img_overlay = img_espuma[:, :, :3]
            img_overlay = cv2.cvtColor(img_overlay, cv2.COLOR_BGR2RGB)
            #Metodo que recibe la imagen base, la imagen a dibuja su posicion y alpha 
            put_img.overlay_image_alpha(img_result, img_overlay, xrd-45, yrd-45, alpha_mask_espuma)


          if (cont_bact < 40):
            if sucio:
              img_overlay = img_bacteria7[:, :, :3]
              img_overlay = cv2.cvtColor(img_overlay, cv2.COLOR_BGR2RGB)
              put_img.overlay_image_alpha(img_result, img_overlay, xld-45, yld-45, alpha_mask_bacteria7)
          else:
            img_overlay = img_espuma[:, :, :3]
            img_overlay = cv2.cvtColor(img_overlay, cv2.COLOR_BGR2RGB)
            #Metodo que recibe la imagen base, la imagen a dibuja su posicion y alpha 
            put_img.overlay_image_alpha(img_result, img_overlay, xld-45, yld-45, alpha_mask_espuma)


          if (cont_bact < 48):
            if sucio:
              img_overlay = img_bacteria8[:, :, :3]
              img_overlay = cv2.cvtColor(img_overlay, cv2.COLOR_BGR2RGB)
              put_img.overlay_image_alpha(img_result, img_overlay, xrd+45, yrd-45, alpha_mask_bacteria8)
          else:
            img_overlay = img_espuma[:, :, :3]
            img_overlay = cv2.cvtColor(img_overlay, cv2.COLOR_BGR2RGB)
            #Metodo que recibe la imagen base, la imagen a dibuja su posicion y alpha 
            put_img.overlay_image_alpha(img_result, img_overlay, xrd+45, yrd-45, alpha_mask_espuma)


          if (cont_bact < 56):
            if sucio:
              img_overlay = img_bacteria11[:, :, :3]
              img_overlay = cv2.cvtColor(img_overlay, cv2.COLOR_BGR2RGB)
              put_img.overlay_image_alpha(img_result, img_overlay, xld+35, yld-35, alpha_mask_bacteria11)
          else:
            img_overlay = img_espuma[:, :, :3]
            img_overlay = cv2.cvtColor(img_overlay, cv2.COLOR_BGR2RGB)
            #Metodo que recibe la imagen base, la imagen a dibuja su posicion y alpha 
            put_img.overlay_image_alpha(img_result, img_overlay, xld+35, yld-35, alpha_mask_espuma)


          if (cont_bact < 64):
            if sucio:
              img_overlay = img_bacteria12[:, :, :3]
              img_overlay = cv2.cvtColor(img_overlay, cv2.COLOR_BGR2RGB)
              put_img.overlay_image_alpha(img_result, img_overlay, xrd-35, yrd+35, alpha_mask_bacteria12)
          else:
            img_overlay = img_espuma[:, :, :3]
            img_overlay = cv2.cvtColor(img_overlay, cv2.COLOR_BGR2RGB)
            #Metodo que recibe la imagen base, la imagen a dibuja su posicion y alpha 
            put_img.overlay_image_alpha(img_result, img_overlay, xrd-35, yrd+35, alpha_mask_espuma)

            if sucio:
              mixer.music.load('recursos/autoc/lavmanos/check.ogg')
              mixer.music.play()

            aguader=False
            aguaizq=False
            sucio=False
            







          #Dibujado de los centros de las manos
          # img_result = cv2.circle(img_result, (xr,yr), radius=10, color=(0, 255, 0), thickness=5)
          # img_result = cv2.circle(img_result, (xl,yl), radius=10, color=(0, 255, 0), thickness=5)

          if paso1 or paso3:
            #Posicion del grifo
            xg=650
            yg=440 
            #Tamaño de la imagen
            xgf=xg+400
            ygf=yg+270

            #Agua control
            if (xl>=xg and xl<=xgf) and (yl>=yg  and yl<=ygf) and narrapaso1==False and narrapaso0==False and t_fin>=3.5:
              aguader=True

            if (xr>=xg and xr<=xgf) and (yr>=yg  and yr<=ygf) and narrapaso1==False and narrapaso0==False and t_fin>=3.5:
              aguaizq=True
              

            if aguader and aguaizq and paso1:
              t_fin=0
              mixer.music.load('recursos/autoc/lavmanos/check.ogg')
              mixer.music.play()
              imagenGuia="recursos/autoc/lavmanos/CogerJabon.png"
              imagenGuiaAlpha="recursos/autoc/lavmanos/Alphas/CogerJabon.png"
              paso1=False
              paso2=True
              narrapaso2=True

            #Dibujado y pocicionado del grifo 
            img_overlay = img_grifo[:, :, :3]
            img_overlay = cv2.cvtColor(img_overlay, cv2.COLOR_BGR2RGB)
            put_img.overlay_image_alpha(img_result, img_overlay, xg, yg, alpha_mask_grifo)

            #Dibujar rectangulo agua
            # start_agua = (xg, yg) 
            # end_agua = (xgf, ygf)
            # color = (255, 0, 0) 
            # thickness = 2
            # img_result = cv2.rectangle(img_result, start_agua, end_agua, color, thickness)

          if paso3:
            #Control del lavado de manos    
            if (xr>=xg and xr<=xgf) and (yr>=yg  and yr<=ygf) and (xl>=xg and xl<=xgf) and (yl>=yg  and yl<=ygf) and sucio ==False and t_fin>=3.5 and narrapaso4==False:
              cont_bact = cont_bact -1
              dibujar_gotas(12,img_result)
              if soundWashingHands:
                mixer.music.load('recursos/autoc/lavmanos/washing-hands.ogg')
                mixer.music.play()
                soundWashingHands=False
              if cont_bact < 5:
                imagenGuia="recursos/autoc/lavmanos/SecarseManos.png"
                imagenGuiaAlpha="recursos/autoc/lavmanos/Alphas/SecarseManos.png"
                mixer.music.load('recursos/autoc/lavmanos/check.ogg')
                mixer.music.play()
                paso4=True
                paso3=False
                narrapaso5=True
                t_fin=0
            elif not soundWashingHands:
              mixer.music.stop()
              soundWashingHands=True
                

          



          #Ajustes de la posicion del jabon y funcion
          if(paso2):
            if(jabMano):
              if(manoR):
                xj = xr-75
                yj = yr-100

              else:
                xj = xl-75
                yj = yl-100

            else:
              xj = 380
              yj = 260 

              #Tamaño de la imagen
              xjf = xj+154
              yjf = yj+200

              # 'Hitbox' del jabon 
              if ((xl>=xj and xl<=xjf) and (yl>=yj  and yl<=yjf) & aguader & aguaizq and t_fin>=3.5 and narrapaso2==False):
                
                
                jabMano=True
                espuma=True
                imagenGuia="recursos/autoc/lavmanos/FregarseManos.png"
                imagenGuiaAlpha="recursos/autoc/lavmanos/Alphas/FregarseManos.png"
                mixer.music.load('recursos/autoc/lavmanos/check.ogg')
                mixer.music.play()
                narrapaso3=True
                t_fin=0

              elif ((xr>=xj and xr<=xjf) and (yr>=yj  and yr<=yjf)& aguader & aguaizq and t_fin>=3.5 and narrapaso2==False):
                
                
                jabMano=True
                manoR=True
                espuma=True
                imagenGuia="recursos/autoc/lavmanos/FregarseManos.png"
                imagenGuiaAlpha="recursos/autoc/lavmanos/Alphas/FregarseManos.png"
                mixer.music.load('recursos/autoc/lavmanos/check.ogg')
                mixer.music.play()
                narrapaso3=True
                t_fin=0
            
            
            #Tamaño de la imagen (Uso solo para ver el rectangulo)
            # xjf = xj+154
            # yjf = yj+200
            

            #Dibujado del jabon
            img_overlay = img_jabon[:, :, :3]
            img_overlay = cv2.cvtColor(img_overlay, cv2.COLOR_BGR2RGB)
            put_img.overlay_image_alpha(img_result, img_overlay, xj, yj, alpha_mask_jabon) 
            
            #Dibujar rectangulo jabon
            # start_jabon = (xj, yj) 
            # end_jabon = (xjf, yjf)
            # color = (255, 0, 255) 
            # thickness = 2
            # img_result = cv2.rectangle(img_result, start_jabon, end_jabon, color, thickness)







          if(espuma):
            if (xl+170>=xr and xl-150<=xr) and (yl+170>=yr  and yl-150<=yr) and t_fin>=3.5 and narrapaso3==False :
              dibujar_burbujas(12,img_result)
              #Registro del tiempo para que desaparescan las bacterias
              if cont_bact < 65 :
                cont_bact = cont_bact +1
              else: 
                imagenGuia="recursos/autoc/lavmanos/QuitarJabon.png"
                imagenGuiaAlpha="recursos/autoc/lavmanos/Alphas/QuitarJabon.png"
                paso3=True
                paso2=False
                espuma=False
                narrapaso4=True
                t_fin=0
                



          if paso4:
            #Posicion del toalla
            xt=700
            yt=0
            #Tamaño de la imagen
            xtf=xt+275
            ytf=yt+250
            #Toalla control y brillo
            if (xl>=xt and xl<=xtf) and (yl>=yt  and yl<=ytf) and t_fin>=3.5 and narrapaso5==False:
              aguader=False
              if sucio == False:
                brilloder = True

            if (xr>=xt and xr<=xtf) and (yr>=yt  and yr<=ytf) and t_fin>=3.5 and narrapaso5==False:
              aguaizq=False
              if sucio == False:
                brilloizq = True


            if brilloder and brilloizq:
              mixer.music.load('recursos/autoc/lavmanos/check.ogg')
              mixer.music.play()
              paso4=False
              narrapaso6=True
              t_fin=0

            #Dibujado y pocicionado de la Toalla
            img_overlay = img_toalla[:, :, :3]
            img_overlay = cv2.cvtColor(img_toalla, cv2.COLOR_BGR2RGB)
            put_img.overlay_image_alpha(img_result, img_overlay, xt, yt, alpha_mask_toalla)


            #Dibujar rectangulo Toalla
            # start_agua = (xt, yt) 
            # end_agua = (xtf, ytf)
            # color = (0, 255, 0) 
            # thickness = 2
            # img_result = cv2.rectangle(img_result, start_agua, end_agua, color, thickness) 


          

          if aguaizq:
            img_overlay = img_gota[:, :, :3]
            img_overlay = cv2.cvtColor(img_overlay, cv2.COLOR_BGR2RGB)
            put_img.overlay_image_alpha(img_result, img_overlay, xr-15, yr+45, alpha_mask_gota)
            put_img.overlay_image_alpha(img_result, img_overlay, xr-65, yr-15, alpha_mask_gota)
            put_img.overlay_image_alpha(img_result, img_overlay, xr-35, yr+15, alpha_mask_gota)
            put_img.overlay_image_alpha(img_result, img_overlay, xr, yr-25, alpha_mask_gota)

          if aguader:
            img_overlay = img_gota[:, :, :3]
            img_overlay = cv2.cvtColor(img_overlay, cv2.COLOR_BGR2RGB)
            put_img.overlay_image_alpha(img_result, img_overlay, xl-80, yl-50, alpha_mask_gota)
            put_img.overlay_image_alpha(img_result, img_overlay, xl+10, yl+25, alpha_mask_gota)
            put_img.overlay_image_alpha(img_result, img_overlay, xl+5, yl-15, alpha_mask_gota)
            put_img.overlay_image_alpha(img_result, img_overlay, xl-70, yl+50, alpha_mask_gota)


          if brilloizq:
            dibujar_brilloizq(6,img_result)

          if brilloder:
            dibujar_brilloder(6,img_result)


         


          #Dibujado y pocicionado de los pictogramas Guia (Estos cambian progresivamente)
          if showtuto:
            img_guia = np.array(Image.open(imagenGuia))
            img_guia=cv2.rotate(img_guia, cv2.ROTATE_90_COUNTERCLOCKWISE)
            alpha_guia= np.array(Image.open(imagenGuiaAlpha))
            img_overlay = img_guia[:, :, :3]
            img_overlay = cv2.cvtColor(img_guia, cv2.COLOR_BGR2RGB)
            put_img.overlay_image_alpha(img_result, img_overlay, 50, 240, alpha_guia)



          #Esta linea se encarga de dibujar el esqueleto sobre la persona 
          #mp_drawing.draw_landmarks(img_result, results2.pose_landmarks, mp_pose.POSE_CONNECTIONS)
          

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
          cv2.imshow('lavado_manos', img_result)
          cv2.setMouseCallback('lavado_manos', lavado_manos.click_event)
        except:
          image=put_img.put_elements_in_viedo(20,10,image,img_return)
          image=cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
          cv2.imshow('lavado_manos', image)
          cv2.setMouseCallback('lavado_manos', lavado_manos.click_event)

        #Tecla de salida ESC
        try:
          if return_action or (cv2.waitKey(5) & 0xFF == 27):
            mixer.music.load('recursos/audios/Bubble.ogg')
            mixer.music.play()
            cv2.destroyWindow('lavado_manos')
            break
        except Exception as e:
          print(e)

        t_fin+=time.time()-t_ini

    cap.release()