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

class lavado_manos():

  return_action=False
  
  def click_event(event, x, y, flags, params):
    global return_action
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x," ",y, "sdahsjk")
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
    cap = cv2.VideoCapture(2)
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
    #Creacion de Alpha
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
    img_gota=cv2.resize(img_gota,(60,30))
    #Creacion de Alpha
    alpha_mask_gota= img_gota[:, :, 3] / 255.0


    #Jabon
    img_jabon = np.array(Image.open("recursos/autoc/lavmanos/jabon.png"))
    img_jabon=cv2.rotate(img_jabon, cv2.ROTATE_90_COUNTERCLOCKWISE)
    img_jabon=cv2.resize(img_jabon,(120,160))
    #Creacion de Alpha
    alpha_mask_jabon= img_jabon[:, :, 3] / 255.0

    #Burbujas
    img_burbuja1 = np.array(Image.open("recursos/autoc/lavmanos/Burbuja1.png"))
    img_burbuja1=cv2.rotate(img_burbuja1, cv2.ROTATE_90_COUNTERCLOCKWISE)
    #img_burbuja=cv2.resize(img_burbuja,(100 ,100 ))
    #Creacion de Alpha
    alpha_mask_burbuja1= img_burbuja1[:, :, 3] / 255.0

    img_burbuja2 = np.array(Image.open("recursos/autoc/lavmanos/Burbuja2.png"))
    img_burbuja2=cv2.rotate(img_burbuja2, cv2.ROTATE_90_COUNTERCLOCKWISE)
    #img_burbuja=cv2.resize(img_burbuja,(100 ,100 ))
    #Creacion de Alpha
    alpha_mask_burbuja2= img_burbuja2[:, :, 3] / 255.0
  
    img_burbuja3 = np.array(Image.open("recursos/autoc/lavmanos/Burbuja3.png"))
    img_burbuja3=cv2.rotate(img_burbuja3, cv2.ROTATE_90_COUNTERCLOCKWISE)
    #img_burbuja=cv2.resize(img_burbuja,(100 ,100 ))
    #Creacion de Alpha
    alpha_mask_burbuja3= img_burbuja3[:, :, 3] / 255.0

    img_burbuja4 = np.array(Image.open("recursos/autoc/lavmanos/Burbuja4.png"))
    img_burbuja4=cv2.rotate(img_burbuja4, cv2.ROTATE_90_COUNTERCLOCKWISE)
    #img_burbuja=cv2.resize(img_burbuja,(100 ,100 ))
    #Creacion de Alpha
    alpha_mask_burbuja4= img_burbuja4[:, :, 3] / 255.0


    #Toalla
    img_toalla= np.array(Image.open("recursos/autoc/lavmanos/toalla.png"))
    img_toalla=cv2.rotate(img_toalla, cv2.ROTATE_90_COUNTERCLOCKWISE)
    img_toalla=cv2.resize(img_toalla,(230,150))
    #Creacion de Alpha
    alpha_mask_toalla = img_toalla[:, :, 3] / 255.0


    #Espuma
    img_espuma= np.array(Image.open("recursos/autoc/lavmanos/espuma.png"))
    img_espuma=cv2.rotate(img_espuma, cv2.ROTATE_90_COUNTERCLOCKWISE)
    img_espuma=cv2.resize(img_espuma,(150,150))
    #Creacion de Alpha
    alpha_mask_espuma = img_espuma[:, :, 3] / 255.0

    #Grifo
    img_grifo= np.array(Image.open("recursos/autoc/lavmanos/grifo.png"))
    img_grifo=cv2.rotate(img_grifo, cv2.ROTATE_90_COUNTERCLOCKWISE)
    img_grifo=cv2.resize(img_grifo,(330,180))
    #Creacion de Alpha
    alpha_mask_grifo = img_grifo[:, :, 3] / 255.0

    #Brillo
    img_brillo= np.array(Image.open("recursos/autoc/lavmanos/brillo.png"))
    img_brillo=cv2.rotate(img_brillo, cv2.ROTATE_90_COUNTERCLOCKWISE)
    img_brillo=cv2.resize(img_brillo,(50,100))
    #Creacion de Alpha
    alpha_mask_brillo = img_brillo[:, :, 3] / 255.0

    #Atras
    img_return = np.array(Image.open("recursos/menu/return.png"))
    img_return=cv2.rotate(img_return, cv2.ROTATE_90_COUNTERCLOCKWISE)
    img_return=cv2.resize(img_return,(150,150))


    def dibujar_brilloizq (n, img_result):
      while (n > 0):
        xbl = random.randint(xr-75, xr+75)
        ybl = random.randint(yr-75, yr+75)
        xb = int(xbl)  
        yb = int(ybl)

        img_overlay = img_brillo[:, :, :3]
        img_overlay = cv2.cvtColor(img_overlay, cv2.COLOR_BGR2RGB)
        put_img.overlay_image_alpha(img_result, img_overlay, xb, yb, alpha_mask_brillo)

        n= n-1

    def dibujar_brilloder (n, img_result):
      while (n > 0):
        xbr = random.randint(xl-75, xl+75)
        ybr = random.randint(yl-75, yl+75)
        xb = int(xbr)  
        yb = int(ybr)

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
        
        
      


    with mp_pose.Pose(
      
      #Ajustes de la red neuronal
      min_detection_confidence=0.6,
      min_tracking_confidence=0.6,
      model_complexity=0) as pose:
      


      while cap.isOpened():

        #Sonido Check
        checkSound = multiprocessing.Process(target=playsound, args=("recursos/autoc/cepilladodientes/check.mp3",))

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
              #Metodo que recive la imagen base, la imagen a dibuja su posicion y alpha 
              put_img.overlay_image_alpha(img_result, img_overlay, xld, yld, alpha_mask_bacteria4)
          else:
            img_overlay = img_espuma[:, :, :3]
            img_overlay = cv2.cvtColor(img_overlay, cv2.COLOR_BGR2RGB)
            #Metodo que recive la imagen base, la imagen a dibuja su posicion y alpha 
            put_img.overlay_image_alpha(img_result, img_overlay, xld, yld, alpha_mask_espuma)


          if (cont_bact < 16):
            if sucio:
              img_overlay = img_bacteria4[:, :, :3]
              img_overlay = cv2.cvtColor(img_overlay, cv2.COLOR_BGR2RGB)
              put_img.overlay_image_alpha(img_result, img_overlay, xrd+45, yrd+45, alpha_mask_bacteria4)
          else:
            img_overlay = img_espuma[:, :, :3]
            img_overlay = cv2.cvtColor(img_overlay, cv2.COLOR_BGR2RGB)
            #Metodo que recive la imagen base, la imagen a dibuja su posicion y alpha 
            put_img.overlay_image_alpha(img_result, img_overlay, xrd+45, yrd+45, alpha_mask_espuma)

          
          if (cont_bact < 24):
            if sucio:
              img_overlay = img_bacteria5[:, :, :3]
              img_overlay = cv2.cvtColor(img_overlay, cv2.COLOR_BGR2RGB)
              put_img.overlay_image_alpha(img_result, img_overlay, xld+45, yld+45, alpha_mask_bacteria5)
          else:
            img_overlay = img_espuma[:, :, :3]
            img_overlay = cv2.cvtColor(img_overlay, cv2.COLOR_BGR2RGB)
            #Metodo que recive la imagen base, la imagen a dibuja su posicion y alpha 
            put_img.overlay_image_alpha(img_result, img_overlay, xld+45, yld+45, alpha_mask_espuma)



          if (cont_bact < 32):
            if sucio:
              img_overlay = img_bacteria6[:, :, :3]
              img_overlay = cv2.cvtColor(img_overlay, cv2.COLOR_BGR2RGB)
              put_img.overlay_image_alpha(img_result, img_overlay, xrd-45, yrd-45, alpha_mask_bacteria6)
          else:
            img_overlay = img_espuma[:, :, :3]
            img_overlay = cv2.cvtColor(img_overlay, cv2.COLOR_BGR2RGB)
            #Metodo que recive la imagen base, la imagen a dibuja su posicion y alpha 
            put_img.overlay_image_alpha(img_result, img_overlay, xrd-45, yrd-45, alpha_mask_espuma)


          if (cont_bact < 40):
            if sucio:
              img_overlay = img_bacteria7[:, :, :3]
              img_overlay = cv2.cvtColor(img_overlay, cv2.COLOR_BGR2RGB)
              put_img.overlay_image_alpha(img_result, img_overlay, xld-45, yld-45, alpha_mask_bacteria7)
          else:
            img_overlay = img_espuma[:, :, :3]
            img_overlay = cv2.cvtColor(img_overlay, cv2.COLOR_BGR2RGB)
            #Metodo que recive la imagen base, la imagen a dibuja su posicion y alpha 
            put_img.overlay_image_alpha(img_result, img_overlay, xld-45, yld-45, alpha_mask_espuma)


          if (cont_bact < 48):
            if sucio:
              img_overlay = img_bacteria8[:, :, :3]
              img_overlay = cv2.cvtColor(img_overlay, cv2.COLOR_BGR2RGB)
              put_img.overlay_image_alpha(img_result, img_overlay, xrd+45, yrd-45, alpha_mask_bacteria8)
          else:
            img_overlay = img_espuma[:, :, :3]
            img_overlay = cv2.cvtColor(img_overlay, cv2.COLOR_BGR2RGB)
            #Metodo que recive la imagen base, la imagen a dibuja su posicion y alpha 
            put_img.overlay_image_alpha(img_result, img_overlay, xrd+45, yrd-45, alpha_mask_espuma)


          if (cont_bact < 56):
            if sucio:
              img_overlay = img_bacteria11[:, :, :3]
              img_overlay = cv2.cvtColor(img_overlay, cv2.COLOR_BGR2RGB)
              put_img.overlay_image_alpha(img_result, img_overlay, xld+35, yld-35, alpha_mask_bacteria11)
          else:
            img_overlay = img_espuma[:, :, :3]
            img_overlay = cv2.cvtColor(img_overlay, cv2.COLOR_BGR2RGB)
            #Metodo que recive la imagen base, la imagen a dibuja su posicion y alpha 
            put_img.overlay_image_alpha(img_result, img_overlay, xld+35, yld-35, alpha_mask_espuma)


          if (cont_bact < 64):
            if sucio:
              img_overlay = img_bacteria12[:, :, :3]
              img_overlay = cv2.cvtColor(img_overlay, cv2.COLOR_BGR2RGB)
              put_img.overlay_image_alpha(img_result, img_overlay, xrd-35, yrd+35, alpha_mask_bacteria12)
          else:
            img_overlay = img_espuma[:, :, :3]
            img_overlay = cv2.cvtColor(img_overlay, cv2.COLOR_BGR2RGB)
            #Metodo que recive la imagen base, la imagen a dibuja su posicion y alpha 
            put_img.overlay_image_alpha(img_result, img_overlay, xrd-35, yrd+35, alpha_mask_espuma)

            if sucio:
              checkSound.start()

            aguader=False
            aguaizq=False
            sucio=False
            







          #Dibujado de los centros de las manos
          img_result = cv2.circle(img_result, (xr,yr), radius=10, color=(0, 255, 0), thickness=5)
          img_result = cv2.circle(img_result, (xl,yl), radius=10, color=(0, 255, 0), thickness=5)

          if paso1 or paso3:
            #Posicion del grifo
            xg=650
            yg=550 


            #Agua control
            if (xl>=xg and xl<=xg+400) and (yl>=yg  and yl<=yg+170):
              aguader=True

            if (xr>=xg and xr<=xg+400) and (yr>=yg  and yr<=yg+170):
              aguaizq=True
              

            if aguader and aguaizq and paso1:
              checkSound.start()
              paso1=False
              paso2=True

            #Dibujado y pocicionado del grifo 
            img_overlay = img_grifo[:, :, :3]
            img_overlay = cv2.cvtColor(img_overlay, cv2.COLOR_BGR2RGB)
            put_img.overlay_image_alpha(img_result, img_overlay, xg, yg, alpha_mask_grifo)

            #Dibujar rectangulo agua
            start_agua = (xg, yg) 
            end_agua = (xg+400, yg+170)
            color = (255, 0, 0) 
            thickness = 2
            img_result = cv2.rectangle(img_result, start_agua, end_agua, color, thickness)

          if paso3:
            #Control del lavado de manos    
            if (xr>=xg and xr<=xg+400) and (yr>=yg  and yr<=yg+170) and (xl>=xg and xl<=xg+400) and (yl>=yg  and yl<=yg+170) and sucio ==False:
              cont_bact = cont_bact -1
              dibujar_gotas(12,img_result)
              if cont_bact < 5:
                checkSound.start()
                paso4=True
                paso3=False
                

          



          #Ajustes de la posicion del jabon y funcion
          if(paso2):
            if(jabMano):
              if(manoR):
                xj = xr
                yj = yr

              else:
                xj = xl
                yj = yl

            else:
              xj = 350
              yj = 250 
              # 'Hitbox' del jabon 
              if ((xl>=xj and xl<=xj+50) and (yl>=yj  and yl<=xj+50) & aguader & aguaizq):
                print('jabon der')
                
                jabMano=True
                espuma=True
                checkSound.start()

              elif ((xr>=xj and xr<=xj+50) and (yr>=yj  and yr<=xj+50)& aguader & aguaizq):
                print('jabon izq')
                
                jabMano=True
                manoR=True
                espuma=True
                checkSound.start()

            #Dibujado del jabon
            img_overlay = img_jabon[:, :, :3]
            img_overlay = cv2.cvtColor(img_overlay, cv2.COLOR_BGR2RGB)
            put_img.overlay_image_alpha(img_result, img_overlay, xj, yj, alpha_mask_jabon) 
            
            #Dibujar rectangulo agua
            start_agua = (xj, yj) 
            end_agua = (xj+50, yj+50)
            color = (255, 0, 255) 
            thickness = 2
            img_result = cv2.rectangle(img_result, start_agua, end_agua, color, thickness)







          if(espuma):
            if (xl+170>=xr and xl-150<=xr) and (yl+170>=yr  and yl-150<=yr):
              dibujar_burbujas(12,img_result)
              #Registro del tiempo para que desaparescan las bacterias
              if cont_bact < 65 :
                cont_bact = cont_bact +1
              else:  
                paso3=True
                paso2=False
                espuma=False
                



          if paso4:
            #Posicion del toalla
            xt=700
            yt=0
            #Toalla control y brillo
            if (xl>=xt and xl<=xt+100) and (yl>=yt  and yl<=yt+100):
              aguader=False
              if sucio == False:
                brilloder = True

            if (xr>=xt and xr<=xt+100) and (yr>=yt  and yr<=yt+100):
              aguaizq=False
              if sucio == False:
                brilloizq = True


            if brilloder and brilloizq:
              checkSound.start()
              paso4=False


            #Dibujado y pocicionado de la Toalla
            img_overlay = img_toalla[:, :, :3]
            img_overlay = cv2.cvtColor(img_toalla, cv2.COLOR_BGR2RGB)
            put_img.overlay_image_alpha(img_result, img_overlay, xt, yt, alpha_mask_toalla)



            #Dibujar rectangulo Toalla
            start_agua = (xt, yt) 
            end_agua = (xt+100, yt+100)
            color = (0, 255, 0) 
            thickness = 2
            img_result = cv2.rectangle(img_result, start_agua, end_agua, color, thickness) 




          if aguaizq:
            img_overlay = img_gota[:, :, :3]
            img_overlay = cv2.cvtColor(img_overlay, cv2.COLOR_BGR2RGB)
            put_img.overlay_image_alpha(img_result, img_overlay, xr+15, yr+45, alpha_mask_gota)
            put_img.overlay_image_alpha(img_result, img_overlay, xr-35, yr-15, alpha_mask_gota)
            put_img.overlay_image_alpha(img_result, img_overlay, xr-5, yr+15, alpha_mask_gota)
            put_img.overlay_image_alpha(img_result, img_overlay, xr+30, yr-25, alpha_mask_gota)

          if aguader:
            img_overlay = img_gota[:, :, :3]
            img_overlay = cv2.cvtColor(img_overlay, cv2.COLOR_BGR2RGB)
            put_img.overlay_image_alpha(img_result, img_overlay, xl-50, yl-50, alpha_mask_gota)
            put_img.overlay_image_alpha(img_result, img_overlay, xl+40, yl+25, alpha_mask_gota)
            put_img.overlay_image_alpha(img_result, img_overlay, xl+35, yl-15, alpha_mask_gota)
            put_img.overlay_image_alpha(img_result, img_overlay, xl-50, yl+50, alpha_mask_gota)


          if brilloizq:
            dibujar_brilloizq(6,img_result)

          if brilloder:
            dibujar_brilloder(6,img_result)



          #Dibujado y pocicionado del espuma
          #img_overlay = img_espuma[:, :, :3]
          #img_overlay = cv2.cvtColor(img_espuma, cv2.COLOR_BGR2RGB)
          #put_img.overlay_image_alpha(img_result, img_overlay, xe, ye, alpha_mask_espuma)

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
          img_result=put_img.put_elements_in_viedo(20,20,img_result,img_return)
          img_result=cv2.rotate(img_result, cv2.ROTATE_90_CLOCKWISE)
          cv2.imshow('lavado_manos', img_result)
          cv2.setMouseCallback('lavado_manos', lavado_manos.click_event)
        except:
          image=put_img.put_elements_in_viedo(20,20,image,img_return)
          image=cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
          cv2.imshow('lavado_manos', image)
          cv2.setMouseCallback('lavado_manos', lavado_manos.click_event)

        #Tecla de salida ESC
        try:
          if return_action or (cv2.waitKey(5) & 0xFF == 27):
            cv2.destroyWindow('lavado_manos')
            break
        except Exception as e:
          print(e)

    cap.release()