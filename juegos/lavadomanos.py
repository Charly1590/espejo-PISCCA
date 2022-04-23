import cv2
import mediapipe as mp
import time 
import numpy as np
from PIL import Image
from modulos.moduloPosicionarImgs import Posicionamiento as put_img
import screeninfo
import random


class lavado_manos():
  
  def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x," ",y, "pos")



  def actividad():
    #mp_drawing = mp.solutions.drawing_utils

    #Banderas globales
    jabMano=False
    espuma=False
    manoR=False
    aguader=False
    aguaizq=False
    sucio=True
    jabon=True
    brilloder=False
    brilloizq=False


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
          yl=int(yl)-50

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
          yr=int(yr)-50
          
          #Dibujado


          #Tiempo maximo de las baterias
          if (cont_bact < 5):
            if sucio:
              img_overlay = img_bacteria4[:, :, :3]
              img_overlay = cv2.cvtColor(img_overlay, cv2.COLOR_BGR2RGB)
              #Metodo que recive la imagen base, la imagen a dibuja su posicion y alpha 
              put_img.overlay_image_alpha(img_result, img_overlay, xl, yl, alpha_mask_bacteria4)
          else:
            img_overlay = img_espuma[:, :, :3]
            img_overlay = cv2.cvtColor(img_overlay, cv2.COLOR_BGR2RGB)
            #Metodo que recive la imagen base, la imagen a dibuja su posicion y alpha 
            put_img.overlay_image_alpha(img_result, img_overlay, xl, yl, alpha_mask_espuma)


          if (cont_bact < 10):
            if sucio:
              img_overlay = img_bacteria4[:, :, :3]
              img_overlay = cv2.cvtColor(img_overlay, cv2.COLOR_BGR2RGB)
              put_img.overlay_image_alpha(img_result, img_overlay, xr+45, yr+45, alpha_mask_bacteria4)
          else:
            img_overlay = img_espuma[:, :, :3]
            img_overlay = cv2.cvtColor(img_overlay, cv2.COLOR_BGR2RGB)
            #Metodo que recive la imagen base, la imagen a dibuja su posicion y alpha 
            put_img.overlay_image_alpha(img_result, img_overlay, xr+45, yr+45, alpha_mask_espuma)

          
          if (cont_bact < 15):
            if sucio:
              img_overlay = img_bacteria5[:, :, :3]
              img_overlay = cv2.cvtColor(img_overlay, cv2.COLOR_BGR2RGB)
              put_img.overlay_image_alpha(img_result, img_overlay, xl+45, yl+45, alpha_mask_bacteria5)
          else:
            img_overlay = img_espuma[:, :, :3]
            img_overlay = cv2.cvtColor(img_overlay, cv2.COLOR_BGR2RGB)
            #Metodo que recive la imagen base, la imagen a dibuja su posicion y alpha 
            put_img.overlay_image_alpha(img_result, img_overlay, xl+45, yl+45, alpha_mask_espuma)



          if (cont_bact < 20):
            if sucio:
              img_overlay = img_bacteria6[:, :, :3]
              img_overlay = cv2.cvtColor(img_overlay, cv2.COLOR_BGR2RGB)
              put_img.overlay_image_alpha(img_result, img_overlay, xr-45, yr-45, alpha_mask_bacteria6)
          else:
            img_overlay = img_espuma[:, :, :3]
            img_overlay = cv2.cvtColor(img_overlay, cv2.COLOR_BGR2RGB)
            #Metodo que recive la imagen base, la imagen a dibuja su posicion y alpha 
            put_img.overlay_image_alpha(img_result, img_overlay, xr-45, yr-45, alpha_mask_espuma)


          if (cont_bact < 25):
            if sucio:
              img_overlay = img_bacteria7[:, :, :3]
              img_overlay = cv2.cvtColor(img_overlay, cv2.COLOR_BGR2RGB)
              put_img.overlay_image_alpha(img_result, img_overlay, xl-45, yl-45, alpha_mask_bacteria7)
          else:
            img_overlay = img_espuma[:, :, :3]
            img_overlay = cv2.cvtColor(img_overlay, cv2.COLOR_BGR2RGB)
            #Metodo que recive la imagen base, la imagen a dibuja su posicion y alpha 
            put_img.overlay_image_alpha(img_result, img_overlay, xl-45, yl-45, alpha_mask_espuma)


          if (cont_bact < 30):
            if sucio:
              img_overlay = img_bacteria8[:, :, :3]
              img_overlay = cv2.cvtColor(img_overlay, cv2.COLOR_BGR2RGB)
              put_img.overlay_image_alpha(img_result, img_overlay, xr+45, yr-45, alpha_mask_bacteria8)
          else:
            img_overlay = img_espuma[:, :, :3]
            img_overlay = cv2.cvtColor(img_overlay, cv2.COLOR_BGR2RGB)
            #Metodo que recive la imagen base, la imagen a dibuja su posicion y alpha 
            put_img.overlay_image_alpha(img_result, img_overlay, xr+45, yr-45, alpha_mask_espuma)


          if (cont_bact < 35):
            if sucio:
              img_overlay = img_bacteria11[:, :, :3]
              img_overlay = cv2.cvtColor(img_overlay, cv2.COLOR_BGR2RGB)
              put_img.overlay_image_alpha(img_result, img_overlay, xl+35, yl-35, alpha_mask_bacteria11)
          else:
            img_overlay = img_espuma[:, :, :3]
            img_overlay = cv2.cvtColor(img_overlay, cv2.COLOR_BGR2RGB)
            #Metodo que recive la imagen base, la imagen a dibuja su posicion y alpha 
            put_img.overlay_image_alpha(img_result, img_overlay, xl+35, yl-35, alpha_mask_espuma)


          if (cont_bact < 40):
            if sucio:
              img_overlay = img_bacteria12[:, :, :3]
              img_overlay = cv2.cvtColor(img_overlay, cv2.COLOR_BGR2RGB)
              put_img.overlay_image_alpha(img_result, img_overlay, xr-35, yr+35, alpha_mask_bacteria12)
          else:
            img_overlay = img_espuma[:, :, :3]
            img_overlay = cv2.cvtColor(img_overlay, cv2.COLOR_BGR2RGB)
            #Metodo que recive la imagen base, la imagen a dibuja su posicion y alpha 
            put_img.overlay_image_alpha(img_result, img_overlay, xr-35, yr+35, alpha_mask_espuma)

            aguader=False
            aguaizq=False
            sucio=False
            espuma=False

          


          #Ajustes de la posicion del jabon y funcion
          #Funciones
          if(jabMano ):
            if(manoR):
              xj = xr
              yj = yr

            else:
              xj = xl
              yj = yl

          else:
            xj = 250
            yj = 250 
            # 'Hitbox' del jabon 
            if ((xl>=xj and xl<=xj+50) and (yl>=yj  and yl<=xj+50) & aguader & aguaizq):
              print('jabon der')
              
              jabMano=True
              espuma=True

            elif ((xr>=xj and xr<=xj+50) and (yr>=yj  and yr<=xj+50)& aguader & aguaizq):
              print('jabon izq')
              
              jabMano=True
              manoR=True
              espuma=True


          if(espuma):
            if (xl+160>=xr and xl-140<=xr) and (yl+160>=yr  and yl-140<=yr):
              dibujar_burbujas(12,img_result)
              #Registro del tiempo para que desaparescan las bacterias
              if cont_bact < 41 :
                cont_bact = cont_bact +1



          #Agua control
          if (xl>=900 and xl<=1200) and (yl>=550  and yl<=670):
            aguader=True

          if (xr>=900 and xr<=1200) and (yr>=550  and yr<=670):
            aguaizq=True


          #Toalla control y brillo
          if (xl>=900 and xl<=1200) and (yl>=0  and yl<=100):
            aguader=False
            if sucio == False:
              brilloder = True

          if (xr>=900 and xr<=1200) and (yr>=10  and yr<=100):
            aguaizq=False
            if sucio == False:
              brilloizq = True


          
          if (xr>=900 and xr<=1200) and (yr>=550  and yr<=670) and (xl>=900 and xl<=1200) and (yl>=550  and yl<=670) and sucio ==False:
            cont_bact = cont_bact -1
            jabon=False


              
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
            img_overlay = img_brillo[:, :, :3]
            img_overlay = cv2.cvtColor(img_overlay, cv2.COLOR_BGR2RGB)
            put_img.overlay_image_alpha(img_result, img_overlay, xr+15, yr+45, alpha_mask_brillo)
            put_img.overlay_image_alpha(img_result, img_overlay, xr-35, yr-15, alpha_mask_brillo)
            put_img.overlay_image_alpha(img_result, img_overlay, xr-5, yr+15, alpha_mask_brillo)
            put_img.overlay_image_alpha(img_result, img_overlay, xr+30, yr-25, alpha_mask_brillo)

          if brilloder:
            img_overlay = img_brillo[:, :, :3]
            img_overlay = cv2.cvtColor(img_overlay, cv2.COLOR_BGR2RGB)
            put_img.overlay_image_alpha(img_result, img_overlay, xl-50, yl-50, alpha_mask_brillo)
            put_img.overlay_image_alpha(img_result, img_overlay, xl+40, yl+25, alpha_mask_brillo)
            put_img.overlay_image_alpha(img_result, img_overlay, xl+35, yl-15, alpha_mask_brillo)
            put_img.overlay_image_alpha(img_result, img_overlay, xl-50, yl+50, alpha_mask_brillo)


          #Dibujado
          #img = np.array(img_result)
          #img_result = img[:, :, :3].copy()
          if jabon:
            img_overlay = img_jabon[:, :, :3]
            img_overlay = cv2.cvtColor(img_overlay, cv2.COLOR_BGR2RGB)
            put_img.overlay_image_alpha(img_result, img_overlay, xj, yj, alpha_mask_jabon) 


          #Dibujado y pocicionado del grifo 
          img_overlay = img_grifo[:, :, :3]
          img_overlay = cv2.cvtColor(img_overlay, cv2.COLOR_BGR2RGB)
          put_img.overlay_image_alpha(img_result, img_overlay, 1000, 550, alpha_mask_grifo) 

          #Dibujado y pocicionado del espuma
          #img_overlay = img_espuma[:, :, :3]
          #img_overlay = cv2.cvtColor(img_espuma, cv2.COLOR_BGR2RGB)
          #put_img.overlay_image_alpha(img_result, img_overlay, xe, ye, alpha_mask_espuma)

          #Dibujado y pocicionado de la Toalla
          img_overlay = img_toalla[:, :, :3]
          img_overlay = cv2.cvtColor(img_toalla, cv2.COLOR_BGR2RGB)
          put_img.overlay_image_alpha(img_result, img_overlay, 1000, 0, alpha_mask_toalla)



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
          cv2.setMouseCallback('lavado_manos', lavado_manos.click_event)
        except:
          image=cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
          cv2.imshow('lavado_manos', image)
          cv2.setMouseCallback('lavado_manos', lavado_manos.click_event)

        #Tecla de salida ESC
        if cv2.waitKey(5) & 0xFF == 27:
          cv2.destroyWindow('lavado_manos')
          break
    cap.release()