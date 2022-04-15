import cv2
import mediapipe as mp
import time 
import numpy as np
from PIL import Image
from modulos.moduloPosicionarImgs import Posicionamiento as put_img
import screeninfo

class lavado_dientes():

  

  def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x," ",y, "sdahsjk")

  def actividad():
    
    cepillo_mano=False
    cepillo_mano_derecha=False
    cepillo_mano_izquierda=False

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


    img_boca = np.array(Image.open("recursos/autoc/cepilladodientes/Boca.png"))
    img_boca=cv2.rotate(img_boca, cv2.ROTATE_90_COUNTERCLOCKWISE)
    img_boca=cv2.resize(img_boca,(50,100))

    img_cepillo = np.array(Image.open("recursos/autoc/cepilladodientes/cepillo1.png"))
    img_cepillo=cv2.rotate(img_cepillo, cv2.ROTATE_90_COUNTERCLOCKWISE)
    img_cepillo=cv2.rotate(img_cepillo, cv2.ROTATE_90_COUNTERCLOCKWISE)
    img_cepillo=cv2.resize(img_cepillo,(200,50))

    img_cepillo_right= np.array(Image.open("recursos/autoc/cepilladodientes/cepillo1.png"))
    img_cepillo_right=cv2.rotate(img_cepillo_right, cv2.ROTATE_90_CLOCKWISE)
    img_cepillo_right=cv2.resize(img_cepillo_right,(50,200))

    img_cepillo_left= np.array(Image.open("recursos/autoc/cepilladodientes/cepillo1.png"))
    img_cepillo_left=cv2.rotate(img_cepillo_left, cv2.ROTATE_90_COUNTERCLOCKWISE)
    img_cepillo_left=cv2.resize(img_cepillo_left,(50,200))
    


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
        success, image = cap.read()

        image=cv2.flip(image, 1)

        start = time.time()

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image.flags.writeable = False

        results2 = pose.process(image)

        image.flags.writeable = True

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
          
        img_result=None  
        image_height, image_width, _ = image.shape

        if results2.pose_landmarks:

          """
            Se obtendran las coordenadas de la boca en la parte derecha
            se utiliza la misma logica para las otras partes del cuerpo
            si se decea obtener mas informacion de como llamar a otras 
            partes del cuerpo se puede encontrar en: https://google.github.io/mediapipe/solutions/pose
          """
          r_mouth_position_x=int(results2.pose_landmarks.landmark[mp_pose.PoseLandmark.MOUTH_RIGHT].x*image_width)
          r_mouth_position_y=int(results2.pose_landmarks.landmark[mp_pose.PoseLandmark.MOUTH_RIGHT].y*image_height)
          mx, my = r_mouth_position_x, r_mouth_position_y
          mx-=25
          my-=70
          img_result=put_img.put_elements_in_viedo(mx,my,image,img_boca)

          if cepillo_mano == False:
            l_hand_position_x=int(results2.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_INDEX].x*image_width)
            l_hand_position_y=int(results2.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_INDEX].y*image_height)
            lhx, lhy = l_hand_position_x, l_hand_position_y

            r_hand_position_x=int(results2.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_INDEX].x*image_width)
            r_hand_position_y=int(results2.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_INDEX].y*image_height)
            rhx, rhy = r_hand_position_x, r_hand_position_y

            if (rhx>=480 and rhx<=516) and (rhy>=300 and rhy<=496):
              cepillo_mano=True
              cepillo_mano_derecha=True
            if (lhx>=480 and lhx<=516) and (lhy>=300 and lhy<=496):
              cepillo_mano=True
              cepillo_mano_izquierda=True

            img_result=put_img.put_elements_in_viedo(300,300,img_result,img_cepillo)
          
          if cepillo_mano_derecha:
            r_hand_position_x=int(results2.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_INDEX].x*image_width)
            r_hand_position_y=int(results2.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_INDEX].y*image_height)
            rhx, rhy = r_hand_position_x, r_hand_position_y
            rhy-=90
            rhy+=60
            img_result=put_img.put_elements_in_viedo(rhx,rhy,img_result,img_cepillo_right)
          
          if cepillo_mano_izquierda:
            l_hand_position_x=int(results2.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_INDEX].x*image_width)
            l_hand_position_y=int(results2.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_INDEX].y*image_height)
            lhx, lhy = l_hand_position_x, l_hand_position_y
            lhy-=100
            lhy-=60
            img_result=put_img.put_elements_in_viedo(lhx,lhy,img_result,img_cepillo_left)
          

        end = time.time()
        totalTime = end-start
        fps = 1 / totalTime

        # cv2.putText(img_result, str(int(fps)), (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
        #                 (255, 0, 255), 3)
                        
        try:
          img_result=cv2.rotate(img_result, cv2.ROTATE_90_CLOCKWISE)
          cv2.imshow('lavado_dientes', img_result)
          cv2.setMouseCallback('lavado_dientes', lavado_dientes.click_event)
        except:
          image=cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
          cv2.imshow('lavado_dientes', image)
          cv2.setMouseCallback('lavado_dientes', lavado_dientes.click_event)
        if cv2.waitKey(5) & 0xFF == 27:
          cv2.destroyWindow('lavado_dientes')
          break
    cap.release()
