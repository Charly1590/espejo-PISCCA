import cv2
from defer import return_value
import numpy as np
from PIL import Image
from modulos.moduloPosicionarImgs import Posicionamiento as put_img
import screeninfo
import random

from threading import Thread
from playsound import playsound
import time
from pygame import mixer

class introduccion_lavado_manos():
  
  return_action=False

  def click_event(event, x, y, flags, params):
    global return_action
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x," ",y, "sdahsjk")
        if (x>=600 and x<=750) and (y>=30 and y<=140):
          return_action=True

  def secuencia():
    
    global return_action
    return_action=False

    """
      Se pondra a la ventana en pantalla completa para evitar
      los bordes de la interfaz del sistema
    """
    screen = screeninfo.get_monitors()[0] 
    cv2.namedWindow('image3', cv2.WND_PROP_FULLSCREEN)
    cv2.moveWindow('image3', screen.x - 1, screen.y - 1)
    cv2.setWindowProperty('image3', cv2.WND_PROP_FULLSCREEN,  cv2.WINDOW_FULLSCREEN) 
    
    # Carga de imagenes
    img = cv2.imread('recursos/menu/Fondo.jpg')
    img = cv2.resize(img,(1360,768))
    img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    img = put_img.put_image_in_any_position(610, 20, img, "recursos/menu/volver.png")

    introduccion=True
    primera_indicacion=True
    mojarse_manos=True
    coger_jabon=True
    fregarse_manos=True
    quitar_jabon=True
    secar_manos=True
    img_result=None

    showtuto=False
    cicloanim=0.4
    posianim=0
    imgentuto=""

    mixer.init() 
    t_fin=0
    while 3==3:

        
        if cicloanim <= 1.0 :
          cicloanim+=0.06
          posianim+=16
          
        
        t_ini=time.time()
        try:

          if introduccion:
              img_result=put_img.put_image_in_any_position(50, 280, img, "recursos/personajes/Doctora.png")
              cv2.imshow('image3', img_result)
              cv2.setMouseCallback('image3', introduccion_lavado_manos.click_event)
              mixer.music.load('recursos/audios/lavadoManos/inicio.ogg')
              mixer.music.play()
              introduccion=False

          if not introduccion and t_fin>=6.5:
            if primera_indicacion: 
              cv2.setMouseCallback('image3', introduccion_lavado_manos.click_event)
              mixer.music.load('recursos/audios/lavadoManos/indicacionInicial.ogg')
              mixer.music.play()
              primera_indicacion=False
              
              t_fin=0

          if not primera_indicacion and t_fin>=6.5:
            
            if mojarse_manos:
              imgentuto = "recursos/introducciones/lavadomanos/MojarseLasManos.png"
              showtuto=True
              cicloanim=0.4
              posianim=0

              cv2.setMouseCallback('image3', introduccion_lavado_manos.click_event)
              mixer.music.load('recursos/audios/lavadoManos/mojarseManos.ogg')
              mixer.music.play()
              mojarse_manos=False
              t_fin=0


          if not mojarse_manos and t_fin>=3.5:
            if coger_jabon:
              imgentuto = "recursos/introducciones/lavadomanos/CogerJabon.png"
              cicloanim=0.4
              posianim=0
              

              mixer.music.load('recursos/audios/lavadoManos/cogerJabon.ogg')
              mixer.music.play()
              coger_jabon=False
              t_fin=0

    
          if not coger_jabon and t_fin>=3.5:
            if fregarse_manos:
              imgentuto =  "recursos/introducciones/lavadomanos/FregarseManos.png"
              cicloanim=0.4
              posianim=0


              mixer.music.load('recursos/audios/lavadoManos/fregarseManos.ogg')
              mixer.music.play()
              fregarse_manos=False
              t_fin=0

          if not fregarse_manos and t_fin>=3.5:
            if quitar_jabon:
              imgentuto = "recursos/introducciones/lavadomanos/QuitarJabon.png"
              cicloanim=0.4
              posianim=0


              mixer.music.load('recursos/audios/lavadoManos/quitarJAbon.ogg')
              mixer.music.play()
              quitar_jabon=False
              t_fin=0

          if not quitar_jabon and t_fin>=3.5:
            if secar_manos:
              imgentuto = "recursos/introducciones/lavadomanos/SecarseManos.png"
              cicloanim=0.4
              posianim=0


              mixer.music.load('recursos/audios/lavadoManos/secarManos.ogg')
              mixer.music.play()
              secar_manos=False


          if showtuto:
            img_result=put_img.put_image_in_any_position_with_resize(300-posianim, 530-posianim, img, imgentuto ,int(500*cicloanim),int(500*cicloanim))
            cv2.imshow('image3', img_result)
            cv2.setMouseCallback('image3', introduccion_lavado_manos.click_event)


        except Exception as e:
          print(e)

        try:
            if return_action or (cv2.waitKey(5) & 0xFF == 27):
                mixer.music.load('recursos/audios/Bubble.ogg')
                mixer.music.play()
                cv2.destroyWindow('image3')
                break
        except Exception as e:
            print(e)

        t_fin+=time.time()-t_ini