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
    
    
    introduccion_thread=None
    primera_indicacion_thread=None
    mojarse_manos_thread=None
    coger_jabon_thread=None
    fregarse_manos_thread=None
    quitar_jabon_thread=None
    secar_manos_thread=None

    img_result=None

    mixer.init() 
    t_fin=0
    while 3==3:

        t_ini=time.time()
        try:

          if introduccion:
              cv2.imshow('image3', img)
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
              img_result=put_img.put_image_in_any_position(150, 380, img, "recursos/introducciones/lavadomanos/MojarseLasManos.png")
              cv2.imshow('image3', img_result)
              cv2.setMouseCallback('image3', introduccion_lavado_manos.click_event)
              mixer.music.load('recursos/audios/lavadoManos/mojarseManos.ogg')
              mixer.music.play()
              mojarse_manos=False
              t_fin=0

          if not mojarse_manos and t_fin>=2.5:
            if coger_jabon:
              img_result=put_img.put_image_in_any_position(150, 380, img, "recursos/introducciones/lavadomanos/CogerJabon.png")
              cv2.imshow('image3', img_result)
              cv2.setMouseCallback('image3', introduccion_lavado_manos.click_event)
              mixer.music.load('recursos/audios/lavadoManos/cogerJabon.ogg')
              mixer.music.play()
              coger_jabon=False
              t_fin=0

    
          if not coger_jabon and t_fin>=2.5:
            if fregarse_manos:
              img_result=put_img.put_image_in_any_position(150, 380, img, "recursos/introducciones/lavadomanos/FregarseManos.png")
              cv2.imshow('image3', img_result)
              cv2.setMouseCallback('image3', introduccion_lavado_manos.click_event)
              mixer.music.load('recursos/audios/lavadoManos/fregarseManos.ogg')
              mixer.music.play()
              fregarse_manos=False
              t_fin=0

          if not fregarse_manos and t_fin>=2.9:
            if quitar_jabon:
              img_result=put_img.put_image_in_any_position(150, 380, img, "recursos/introducciones/lavadomanos/QuitarJabon.png")
              cv2.imshow('image3', img_result)
              cv2.setMouseCallback('image3', introduccion_lavado_manos.click_event)
              mixer.music.load('recursos/audios/lavadoManos/quitarJAbon.ogg')
              mixer.music.play()
              quitar_jabon=False
              t_fin=0

          if not quitar_jabon and t_fin>=2.5:
            if secar_manos:
              img_result=put_img.put_image_in_any_position(150, 380, img, "recursos/introducciones/lavadomanos/SecarseManos.png")
              cv2.imshow('image3', img_result)
              cv2.setMouseCallback('image3', introduccion_lavado_manos.click_event)
              mixer.music.load('recursos/audios/lavadoManos/secarManos.ogg')
              mixer.music.play()
              secar_manos=False
          

        except Exception as e:
          print(e)

        try:
            if return_action or (cv2.waitKey(5) & 0xFF == 27):
                mixer.music.stop()
                cv2.destroyWindow('image3')
                break
        except Exception as e:
            print(e)

        t_fin+=time.time()-t_ini