import cv2
from defer import return_value
import numpy as np
from PIL import Image
from modulos.moduloPosicionarImgs import Posicionamiento as put_img
import screeninfo
import random

from threading import Thread
from playsound import playsound

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
    while 3==3:

        try:
          
          cv2.setMouseCallback('image3', introduccion_lavado_manos.click_event)

          if introduccion:
              cv2.imshow('image3', img)
              introduccion_thread = Thread(target=playsound, args=("recursos/audios/lavadoDientes/inicio.mp3",))
              introduccion_thread.start()
              introduccion=False

          if introduccion == False:
            if introduccion_thread.is_alive() == False:
              if primera_indicacion: 
                cv2.setMouseCallback('image3', introduccion_lavado_manos.click_event)
                primera_indicacion_thread = Thread(target=playsound, args=("recursos/audios/lavadoDientes/indicacionInicial.mp3",))
                primera_indicacion_thread.start()
                primera_indicacion=False

          if primera_indicacion == False:
            if primera_indicacion_thread.is_alive() == False:
                if mojarse_manos:
                  img_result=put_img.put_image_in_any_position(150, 380, img, "recursos/introducciones/lavadomanos/MojarseLasManos.png")
                  cv2.imshow('image3', img_result)
                  cv2.setMouseCallback('image3', introduccion_lavado_manos.click_event)
                  mojarse_manos_thread = Thread(target=playsound, args=("recursos/audios/lavadoManos/mojarseManos.mp3",))
                  mojarse_manos_thread.start()
                  mojarse_manos=False

          if mojarse_manos == False:
            if mojarse_manos_thread.is_alive() == False:
                if coger_jabon:
                  img_result=put_img.put_image_in_any_position(150, 380, img, "recursos/introducciones/lavadomanos/CogerJabon.png")
                  cv2.imshow('image3', img_result)
                  cv2.setMouseCallback('image3', introduccion_lavado_manos.click_event)
                  coger_jabon_thread = Thread(target=playsound, args=("recursos/audios/lavadoManos/cogerJabon.mp3",))
                  coger_jabon_thread.start()
                  coger_jabon=False

          if coger_jabon == False:
            if coger_jabon_thread.is_alive() == False:
                if fregarse_manos:
                  img_result=put_img.put_image_in_any_position(150, 380, img, "recursos/introducciones/lavadomanos/FregarseManos.png")
                  cv2.imshow('image3', img_result)
                  cv2.setMouseCallback('image3', introduccion_lavado_manos.click_event)
                  fregarse_manos_thread = Thread(target=playsound, args=("recursos/audios/lavadoManos/fregarseManos.mp3",))
                  fregarse_manos_thread.start()
                  fregarse_manos=False

          if fregarse_manos == False:
            if fregarse_manos_thread.is_alive() == False:
                if quitar_jabon:
                  img_result=put_img.put_image_in_any_position(150, 380, img, "recursos/introducciones/lavadomanos/QuitarJabon.png")
                  cv2.imshow('image3', img_result)
                  cv2.setMouseCallback('image3', introduccion_lavado_manos.click_event)
                  quitar_jabon_thread = Thread(target=playsound, args=("recursos/audios/lavadoManos/quitarJAbon.mp3",))
                  quitar_jabon_thread.start()
                  quitar_jabon=False
          
          if quitar_jabon == False:
            if quitar_jabon_thread.is_alive() == False:
                if secar_manos:
                  img_result=put_img.put_image_in_any_position(150, 380, img, "recursos/introducciones/lavadomanos/SecarseManos.png")
                  cv2.imshow('image3', img_result)
                  cv2.setMouseCallback('image3', introduccion_lavado_manos.click_event)
                  secar_manos_thread = Thread(target=playsound, args=("recursos/audios/lavadoManos/secarManos.mp3",))
                  secar_manos_thread.start()
                  secar_manos=False
          

        except Exception as e:
          print(e)

        try:
            if return_action or (cv2.waitKey(5) & 0xFF == 27):
                cv2.destroyWindow('image3')
                break
        except Exception as e:
            print(e)
        