import cv2
from defer import return_value
import numpy as np
from PIL import Image
from modulos.moduloPosicionarImgs import Posicionamiento as put_img
import screeninfo
import random

from playsound import playsound
import multiprocessing

class introduccion_lavado_dientes():
  
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
    cv2.namedWindow('image2', cv2.WND_PROP_FULLSCREEN)
    cv2.moveWindow('image2', screen.x - 1, screen.y - 1)
    cv2.setWindowProperty('image2', cv2.WND_PROP_FULLSCREEN,  cv2.WINDOW_FULLSCREEN) 
    
    # Carga de imagenes
    img = cv2.imread('recursos/menu/Fondo.jpg')
    img = cv2.resize(img,(1360,768))
    img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    img = put_img.put_image_in_any_position(610, 20, img, "recursos/menu/volver.png")

    introduccion=True
    primera_indicacion=True
    agarra_cepillo=True
    agarra_pasta=True
    poner_pasta=True
    cepillarse_dientes=True
    arriba_abajo=True

    
    introduccion_thread=None
    primera_indicacion_thread=None
    agarra_cepillo_thread=None
    agarra_pasta_thread=None
    poner_pasta_thread=None
    cepillarse_dientes_thread=None
    arriba_abajo_thread=None

    img_result=None
    while 3==3:

        try:
          
          if introduccion:
              cv2.imshow('image2', img)
              cv2.setMouseCallback('image2', introduccion_lavado_dientes.click_event)
              introduccion_thread = multiprocessing.Process(target=playsound, args=("recursos/audios/lavadoDientes/inicio.mp3",))
              introduccion_thread.start()
              introduccion=False

          print(introduccion_thread.is_alive())

          if not introduccion_thread.is_alive():
            if primera_indicacion: 
              cv2.imshow('image2', img)
              cv2.setMouseCallback('image2', introduccion_lavado_dientes.click_event)
              primera_indicacion_thread = multiprocessing.Process(target=playsound, args=("recursos/audios/lavadoDientes/indicacionInicial.mp3",))
              primera_indicacion_thread.start()
              primera_indicacion=False

            if not primera_indicacion_thread.is_alive():
                if agarra_cepillo:
                  img_result=put_img.put_image_in_any_position(150, 380, img, "recursos/introducciones/lavadodientes/CogerCepillo.png")
                  cv2.imshow('image2', img_result)
                  cv2.setMouseCallback('image2', introduccion_lavado_dientes.click_event)
                  agarra_cepillo_thread = multiprocessing.Process(target=playsound, args=("recursos/audios/lavadoDientes/cogerCepillo.mp3",))
                  agarra_cepillo_thread.start()
                  agarra_cepillo=False

                if not agarra_cepillo_thread.is_alive():
                    if agarra_pasta:
                      img_result=put_img.put_image_in_any_position(150, 380, img, "recursos/introducciones/lavadodientes/AgarrarPasta.png")
                      cv2.imshow('image2', img_result)
                      cv2.setMouseCallback('image2', introduccion_lavado_dientes.click_event)
                      agarra_pasta_thread = multiprocessing.Process(target=playsound, args=("recursos/audios/lavadoDientes/aggarraLaPasta.mp3",))
                      agarra_pasta_thread.start()
                      agarra_pasta=False

                    if not agarra_pasta_thread.is_alive():
                        if poner_pasta:
                          img_result=put_img.put_image_in_any_position(150, 380, img, "recursos/introducciones/lavadodientes/PonersePasta.png")
                          cv2.imshow('image2', img_result)
                          cv2.setMouseCallback('image2', introduccion_lavado_dientes.click_event)
                          poner_pasta_thread = multiprocessing.Process(target=playsound, args=("recursos/audios/lavadoDientes/ponerPastaCepillo.mp3",))
                          poner_pasta_thread.start()
                          poner_pasta=False

                        if not poner_pasta_thread.is_alive():
                            if cepillarse_dientes:
                              img_result=put_img.put_image_in_any_position(150, 380, img, "recursos/introducciones/lavadodientes/lavarseDientes.png")
                              cv2.imshow('image2', img_result)
                              cv2.setMouseCallback('image2', introduccion_lavado_dientes.click_event)
                              cepillarse_dientes_thread = multiprocessing.Process(target=playsound, args=("recursos/audios/lavadoDientes/cepillarseDientes.mp3",))
                              cepillarse_dientes_thread.start()
                              cepillarse_dientes=False
            
            if not cepillarse_dientes_thread.is_alive():
                if arriba_abajo:
                  cv2.setMouseCallback('image2', introduccion_lavado_dientes.click_event)
                  arriba_abajo_thread = multiprocessing.Process(target=playsound, args=("recursos/audios/lavadoDientes/arribaAbajoCepillado.mp3",))
                  arriba_abajo_thread.start()
                  arriba_abajo=False
          
        except Exception as e:
          print(e)

        try:
            if return_action or (cv2.waitKey(5) & 0xFF == 27):
                cv2.destroyWindow('image2')
                break
        except Exception as e:
            print(e)

          