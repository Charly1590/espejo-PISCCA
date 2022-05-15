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

class introduccion_introEducacionSexualNino():
  
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
    cv2.namedWindow('image4', cv2.WND_PROP_FULLSCREEN)
    cv2.moveWindow('image4', screen.x - 1, screen.y - 1)
    cv2.setWindowProperty('image4', cv2.WND_PROP_FULLSCREEN,  cv2.WINDOW_FULLSCREEN) 
    
    # Carga de imagenes
    img = cv2.imread('recursos/menu/Fondo.jpg')
    img = cv2.resize(img,(1360,768))
    img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    img = put_img.put_image_in_any_position(610, 20, img, "recursos/menu/volver.png")

    img2 = put_img.put_image_in_any_position(150, 100, img, "recursos/introducciones/educacionSex/hombre/DelanteHombre.png")

    img3 = put_img.put_image_in_any_position(150, 100, img, "recursos/introducciones/educacionSex/hombre/AtrasHombre.png")

    nadie_puede_tocar_tu_cuerpo=True
    si_alguien_toca_tu_cuerpo=True
    partes_privada_hombre=True
    boca=True
    pecho=True
    pene=True
    nalgas=True

    img_result=None

    mixer.init() 
    t_fin=0
    while 3==3:

        t_ini=time.time()
        try:

          if nadie_puede_tocar_tu_cuerpo:
              img_result=put_img.put_image_in_any_position(50, 280, img, "recursos/personajes/Doctora.png")
              cv2.imshow('image4', img_result)
              cv2.setMouseCallback('image4', introduccion_introEducacionSexualNino.click_event)
              mixer.music.load('recursos/audios/prevencionAbusoSexual/nadiePuedeTocarTuCuerpo.ogg')
              mixer.music.play()
              nadie_puede_tocar_tu_cuerpo=False

          if not nadie_puede_tocar_tu_cuerpo and t_fin>=3.5:
            if si_alguien_toca_tu_cuerpo: 
              cv2.setMouseCallback('image4', introduccion_introEducacionSexualNino.click_event)
              mixer.music.load('recursos/audios/prevencionAbusoSexual/siAlguienTocaTuCuerpoCuentaATuMama.ogg')
              mixer.music.play()
              si_alguien_toca_tu_cuerpo=False
              t_fin=0

          if not si_alguien_toca_tu_cuerpo and t_fin>=3.5:
            if partes_privada_hombre:
              cv2.setMouseCallback('image4', introduccion_introEducacionSexualNino.click_event)
              mixer.music.load('recursos/audios/prevencionAbusoSexual/partesHombre/PartesPrivHombre.ogg')
              mixer.music.play()
              partes_privada_hombre=False
              t_fin=0

          if not partes_privada_hombre and t_fin>=4.5:
            if boca:
              img_result=put_img.put_image_in_any_position(350, 260, img2, "recursos/audios/prevencionAbusoSexual/partesHombre/focus2.png")
              cv2.imshow('image4', img_result)
              cv2.setMouseCallback('image4', introduccion_introEducacionSexualNino.click_event)
              mixer.music.load('recursos/audios/prevencionAbusoSexual/partesHombre/Boca.ogg')
              mixer.music.play()
              boca=False
              t_fin=0

    
          if not boca and t_fin>=2.5:
            if pecho:
              img_result=put_img.put_image_in_any_position(285, 400, img2, "recursos/audios/prevencionAbusoSexual/partesHombre/focus.png")
              cv2.imshow('image4', img_result)
              cv2.setMouseCallback('image4', introduccion_introEducacionSexualNino.click_event)
              mixer.music.load('recursos/audios/prevencionAbusoSexual/partesHombre/Pecho.ogg')
              mixer.music.play()
              pecho=False
              t_fin=0

          if not pecho and t_fin>=3.5:
            if pene:
              img_result=put_img.put_image_in_any_position(290, 680, img2, "recursos/audios/prevencionAbusoSexual/partesHombre/focus.png")
              cv2.imshow('image4', img_result)
              cv2.setMouseCallback('image4', introduccion_introEducacionSexualNino.click_event)
              mixer.music.load('recursos/audios/prevencionAbusoSexual/partesHombre/Pene.ogg')
              mixer.music.play()
              pene=False
              t_fin=0

          if not pene and t_fin>=3.5:
            if nalgas:
              img_result=put_img.put_image_in_any_position(285, 670, img3, "recursos/audios/prevencionAbusoSexual/partesHombre/focus.png")
              cv2.imshow('image4', img_result)
              cv2.setMouseCallback('image4', introduccion_introEducacionSexualNino.click_event)
              mixer.music.load('recursos/audios/prevencionAbusoSexual/partesHombre/Nalgas.ogg')
              mixer.music.play()
              nalgas=False
        except Exception as e:
          print(e)

        try:
            if return_action or (cv2.waitKey(5) & 0xFF == 27):
                mixer.music.load('recursos/audios/Bubble.ogg')
                mixer.music.play()
                cv2.destroyWindow('image4')
                break
        except Exception as e:
            print(e)

        t_fin+=time.time()-t_ini