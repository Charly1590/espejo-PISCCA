import cv2
from defer import return_value
from modulos.moduloPosicionarImgs import Posicionamiento as put_img
import screeninfo
import time

from playsound import playsound
from threading import Thread
from pygame import mixer

class introduccion_lavado_dientes():

  return_action=False

  def click_event(event, x, y, flags, params):
    global return_action
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x," ",y, "sdahsjk")
        if (x>=600 and x<=750) and (y>=30 and y<=140):
          return_action=True
        else:
          return_action=False

  def play_sound(path):
      mixer.music.load(path)
      mixer.music.play()

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

    showtuto=False
    cicloanim=0.4
    posianim=0
    imgentuto=""

    mixer.init() 
    t_fin=0
    img_result=None
    while 3==3:

        if cicloanim <= 1.0 :
          cicloanim+=0.06
          posianim+=16

        
        t_ini=time.time()
        try:

          if introduccion:
              img_result=put_img.put_image_in_any_position(50, 280, img, "recursos/personajes/Doctora.png")
              cv2.imshow('image2', img_result)
              mixer.music.load('recursos/audios/lavadoDientes/inicio.ogg')
              mixer.music.play()
              introduccion=False

          if not introduccion and t_fin>=6.5:
            if primera_indicacion: 
              
              mixer.music.load('recursos/audios/lavadoDientes/indicacionInicial.ogg')
              mixer.music.play()
              primera_indicacion=False
              t_fin=0

          if not primera_indicacion and t_fin>=6.5:
            if agarra_cepillo:
              imgentuto ="recursos/introducciones/lavadodientes/CogerCepillo.png"
              showtuto=True
              cicloanim=0.4
              posianim=0
              mixer.music.load('recursos/audios/lavadoDientes/cogerCepillo.ogg')
              mixer.music.play()
              agarra_cepillo=False
              t_fin=0

          if not agarra_cepillo and t_fin>=4:
            if agarra_pasta:
              imgentuto ="recursos/introducciones/lavadodientes/AgarrarPasta.png"
              cicloanim=0.4
              posianim=0
              mixer.music.load('recursos/audios/lavadoDientes/aggarraLaPasta.ogg')
              mixer.music.play()
              agarra_pasta=False
              t_fin=0

          if not agarra_pasta and t_fin>=3:
            if poner_pasta:
              imgentuto ="recursos/introducciones/lavadodientes/PonersePasta.png"
              cicloanim=0.4
              posianim=0
              mixer.music.load('recursos/audios/lavadoDientes/ponerPastaCepillo.ogg')
              mixer.music.play()
              poner_pasta=False
              t_fin=0

          if not poner_pasta and t_fin>=4:
            if cepillarse_dientes:
              imgentuto = "recursos/introducciones/lavadodientes/lavarseDientes.png"
              cicloanim=0.4
              posianim=0
              
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
          
          if showtuto:
            img_result=put_img.put_image_in_any_position_with_resize(300-posianim, 530-posianim, img, imgentuto ,int(500*cicloanim),int(500*cicloanim))
            cv2.imshow('image2', img_result)
            cv2.setMouseCallback('image2', introduccion_lavado_dientes.click_event)

        except Exception as e:
          print(e)
        try:
            if return_action or (cv2.waitKey(5) & 0xFF == 27):
              mixer.music.stop()
              cv2.destroyWindow('image2')
              break
        except Exception as e:
            print(e)
        
        t_fin+=time.time()-t_ini
        