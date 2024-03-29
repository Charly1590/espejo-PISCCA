import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import screeninfo 
from modulos.moduloPosicionarImgs import Posicionamiento as put_img
from juegos.lavado_dientes import lavado_dientes as act_dientes
from juegos.lavadomanos import lavado_manos as act_manos
from introducciones.introlavadoDientes import introduccion_lavado_dientes
from introducciones.introlavadoManos import introduccion_lavado_manos
from introducciones.introEducacionSexualNina import introduccion_introEducacionSexualNina
from introducciones.introEducacionSexualNino import introduccion_introEducacionSexualNino
from juegos.prevencionnina import prevencion_nina as prev_nina
from pygame import mixer

class menus():

    screen = screeninfo.get_monitors()[0] 
    cv2.namedWindow('image', cv2.WND_PROP_FULLSCREEN)
    cv2.moveWindow('image', screen.x - 1, screen.y - 1)
    cv2.setWindowProperty('image', cv2.WND_PROP_FULLSCREEN,  cv2.WINDOW_FULLSCREEN) 


    img = cv2.imread('recursos/menu/Fondo.jpg')
    img = cv2.resize(img,(1360,768))
    img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    mixer.init()

    def click_event_menu_inicial(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            if (x>=70 and x<=700) and (y>=660 and y<=900):
                mixer.music.load('recursos/audios/Bubble.ogg')
                mixer.music.play()
                menus.menu_principal()

    def click_event_menu_principal(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(x," ",y, "principal")
            if (x>=170 and x<=620) and (y>=150 and y<=620):
                mixer.music.load('recursos/audios/Bubble.ogg')
                mixer.music.play()
                menus.menu_autocuidado()
            if (x>=170 and x<=620) and (y>=720 and y<=1190):
                mixer.music.load('recursos/audios/Bubble.ogg')
                mixer.music.play()
                menus.menu_seleccion_sexo()
            if (x>=600 and x<=750) and (y>=30 and y<=140):
                mixer.music.load('recursos/audios/Bubble.ogg')
                mixer.music.play()
                menus.menu_inicial()

    def click_event_menu_autocuidado(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(x," ",y, "AUTOCUIDADO")
            if (x>=170 and x<=620) and (y>=150 and y<=620):
                mixer.music.load('recursos/audios/Bubble.ogg')
                mixer.music.play()
                menus.menu_cepillado_dientes()
                # act_dientes.actividad()
                # intro_dientes.menu_seleccion_generoact_manos.actividad()
            if (x>=170 and x<=620) and (y>=720 and y<=1190):
                mixer.music.load('recursos/audios/Bubble.ogg')
                mixer.music.play()
                menus.menu_lavado_manos()
                # act_manos.actividad()
            if (x>=600 and x<=750) and (y>=30 and y<=140):
                mixer.music.load('recursos/audios/Bubble.ogg')
                mixer.music.play()
                menus.menu_principal()

    def click_event_menu_lavado_dientes(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(x," ",y, "AUTOCUIDADO")
            if (x>=170 and x<=620) and (y>=150 and y<=620):
                mixer.music.load('recursos/audios/Bubble.ogg')
                mixer.music.play()
                introduccion_lavado_dientes.secuencia()
                # menus.menu_cepillado_dientes()
                # act_dientes.actividad()
                # intro_dientes.menu_seleccion_sexo
            if (x>=170 and x<=620) and (y>=720 and y<=1190):
                mixer.music.load('recursos/audios/Bubble.ogg')
                mixer.music.play()
                act_dientes.actividad()
            if (x>=600 and x<=750) and (y>=30 and y<=140):
                mixer.music.load('recursos/audios/Bubble.ogg')
                mixer.music.play()
                menus.menu_autocuidado()
    
    def click_event_menu_lavado_manos(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(x," ",y, "AUTOCUIDADO")
            if (x>=170 and x<=620) and (y>=150 and y<=620):
                mixer.music.load('recursos/audios/Bubble.ogg')
                mixer.music.play()
                introduccion_lavado_manos.secuencia()
            if (x>=170 and x<=620) and (y>=720 and y<=1190):
                mixer.music.load('recursos/audios/Bubble.ogg')
                mixer.music.play()
                act_manos.actividad()
            if (x>=600 and x<=750) and (y>=30 and y<=140):
                mixer.music.load('recursos/audios/Bubble.ogg')
                mixer.music.play()
                menus.menu_autocuidado()
    
    def click_event_menu_seleccion_sexo(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(x," ",y, "SELECCION_GENERO")
            if (x>=170 and x<=620) and (y>=150 and y<=620):
                mixer.music.load('recursos/audios/Bubble.ogg')
                mixer.music.play()
                menus.menu_educacionSexual_Nina()
            if (x>=170 and x<=620) and (y>=720 and y<=1190):
                mixer.music.load('recursos/audios/Bubble.ogg')
                mixer.music.play()
                menus.menu_educacionSexual_Nino()
            if (x>=600 and x<=750) and (y>=30 and y<=140):
                mixer.music.load('recursos/audios/Bubble.ogg')
                mixer.music.play()
                menus.menu_principal()

    def click_event_menu_educacionSexual_Nina(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(x," ",y, "SELECCION_GENERO")
            if (x>=170 and x<=620) and (y>=150 and y<=620):
                mixer.music.load('recursos/audios/Bubble.ogg')
                mixer.music.play()
                introduccion_introEducacionSexualNina.secuencia()
            if (x>=170 and x<=620) and (y>=720 and y<=1190):
                mixer.music.load('recursos/audios/Bubble.ogg')
                mixer.music.play()
                prev_nina.actividad()
            if (x>=600 and x<=750) and (y>=30 and y<=140):
                mixer.music.load('recursos/audios/Bubble.ogg')
                mixer.music.play()
                menus.menu_seleccion_sexo()
    
    def click_event_menu_educacionSexual_Nino(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(x," ",y, "SELECCION_GENERO")
            if (x>=170 and x<=620) and (y>=150 and y<=620):
                mixer.music.load('recursos/audios/Bubble.ogg')
                mixer.music.play()
                introduccion_introEducacionSexualNino.secuencia()
            if (x>=170 and x<=620) and (y>=720 and y<=1190):
                mixer.music.load('recursos/audios/Bubble.ogg')
                mixer.music.play()
                prev_nina.actividad()
            if (x>=600 and x<=750) and (y>=30 and y<=140):
                mixer.music.load('recursos/audios/Bubble.ogg')
                mixer.music.play()
                menus.menu_seleccion_sexo()

    def menu_inicial():
        img_result=put_img.put_image_in_any_position(55, 130, menus.img, "recursos/personajes/Personajes1.png")
        img_result=put_img.put_image_in_any_position(355, 400, img_result, "recursos/personajes/Personajes4.png")
        img_result=put_img.put_image_in_any_position(55, 650, img_result, "recursos/menu/ImgMenu1.png")
        cv2.imshow('image',img_result)
        cv2.setMouseCallback('image', menus.click_event_menu_inicial)

    def menu_principal():
        img_result=put_img.put_image_in_any_position(150, 130, menus.img, "recursos/menu/btnAutoc.png")
        img_result=put_img.put_image_in_any_position(150, 700, img_result, "recursos/menu/BtnEdSex.png")
        img_result=put_img.put_image_in_any_position(610, 20, img_result, "recursos/menu/volver.png")
        cv2.imshow('image',img_result)
        cv2.setMouseCallback('image', menus.click_event_menu_principal)
        cv2.waitKey(0)
        cv2.destroyAllWiclick_event_menu_educacionSexual_Ninandows()

    def menu_autocuidado():
        img_result=put_img.put_image_in_any_position(150, 130, menus.img, "recursos/menu/btnimg3.png")
        img_result=put_img.put_image_in_any_position(150, 700, img_result, "recursos/menu/btnimg7.png")
        img_result=put_img.put_image_in_any_position(610, 20, img_result, "recursos/menu/volver.png")
        cv2.imshow('image',img_result)
        cv2.setMouseCallback('image', menus.click_event_menu_autocuidado)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def menu_cepillado_dientes():
        img_result=put_img.put_image_in_any_position(55, 230, menus.img, "recursos/menu/btnintro.png")
        img_result=put_img.put_image_in_any_position(55, 700, img_result, "recursos/menu/ImgMenu1.png")
        img_result=put_img.put_image_in_any_position(610, 20, img_result, "recursos/menu/volver.png")
        cv2.imshow('image',img_result)
        cv2.setMouseCallback('image', menus.click_event_menu_lavado_dientes)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def menu_lavado_manos():
        img_result=put_img.put_image_in_any_position(55, 230, menus.img, "recursos/menu/btnintro.png")
        img_result=put_img.put_image_in_any_position(55, 700, img_result, "recursos/menu/ImgMenu1.png")
        img_result=put_img.put_image_in_any_position(610, 20, img_result, "recursos/menu/volver.png")
        cv2.imshow('image',img_result)
        cv2.setMouseCallback('image', menus.click_event_menu_lavado_manos)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def menu_seleccion_sexo():
        img_result=put_img.put_image_in_any_position(150, 130, menus.img, "recursos/menu/niñabtn.png")
        img_result=put_img.put_image_in_any_position(150, 700, img_result, "recursos/menu/niñobtn.png")
        img_result=put_img.put_image_in_any_position(610, 20, img_result, "recursos/menu/volver.png")
        cv2.imshow('image',img_result)
        cv2.setMouseCallback('image', menus.click_event_menu_seleccion_sexo)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def menu_educacionSexual_Nina():
        img_result=put_img.put_image_in_any_position(55, 230, menus.img, "recursos/menu/btnintro.png")
        img_result=put_img.put_image_in_any_position(55, 700, img_result, "recursos/menu/ImgMenu1.png")
        img_result=put_img.put_image_in_any_position(610, 20, img_result, "recursos/menu/volver.png")
        cv2.imshow('image',img_result)
        cv2.setMouseCallback('image', menus.click_event_menu_educacionSexual_Nina)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def menu_educacionSexual_Nino():
        img_result=put_img.put_image_in_any_position(55, 230, menus.img, "recursos/menu/btnintro.png")
        img_result=put_img.put_image_in_any_position(55, 700, img_result, "recursos/menu/ImgMenu1.png")
        img_result=put_img.put_image_in_any_position(610, 20, img_result, "recursos/menu/volver.png")
        cv2.imshow('image',img_result)
        cv2.setMouseCallback('image', menus.click_event_menu_educacionSexual_Nino)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    


if __name__=="__main__":
    menus.menu_inicial()
    cv2.waitKey(0)
    cv2.destroyAllWindows() 