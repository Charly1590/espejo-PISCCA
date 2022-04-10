import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import screeninfo 
from moduloPosicionarImgs import Posicionamiento as put_img
from lavado_dientes import lavado_dientes as act_dientes


screen = screeninfo.get_monitors()[0] 
cv2.namedWindow('image', cv2.WND_PROP_FULLSCREEN)
cv2.moveWindow('image', screen.x - 1, screen.y - 1)
cv2.setWindowProperty('image', cv2.WND_PROP_FULLSCREEN,  cv2.WINDOW_FULLSCREEN) 

img = cv2.imread('Fondo.jpg')
img = cv2.resize(img,(1360,768))
img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)



def click_event_menu_inicial(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        if (x>=70 and x<=700) and (y>=660 and y<=900):
            menu_principal()
def click_event_menu_principal(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x," ",y)
        if (x>=170 and x<=620) and (y>=150 and y<=620):
            act_dientes.actividad()

def menu_principal():
    img_result=put_img.put_image_in_any_position(150, 130, img, "btnAutoc.png", 500,500)
    img_result=put_img.put_image_in_any_position(150, 700, img_result, "BtnEdSex.png", 500,500)
    cv2.imshow('image',img_result)
    cv2.setMouseCallback('image', click_event_menu_principal)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__=="__main__":
    
    img_result=put_img.put_image_in_any_position(55, 130, img, "Personajes1.png", 500,600)
    img_result=put_img.put_image_in_any_position(355, 400, img_result, "Personajes4.png", 350,350)
    img_result=put_img.put_image_in_any_position(55, 650, img_result, "ImgMenu1.png", 660,270)

    cv2.imshow('image',img_result)
    cv2.setMouseCallback('image', click_event_menu_inicial)

    cv2.waitKey(0)
    cv2.destroyAllWindows()