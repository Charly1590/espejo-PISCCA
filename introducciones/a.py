from threading import Thread
from playsound import playsound
import time
import sys
import multiprocessing
brushingSound=None
a=True
while 3==3:
    if a: 
        brushingSound = multiprocessing.Process(target=playsound, args=("/home/piscca/Desktop/espejo-PISCCA/recursos/audios/lavadoDientes/inicio.mp3",))
        brushingSound.start()
        a=False
    print(brushingSound.is_alive())
    if not brushingSound.is_alive():
        print(brushingSound.is_alive())
        break
