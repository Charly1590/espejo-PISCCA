# Python program killing
# threads using stop
# flag

import threading
import time
import audiosegment 
from pygame import mixer

from threading import Thread
import time
mixer.init()

class a():
    stop_threads= True
    def play_sound(path):
        global stop_threads
        print(stop_threads)
        if stop_threads:
            mixer.music.load(path)
            mixer.music.play()
        else:
            mixer.music.stop()

    def main():
        ab=True
        global stop_threads
        stop_threads=True
        while True:
            if ab:
                t1 = Thread(target = a.play_sound, args=('/home/piscca/Desktop/espejo-PISCCA/recursos/audios/lavadoDientes/indicacionInicial.ogg',))
                t1.start()
                ab=False
            print("jkdfhsjkdfh")
            time.sleep(3)
            t1.stop_threads = False

            


        print('thread killed')

a.main()