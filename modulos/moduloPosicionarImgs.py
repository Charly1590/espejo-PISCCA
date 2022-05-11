import cv2
import numpy as np
from PIL import Image

class Posicionamiento():
    
    def overlay_image_alpha(img, img_overlay, x, y, alpha_mask):
        """Overlay `img_overlay` onto `img` at (x, y) and blend using `alpha_mask`.

        `alpha_mask` must have same HxW as `img_overlay` and values in range [0, 1].
        """
        # Image ranges
        y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
        x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])

        # Overlay ranges
        y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
        x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)

        # Exit if nothing to do
        if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
            return

        # Blend overlay within the determined ranges
        img_crop = img[y1:y2, x1:x2]
        img_overlay_crop = img_overlay[y1o:y2o, x1o:x2o]
        alpha = alpha_mask[y1o:y2o, x1o:x2o, np.newaxis]
        alpha_inv = 1.0 - alpha

        img_crop[:] = alpha * img_overlay_crop + alpha_inv * img_crop

    """
        Esta funcion se utilizara para ubicar imagenes sobre otras imagenes
        teniendo como parametros:

        - la posicion x
        - la posicion y
        - la imagen a la que queremos sobreponer una imagen
        - el nombre de la imagen que vamos a montar
        - la altura de la imagen que vamos a montar
        - el ancho de la imagen que vamos a montar
        
    """
    def put_image_in_any_position(x, y, img, name):

        new_image = np.array(Image.open(name))
        alpha_mask_new_image= new_image[:, :, 3] / 255.0

        img = np.array(img)

        img_result = img[:, :, :3].copy()
        img_overlay = new_image[:, :, :3]

        img_overlay = cv2.cvtColor(img_overlay, cv2.COLOR_BGR2RGB)

        Posicionamiento.overlay_image_alpha(img_result, img_overlay, x, y, alpha_mask_new_image)
        
        return img_result
    
    """
        Esta funcion se utilizara para ubicar imagenes sobre el video
        teniendo como parametros:

        - la posicion x
        - la posicion y
        - el frame al que queremos montar la imagen
        - la imagen que vamos a montar

    """
    def put_elements_in_viedo(x, y, img, element):

        alpha_mask_new_image = element[:, :, 3] / 255.0

        img = np.array(img)

        img_result = img[:, :, :3].copy()
        img_overlay = element[:, :, :3]

        img_overlay = cv2.cvtColor(img_overlay, cv2.COLOR_BGR2RGB)

        Posicionamiento.overlay_image_alpha(img_result, img_overlay, x, y, alpha_mask_new_image)
        
        return img_result