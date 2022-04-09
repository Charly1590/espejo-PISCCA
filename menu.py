import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

img = cv2.imread('Fondo.jpg')
img = cv2.resize(img,(1280,720))
img=cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

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

def put_image_in_any_position(x, y, img, name, height, weight):

    img_per1 = np.array(Image.open(name))
    img_per1=cv2.resize(img_per1,(height,weight))
    alpha_mask_per1= img_per1[:, :, 3] / 255.0

    img = np.array(img)

    img_result = img[:, :, :3].copy()
    img_overlay = img_per1[:, :, :3]

    img_overlay = cv2.cvtColor(img_overlay, cv2.COLOR_BGR2RGB)

    overlay_image_alpha(img_result, img_overlay, x, y, alpha_mask_per1)
    
    return img_result

img_result=put_image_in_any_position(100, 100, img, "Personajes1.png", 400,500)
img_result=put_image_in_any_position(200, 100, img, "Personajes1.png", 400,500)

cv2.imshow('image',img_result)
cv2.waitKey(0)
cv2.destroyAllWindows()