import sys
from rembg.bg import remove
import cv2
from PIL import Image

img = cv2.imread("data/A/1.jpg")

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
im_pil = Image.fromarray(img)
cutout = remove(im_pil)

save_path = "out.jpg"

cutout.save(save_path, "PNG")