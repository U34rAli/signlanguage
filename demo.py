import argparse
import glob
import os

import cv2

from yolo import YOLO
from pathlib import Path


resultfolder = "result"

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--images', default="images", help='Path to images or image file')
ap.add_argument('-n', '--network', default="normal", help='Network Type: normal / tiny / prn')
ap.add_argument('-d', '--device', default=0, help='Device to use')
ap.add_argument('-s', '--size', default=640, help='Size for yolo')
ap.add_argument('-c', '--confidence', default=0.25, help='Confidence for yolo')
args = ap.parse_args()

if args.network == "normal":
    print("loading yolo...")
    yolo = YOLO("models/cross-hands.cfg", "models/cross-hands.weights", ["hand"])
elif args.network == "prn":
    print("loading yolo-tiny-prn...")
    yolo = YOLO("models/cross-hands-tiny-prn.cfg", "models/cross-hands-tiny-prn.weights", ["hand"])
else:
    print("loading yolo-tiny...")
    yolo = YOLO("models/cross-hands-tiny.cfg", "models/cross-hands-tiny.weights", ["hand"])

yolo.size = int(args.size)
yolo.confidence = float(args.confidence)

print("extracting tags for each image...")
if args.images.endswith(".txt"):
    with open(args.images, "r") as myfile:
        lines = myfile.readlines()
        files = map(lambda x: os.path.join(os.path.dirname(args.images), x.strip()), lines)
else:
    files = sorted(glob.glob("%s/*.jpg" % args.images))

conf_sum = 0
detection_count = 0

files = glob.glob("data/*/**")
for file in files:

    path = Path(file)
    path =str(path)
    path = os.path.join(resultfolder,  path)

    if not os.path.exists(path):
        
        mat = cv2.imread(file)
        mat = cv2.resize(mat, (256,256))
        width, height, inference_time, results = yolo.inference(mat)

        print("%s in %s seconds: %s classes found!" %
            (os.path.basename(file), round(inference_time, 2), len(results)))

        output = []

        for detection in results:
            id, name, confidence, x, y, w, h = detection
            cx = x + (w / 2)
            cy = y + (h / 2)

            conf_sum += confidence
            detection_count += 1

            # draw a bounding box rectangle and label on the image
            color = (255, 0, 255)
            segment = mat[y:y+h, x:x+w]
            
            if not os.path.exists(os.path.dirname(path)):
                try:
                    os.makedirs(os.path.dirname(path))
                except: # Guard against race condition
                    print("folder error")

            try:
                segment = cv2.resize(segment, (256,256))
                cv2.imwrite(path, segment)
            except:
                pass

