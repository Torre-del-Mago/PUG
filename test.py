from craft_hw_ocr.craft_hw_ocr import OCR

import cv2
import numpy as np

img = OCR.load_image('D:\\magisterka\\PUG\\splits\\training\\images\\2.png')

# do the below step if your image is tilted by some angle else ignore
# img = OCR.process_image(img)

ocr_models = OCR.load_models("microsoft/trocr-base-handwritten")

img, results = OCR.detection(img, ocr_models[2])

bboxes, text = OCR.recoginition(img, results, ocr_models[0], ocr_models[1])

pilImage = OCR.visualize(img, results)