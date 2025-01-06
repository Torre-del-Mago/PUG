from craft_hw_ocr import OCR

import cv2
import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# img = OCR.load_image('D:\\magisterka\\PUG\\splits\\training\\images\\2.png')
img = OCR.load_image('/home/user/PUG/splits/training/images/2.png')

# do the below step if your image is tilted by some angle else ignore
# img = OCR.process_image(img)

ocr_models = OCR.load_models("microsoft/trocr-base-handwritten", device)

img, results = OCR.detection(img, ocr_models[2])
print(results)

bboxes, text = OCR.recoginition(img, results, ocr_models[0], ocr_models[1], device)

print(text)
print(type(text))

pilImage = OCR.visualize(img, results)

OCR.save_image(pilImage, "test.png")