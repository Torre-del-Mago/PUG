from craft_hw_ocr import OCR
from utilities import group_results_into_lines, draw_bboxes_and_lines
from craft_text_detector import Craft
from Fine_tuned_TrOCR.model_training import *
import torch
from PIL import Image, ImageDraw, ImageFont

if __name__ == "__main__":
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)
    
    craft = Craft(output_dir=None, 
                    crop_type="poly",
                    export_extra=False,
                    link_threshold=0.1,
                    text_threshold=0.3,
                    cuda=torch.cuda.is_available(),
                    gpu_id=1)
    
    trOCR_path = "microsoft/trocr-base-handwritten"
    trOCR_path_checkpoint = "/mnt/raid/checkpoints/checkpoint_6.pth"
    
    processor = load_processor(trOCR_path)
    model = load_model(trOCR_path, device)
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id

    checkpoint = torch.load(trOCR_path_checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])

    # img = OCR.load_image('/mnt/raid/splits/test_original/images/0020.jpg')
    img = OCR.load_image('/mnt/raid/splits/test/images/118.png')

    img, results = OCR.detection(img, craft)
    grouped_lines = group_results_into_lines(results, y_tolerance=6, x_tolerance=500)
    
    draw_bboxes_and_lines(img, results, grouped_lines, output_dir="test_img_boxes", image_name=f"test_original_20.png")

    bboxes, text = OCR.recoginition(img, grouped_lines, processor, model, device)

    print(text)
    
    # Rysowanie rozpoznanego tekstu na obrazie
    pilImage = OCR.visualize(img, results)
    draw = ImageDraw.Draw(pilImage)

    for bbox, txt in zip(bboxes, text):
        x_min = int(min(bbox[:, 0]))  # Najmniejsza wartość x
        y_min = int(min(bbox[:, 1]))  # Najmniejsza wartość y
        x_max = int(max(bbox[:, 0]))  # Największa wartość x
        y_max = int(max(bbox[:, 1]))  # Największa wartość y

        draw.text((x_min, y_min - 10), str(txt), fill=(255, 0, 0)) 
    
    OCR.save_image(pilImage, "test_with_text_2.png")
