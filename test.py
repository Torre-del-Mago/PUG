from craft_hw_ocr import OCR
from utilities import group_results_into_lines, draw_bboxes_and_lines
from craft_text_detector import Craft
from Fine_tuned_TrOCR.model_training import *
import torch


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)
    
    craft = Craft(output_dir=None, 
                    crop_type="poly",
                    export_extra=False,
                    link_threshold=0.1,
                    text_threshold=0.3,
                    cuda=torch.cuda.is_available())
    
    trOCR_path = "microsoft/trocr-base-handwritten"
     
    processor = load_processor(trOCR_path)
    
    model = load_model(trOCR_path, device)
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    # img = OCR.load_image('D:\\magisterka\\PUG\\splits\\training\\images\\2.png')
    
    # do the below step if your image is tilted by some angle else ignore
    # img = OCR.process_image(img)
    for i in range(10,15):
        img = OCR.load_image(f'splits/training/images/{i}.png')

        img, results = OCR.detection(img, craft)
        print(results["boxes"])

        grouped_lines = group_results_into_lines(results, y_tolerance=6, x_tolerance=500)
        print(grouped_lines["boxes"])

        draw_bboxes_and_lines(img, results,  grouped_lines, output_dir="test_img_boxes", image_name=f"test_image_{i}.png")

        bboxes, text = OCR.recoginition(img, grouped_lines, processor, model, device)

        # print(text)
        # print(type(text))  

        pilImage = OCR.visualize(img, results)

        OCR.save_image(pilImage, "test.png")