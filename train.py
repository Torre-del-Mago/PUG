from craft_hw_ocr.craft_hw_ocr.OCR import * 
from craft_text_detector import Craft
import torch
from Fine_tuned_TrOCR.model_training import *
from tqdm import tqdm
from torch.utils.data import DataLoader
from OCR_dataset import OCRDataset
from craft_dataset import CraftDataset

def train():
    device = torch.device("cpu") #torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
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
    
    trOCR_train_config = {
        "batch_size": 16,
        "lr": 0.001,
        "num_epochs": 10,
        "eval_interval": 1,
        "max_target_length": 128
    }
    optimizer = torch.optim.AdamW(model.parameters(), lr=trOCR_train_config["lr"])

    craft_train_config = {
        "batch_size": 16,
        "lr": 0.001,
        "num_epochs": 10,
        "eval_interval": 1,
    }
    
    one_line_train_dataset = OCRDataset(
        base_path="D:\\magisterka\\PUG\\splits\\training",
        processor=processor,
        max_target_length=trOCR_train_config["max_target_length"]
    )
    one_line_eval_dataset = OCRDataset(
        base_path="D:\\magisterka\\PUG\\splits\\validation",
        processor=processor,
        max_target_length=trOCR_train_config["max_target_length"]
    )
    one_line_train_dataloader = DataLoader(one_line_train_dataset, batch_size=trOCR_train_config["batch_size"], shuffle=True)
    one_line_eval_dataloader = DataLoader(one_line_eval_dataset, batch_size=trOCR_train_config["batch_size"])

    page_train_dataset = CraftDataset(
        base_path="D:\\magisterka\\PUG\\splits", 
        split="training"
    )
    page_val_dataset = CraftDataset(
        base_path="D:\\magisterka\\PUG\\splits", 
        split="validation"
    )


    loop_idx = 0
    while True:
        
        ### TRAIN TROCR
        if loop_idx < trOCR_train_config["num_epochs"]:
            model.train()
            train_loss = 0.0
            for batch in tqdm(one_line_train_dataloader):
                # get the inputs
                for k, v in batch.items():
                    batch[k] = v.to(device)

                # forward + backward + optimize
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                train_loss += loss.item()

            print(f"Loss after epoch {loop_idx+1}:", train_loss / len(one_line_train_dataloader))

            # evaluate
            model.eval()
            valid_cer = 0.0
            with torch.no_grad():
                for batch in tqdm(one_line_eval_dataloader):
                    # run batch generation
                    outputs = model.generate(batch["pixel_values"].to(device))
                    # compute metrics
                    cer = compute_cer(pred_ids=outputs, label_ids=batch["labels"])
                    valid_cer += cer

            print("Validation CER:", valid_cer / len(one_line_eval_dataloader))
        
        if loop_idx < craft_train_config["num_epochs"]:
            craft.train()
            train_loss = 0.0
            for i, (img, regions, transcription) in enumerate(tqdm(page_train_dataset)):
                img = img.to(device)

                # Perform text detection with CRAFT
                img, detection_results = detection(img, craft)
                
                # Now we use TrOCR for recognition
                bboxes, recognized_text = recoginition(img, detection_results, processor, model)

                # Compute loss and update CRAFT optimizer
                # Placeholder: You should add the actual loss calculation for CRAFT here
                # craft_loss = some_loss_function(detection_results, regions)
                # craft_loss.backward()
                # craft_optimizer.step()

                # train_loss += craft_loss.item()
                
            print(f"Loss after epoch {loop_idx+1} for CRAFT:", train_loss / len(page_train_dataset))

            # Optionally, evaluate CRAFT on a validation set (similar to how we evaluate TrOCR)
            craft.eval()
            with torch.no_grad():
                # Do detection on validation set, calculate some metrics if required
                pass
        break  
    
        ### EVALUATE MODELS
train()
        
            
        
    
            
    