from craft_hw_ocr import OCR
from craft_text_detector import Craft
import torch
from Fine_tuned_TrOCR.model_training import *
from Fine_tuned_TrOCR.eval_metric import *
from torch.utils.data import DataLoader
from OCR_dataset import OCRDataset
from craft_dataset import CraftDataset
from torch.utils.tensorboard import SummaryWriter

def log_hparams(writer, trOCR_train_config, avg_train_loss, avg_valid_cer, loop_idx):
    """Logowanie hiperparametrów i wyników do TensorBoard."""
    
    hparams = {
        'batch_size': trOCR_train_config["batch_size"],
        'lr': trOCR_train_config["lr"],
        'num_epochs': trOCR_train_config["num_epochs"],
        'max_target_length': trOCR_train_config["max_target_length"]
    }
    
    # Dodajemy hiperparametry oraz wyniki
    metrics = {
        'train_loss': avg_train_loss,
        'valid_cer': avg_valid_cer
    }
    
    # Logowanie hparams oraz metryk
    writer.add_hparams(hparams, metrics, global_step=loop_idx)



### TRAIN MODELS
def train(config_trOCR):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
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
        "batch_size": config_trOCR["batch_size"],
        "lr": config_trOCR["lr"],
        "num_epochs": config_trOCR["num_epochs"],
        "max_target_length": config_trOCR["max_target_length"]
    }
    optimizer = torch.optim.AdamW(model.parameters(), lr=trOCR_train_config["lr"])

    # craft_train_config = {
    #     "batch_size": 8,
    #     "lr": 0.001,
    #     "num_epochs": num_epochs,
    # }

    one_line_train_dataset = OCRDataset(
        base_path="/home/user/PUG/splits/test",
        processor=processor,
        max_target_length=trOCR_train_config["max_target_length"]
    )
    one_line_eval_dataset = OCRDataset(
        base_path="/home/user/PUG/splits/test",
        processor=processor,
        max_target_length=trOCR_train_config["max_target_length"]
    )
    one_line_train_dataloader = DataLoader(one_line_train_dataset, batch_size=trOCR_train_config["batch_size"], shuffle=True)
    one_line_eval_dataloader = DataLoader(one_line_eval_dataset, batch_size=trOCR_train_config["batch_size"])

    page_train_dataset = CraftDataset(
        base_path="/home/user/PUG/splits",
        split="test"
    )
    page_val_dataset = CraftDataset(
        base_path="/home/user/PUG/splits",
        split="test"
    )

    # TensorBoard writer
    writer = SummaryWriter(log_dir="runs/training_logs")

    loop_idx = 0
    while True:
        
        ### TRAIN TROCR
        if loop_idx < trOCR_train_config["num_epochs"]:
            model.train()
            train_loss = 0.0
            for batch_idx, batch in enumerate(one_line_train_dataloader):
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

                # Log loss to TensorBoard
                writer.add_scalar("TrOCR/Train_Loss", loss.item(), loop_idx * len(one_line_train_dataloader) + batch_idx)

            avg_train_loss = train_loss / len(one_line_train_dataloader)
            print(f"Loss after epoch {loop_idx+1}:", avg_train_loss)

            # Log average loss for the epoch
            writer.add_scalar("TrOCR/Average_Train_Loss", avg_train_loss, loop_idx)

            # evaluate
            model.eval()
            valid_cer = 0.0
            with torch.no_grad():
                for batch in one_line_eval_dataloader:
                    # run batch generation
                    outputs = model.generate(batch["pixel_values"].to(device))
                    # compute metrics
                    cer = compute_cer(pred_ids=outputs, label_ids=batch["labels"], processor=processor)
                    valid_cer += cer

            avg_valid_cer = valid_cer / len(one_line_eval_dataloader)
            print("Validation CER:", avg_valid_cer)

            # Log validation CER to TensorBoard
            writer.add_scalar("TrOCR/Validation_CER", avg_valid_cer, loop_idx)
        
        
        # In the future we can add training of CRAFT model
        # 
        # if loop_idx < craft_train_config["num_epochs"]:
        #     craft.train()
        #     train_loss = 0.0
        #     for i, (img, regions, transcription) in enumerate(page_train_dataset):
        #         img = img.to(device)

        #         # Perform text detection with CRAFT
        #         img, detection_results = OCR.detection(img, craft)
                
        #         # Now we use TrOCR for recognition
        #         bboxes, recognized_text = OCR.recoginition(img, detection_results, processor, model)

        #         # Compute loss and update CRAFT optimizer
        #         # Placeholder: You should add the actual loss calculation for CRAFT here
        #         # craft_loss = some_loss_function(detection_results, regions)
        #         # craft_loss.backward()
        #         # craft_optimizer.step()

        #         # train_loss += craft_loss.item()
                
        #     print(f"Loss after epoch {loop_idx+1} for CRAFT:", train_loss / len(page_train_dataset))

        #     # Optionally, evaluate CRAFT on a validation set (similar to how we evaluate TrOCR)
        #     craft.eval()
        #     with torch.no_grad():
        #         # Do detection on validation set, calculate some metrics if required
        #         pass
        # 
        
        # EVALUATION MODELS
        model.eval()
        results = evaluate_jointly(craft, model, page_val_dataset, processor, device)
        print(f"Wspólne wyniki (CRAFT + TrOCR): {results}")
        
        # Log joint evaluation metrics to TensorBoard
        writer.add_scalars("Evaluation", {
            "CER": results["CER"],
            "WER": results["WER"],
            "LER": results["LER"]
        }, loop_idx)

        log_hparams(writer, trOCR_train_config, avg_train_loss, avg_valid_cer, loop_idx)

        loop_idx += 1
        if loop_idx >= trOCR_train_config["num_epochs"]:
            break

    # Close the writer
    writer.close()
    
### EVALUATE MODELS
def evaluate_jointly(craft, trocr_model, craft_val_dataset, trocr_processor, device):
    """
    Ewaluacja dwóch modeli jednocześnie (CRAFT + TrOCR).

    Args:
        craft: Model CRAFT do detekcji tekstu.
        trocr_model: Model TrOCR do rozpoznawania tekstu.
        craft_val_dataset: Zbiór danych walidacyjnych dla CRAFT.
        trocr_processor: Processor TrOCR (do dekodowania predykcji).
        device: Urządzenie obliczeniowe (CPU/GPU).

    Returns:
        dict: Wyniki metryk (CER, WER, LER).
    """
    cer_sum, wer_sum, ler_sum = 0.0, 0.0, 0.0
    num_samples = 0
    num_regions = 0

    for i, (img, regions, transcriptions) in enumerate(craft_val_dataset):
        img = img.to(device) if isinstance(img, torch.Tensor) else img

        # 1. Detekcja tekstu za pomocą CRAFT
        img, detection_results = OCR.detection(img, craft)

        # 2. Rozpoznawanie tekstu za pomocą TrOCR
        bboxes, recognized_texts = OCR.recoginition(img, detection_results, trocr_processor, trocr_model, device)
        
        # 3. Przygotowanie ground truth
        # `transcriptions` to lista tekstów przypisanych do każdego fragmentu (ground truth)
        if len(recognized_texts) != len(transcriptions):
            print(f"Warning: Mismatch in detected regions and ground truth. Skipping sample {i}.")
            continue
        
        # Obliczanie metryk dla każdego regionu
        for pred_text, true_text in zip(recognized_texts, transcriptions):
            if pred_text == "":
                pred_text = 'a'
            if true_text == "":
                true_text = 'a'

            cer = cer_metric.compute(predictions=[pred_text], references=[true_text])
            wer = wer_metric.compute(predictions=[pred_text], references=[true_text])
            ler = 1.0 if pred_text != true_text else 0.0

            cer_sum += cer
            wer_sum += wer
            ler_sum += ler
            num_regions += 1

    # Oblicz średnie wyniki
    results = {
        "CER": cer_sum / num_regions if num_regions > 0 else 0.0,
        "WER": wer_sum / num_regions if num_regions > 0 else 0.0,
        "LER": ler_sum / num_regions if num_regions > 0 else 0.0
    }
    return results

def run_experiments():
    
    
    hparams_combinations = [
        {"batch_size": 8, "lr": 0.001, "num_epochs": 2, "max_target_length": 128},
        {"batch_size": 4, "lr": 0.01, "num_epochs": 5, "max_target_length": 128},
    ]
    
    for config in hparams_combinations:
        train(config)
    
run_experiments()
        
            
        
    
            
    