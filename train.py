from craft_hw_ocr import OCR
from craft_text_detector import Craft
import torch
from Fine_tuned_TrOCR.model_training import *
from Fine_tuned_TrOCR.eval_metric import *
from torch.utils.data import DataLoader
from OCR_dataset import OCRDataset
from craft_dataset import CraftDataset
from torch.utils.tensorboard import SummaryWriter
from transformers import logging
from checkpoint import *
import warnings

logging.set_verbosity_error()
warnings.simplefilter(action='ignore', category=FutureWarning)

def log_hparams(writer, trOCR_train_config, avg_train_loss, avg_valid_cer, loop_idx):
    """Logging hyperparameters and results to TensorBoard."""
    
    hparams = {
        'batch_size': trOCR_train_config["batch_size"],
        'lr': trOCR_train_config["lr"],
        'num_epochs': trOCR_train_config["num_epochs"],
        'max_target_length': trOCR_train_config["max_target_length"]
    }
    
    # Add hyperparameters and results
    metrics = {
        'train_loss': avg_train_loss,
        'valid_cer': avg_valid_cer
    }
    
    # Log hparams and metrics
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

    # Optionally load from checkpoint
    checkpoint_dir = "checkpoints"
    checkpoint_files = os.listdir(checkpoint_dir)
    checkpoint_file_id = f"checkpoint_{config_trOCR['id']}.pth"

    if checkpoint_file_id in checkpoint_files:
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
        start_epoch, avg_train_loss, avg_valid_cer = load_checkpoint(model, optimizer, checkpoint_path)
        loop_idx = start_epoch  # Continue from the last epoch
    else:
        print(f"Checkpoint {checkpoint_file_id} not found. Starting training from scratch.")
        loop_idx = 0

    while True:
        
        ### TRAIN TROCR
        if loop_idx < trOCR_train_config["num_epochs"]:
            model.train()
            train_loss = 0.0
            for batch_idx, batch in enumerate(one_line_train_dataloader):
                # Get the inputs
                for k, v in batch.items():
                    batch[k] = v.to(device)

                # Forward + backward + optimize
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

            # Evaluate model 
            model.eval()
            valid_cer = 0.0
            with torch.no_grad():
                for batch in one_line_eval_dataloader:

                    # Run batch generation
                    outputs = model.generate(batch["pixel_values"].to(device))

                    # Compute metrics
                    cer = compute_cer(pred_ids=outputs, label_ids=batch["labels"], processor=processor)
                    valid_cer += cer

            avg_valid_cer = valid_cer / len(one_line_eval_dataloader)
            print("Validation CER:", avg_valid_cer)

            # Log validation CER to TensorBoard
            writer.add_scalar("TrOCR/Validation_CER", avg_valid_cer, loop_idx)

            # Save checkpoint after each epoch
            save_checkpoint(model, optimizer, loop_idx, avg_train_loss, avg_valid_cer, checkpoint_dir, config_trOCR["id"])
        
        
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
        print(f"Common results (CRAFT + TrOCR): {results}")
        
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
    Evaluation of two models simultaneously (CRAFT + TrOCR).

    Args:
        craft: CRAFT model for text detection.
        trocr_model: TrOCR model for text recognition.
        craft_val_dataset: Validation dataset for CRAFT.
        trocr_processor: TrOCR processor (for decoding predictions).
        device: Computing device (CPU/GPU).

    Returns:
        dict: Metric results (CER, WER, LER).
    """
    cer_sum, wer_sum, ler_sum = 0.0, 0.0, 0.0
    num_samples = 0
    num_regions = 0

    for i, (img, regions, transcriptions) in enumerate(craft_val_dataset):
        img = img.to(device) if isinstance(img, torch.Tensor) else img

        # 1. Text detection using CRAFT
        img, detection_results = OCR.detection(img, craft)

        # 2. Text recognition using TrOCR
        bboxes, recognized_texts = OCR.recoginition(img, detection_results, trocr_processor, trocr_model, device)
        
        # 3. Prepare ground truth - transcriptions is a list of texts assigned to each region (ground truth)
        if len(recognized_texts) != len(transcriptions):
            # print(f"Warning: Mismatch in detected regions and ground truth. Skipping sample {i}.")
            continue
        
        # Calculate metrics for each region
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

    # Calculate average results
    results = {
        "CER": cer_sum / num_regions if num_regions > 0 else 0.0,
        "WER": wer_sum / num_regions if num_regions > 0 else 0.0,
        "LER": ler_sum / num_regions if num_regions > 0 else 0.0
    }
    return results

def run_experiments():
    
    
    hparams_combinations = [
        {"id": 1, "batch_size": 8, "lr": 0.001, "num_epochs": 50, "max_target_length": 128},
        {"id": 2, "batch_size": 4, "lr": 0.01, "num_epochs": 25, "max_target_length": 128},
    ]
    
    for config in hparams_combinations:
        print("START TRAINING FOR CONFIGURATON:")
        print(f"batch_size: {config['batch_size']}, lr: {config['lr']}, num_epochs: {config['num_epochs']}, max_target_length: {config['max_target_length']}")
        train(config)
        print("FINISH TRAINING")
    
run_experiments()
        
            
        
    
            
    