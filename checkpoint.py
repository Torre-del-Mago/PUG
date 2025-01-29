import os
import torch

def save_checkpoint(model, optimizer, epoch, avg_train_loss, avg_valid_cer, config_trOCR, checkpoint_dir="checkpoints"):
    """Saves the model and optimizer state to a checkpoint file."""
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': avg_train_loss,
        'valid_cer': avg_valid_cer,
        'lr': config_trOCR["lr"],
        'batch_size': config_trOCR["batch_size"],
        'num_epochs': config_trOCR["num_epochs"],
        'max_target_length': config_trOCR["max_target_length"]
    }
    
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{config_trOCR['id']}.pth")
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")

def load_checkpoint(model, optimizer, checkpoint_path, config_trOCR):
    """Loads a model and optimizer state from a checkpoint file."""
    checkpoint = torch.load(checkpoint_path)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    avg_train_loss = checkpoint['train_loss']
    avg_valid_cer = checkpoint['valid_cer']
    config_trOCR["lr"] = checkpoint['lr']
    config_trOCR["batch_size"] = checkpoint['batch_size']
    config_trOCR["num_epochs"] = checkpoint['num_epochs']
    config_trOCR["max_target_length"] = checkpoint['max_target_length']
    
    print(f"Checkpoint loaded from {checkpoint_path}, epoch {epoch}, "
          f"train_loss {avg_train_loss}, valid_cer {avg_valid_cer}")
    
    return epoch, avg_train_loss, avg_valid_cer