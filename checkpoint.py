import os
import torch

def save_checkpoint(model, optimizer, epoch, avg_train_loss, avg_valid_cer, checkpoint_dir="checkpoints", config_id=0):
    """Saves the model and optimizer state to a checkpoint file."""
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': avg_train_loss,
        'valid_cer': avg_valid_cer,
    }
    
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{config_id}.pth")
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")

def load_checkpoint(model, optimizer, checkpoint_path):
    """Loads a model and optimizer state from a checkpoint file."""
    checkpoint = torch.load(checkpoint_path)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    avg_train_loss = checkpoint['train_loss']
    avg_valid_cer = checkpoint['valid_cer']
    
    print(f"Checkpoint loaded from {checkpoint_path}, epoch {epoch}, "
          f"train_loss {avg_train_loss}, valid_cer {avg_valid_cer}")
    
    return epoch, avg_train_loss, avg_valid_cer