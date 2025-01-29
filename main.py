import optuna
import os
import torch
import logging
import atexit
import threading
from datetime import datetime
from train import train

logging.basicConfig(filename='experiment.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger()
handler = logger.handlers[0]
handler.terminator = "\n"

lock = threading.Lock()

def flush_logs():
    for handler in logger.handlers:
        handler.flush()
        
atexit.register(flush_logs)

my_gpu_queue = [0] * torch.cuda.device_count()

def get_gpu():
    with lock:
        for i in range(len(my_gpu_queue)):
            if my_gpu_queue[i] == 0: 
                my_gpu_queue[i] = 1
                return i
    return None

def free_gpu(gpu_id):
    with lock:
        my_gpu_queue[gpu_id] = 0

def objective(trial):
    start_time_trial = datetime.now()
    logging.info(f"Trial {trial.number + 4} started at {start_time_trial}")
    
    gpu_id = get_gpu()
    if gpu_id is None:
        raise RuntimeError("No free GPU available")
    
    config_trOCR = {
        "gpu_id": gpu_id,
        "id": trial.number + 4,
        "batch_size": trial.suggest_categorical("batch_size", [4, 8, 10]),
        "lr": trial.suggest_loguniform("lr", 1e-5, 1e-3),
        "num_epochs": trial.suggest_int("num_epochs", 10, 30),
        "max_target_length": trial.suggest_int("max_target_length", 64, 128)
    }
    
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

    result = train(config_trOCR, device)
    
    free_gpu(gpu_id)
    
    end_time_trial = datetime.now()
    logging.info(f"Trial {trial.number + 4} finished at {end_time_trial}")

    duration_trial = end_time_trial - start_time_trial
    logging.info(f"Total time taken for trial {trial.number + 4}: {duration_trial}")

    return result["CER"]

if __name__ == "__main__":
    start_time = datetime.now()
    logging.info(f"Experiment started at {start_time}")
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.RandomSampler())
    
    study.optimize(objective, n_trials=20, n_jobs=4)

    end_time = datetime.now()
    logging.info(f"Experiment finished at {end_time}")

    duration = end_time - start_time
    logging.info(f"Total time taken: {duration}")

    print("Best hyperparameters: ", study.best_params)
    print(f"Time taken: {duration}")