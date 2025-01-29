import os
import json
import random
from torch.utils.data import Dataset
from craft_hw_ocr.OCR import *

class CraftDataset(Dataset):
    def __init__(self, base_path, split="training", num_samples=100):
        """
        Dataset do trenowania modelu CRAFT, przyjmującego obrazy i dane z pliku JSON.
        
        Args:
            base_path (str): Ścieżka do bazowego folderu, z którego tworzymy ścieżki.
            split (str, optional): Wartość "train" lub "val" określająca, czy używamy danych treningowych czy walidacyjnych.
            num_samples (int, optional): Liczba losowych próbek do załadowania. Domyślnie 100.
        """
        self.base_path = base_path
        self.split = split
        self.num_samples = num_samples

        self.image_dir = os.path.join(self.base_path, split, "images")
        self.json_file = os.path.join(self.base_path, split, f"{split}_combined.json")

        with open(self.json_file, 'r') as f:
            self.all_data = json.load(f)

        # Wybierz początkowe próbki
        self.refresh_samples()

    def refresh_samples(self):
        """
        Odśwież losowy wybór próbek.
        """
        if len(self.all_data) > self.num_samples:
            self.data = random.sample(self.all_data, self.num_samples)
        else:
            self.data = self.all_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = self.data[idx]["ocr"]   
        img = load_image(os.path.join(self.image_dir, image_path))
        
        if img is None:
            raise ValueError(f"Image at index {idx} is None. Check your data loading process. Path to image {image_path}. My base path: {self.image_dir} ")

        # img = process_image(img)

        bboxes = self.data[idx]["bbox"]
        transcription = self.data[idx]["transcription"]

        regions = []
        for bbox in bboxes:
            x, y, w, h = bbox["x"], bbox["y"], bbox["width"], bbox["height"]
            cropped_region = img[int(y):int(y+h), int(x):int(x+w)]  # Wycinamy region tekstowy
            regions.append(cropped_region)

        # Zwróć obraz i regiony
        return img, regions, transcription

        
if __name__ == "__main__":
    # Przykład użycia
    dataset = CraftDataset(base_path="/mnt/raid/splits", split="test", num_samples=100)
    # Aby odświeżyć losowo wybrane indeksy:
    dataset.refresh_samples()