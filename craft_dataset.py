import os
import json
from torch.utils.data import Dataset
from craft_hw_ocr.OCR import * 

class CraftDataset(Dataset):
    def __init__(self, base_path, split="training"):
        """
        Dataset do trenowania modelu CRAFT, przyjmującego obrazy i dane z pliku JSON.
        
        Args:
            base_path (str): Ścieżka do bazowego folderu, z którego tworzymy ścieżki.
            split (str, optional): Wartość "train" lub "val" określająca, czy używamy danych treningowych czy walidacyjnych.
            transform (callable, optional): Funkcja transformująca obrazy, np. augmentacja.
        """
        self.base_path = base_path
        self.split = split

        self.image_dir = os.path.join(self.base_path, split, "images")
        self.json_file = os.path.join(self.base_path, split, f"{split}_combined.json")

        with open(self.json_file, 'r') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = self.data[idx]["ocr"]   
        img = load_image(image_path)

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
