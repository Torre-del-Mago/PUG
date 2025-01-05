import os
import csv
from PIL import Image
import torch    
from torch.utils.data import Dataset

class OCRDataset(Dataset):
    def __init__(self, base_path, processor, max_target_length=128):

        """
        Klasa Dataset dla OCR, która ładuje obrazy z katalogu i transkrypcje z pliku CSV.

        Args:
            base_path (str): Ścieżka bazowa, która zawiera obrazy i plik CSV.
            processor (TrOCRProcessor): Procesor TrOCR do przetwarzania obrazów i transkrypcji.
        """

        self.base_path = base_path
        self.image_dir = os.path.join(base_path, "output")
        self.csv_file = os.path.join(base_path, "output", "labels.csv")
        
        self.processor = processor
        self.max_target_length = max_target_length

        # Ładujemy dane z pliku CSV
        self.data = []
        with open(self.csv_file, mode='r', encoding='utf-8') as file:
            reader = csv.reader(file, delimiter=';')
            for row in reader:
                image_path, transcription = row
                self.data.append({
                    "image_path": os.path.join(self.image_dir, image_path),
                    "transcription": transcription.strip()
                })

    def __len__(self):
        """Zwraca liczbę próbek w zbiorze danych."""
        return len(self.data)

    def __getitem__(self, idx):
        """
        Zwraca obraz i transkrypcję w formacie odpowiednim dla modelu.

        Args:
            idx (int): Indeks próbki w zbiorze danych.

        Returns:
            dict: Słownik z wartościami obrazu (pixel_values) i etykietami (labels).
        """
        # Pobierz nazwę pliku obrazu i tekst
        entry = self.data[idx]
        image_path = entry["image_path"]
        transcription = entry["transcription"]

        # prepare image (i.e. resize + normalize)
        image = Image.open(image_path).convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        # add labels (input_ids) by encoding the text
        labels = self.processor.tokenizer(transcription, 
                                          padding="max_length", 
                                          max_length=self.max_target_length,
                                          truncation=True).input_ids
        # important: make sure that PAD tokens are ignored by the loss function
        labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]
        
        # returning as dictionary
        encoding = {
            "pixel_values": pixel_values.squeeze(),  # Usuwamy nadmiarowy wymiar wsadu
            "labels": torch.tensor(labels)  # Etykiety jako tensor
        }
        return encoding
