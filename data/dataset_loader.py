"""
Dataset Loader
Loads various math reasoning datasets
"""

from datasets import load_dataset
from typing import Optional


class DatasetLoader:
    """Load and prepare math datasets for GRPO training"""
    
    def __init__(self):
        self.dataset = None
        self.dataset_name = None
    
    def load_openr1_math(self, language: str = "en", split: str = "train"):
        """
        Load Open-R1 Math dataset
        
        Args:
            language: Language code (en, zh, etc.)
            split: Dataset split to load
            
        Returns:
            Loaded dataset
        """
        print(f"Loading Open-R1 Math dataset (language: {language}, split: {split})...")
        
        self.dataset = load_dataset(
            "open-r1/DAPO-Math-17k-Processed",
            language,
            split=split
        )
        self.dataset_name = f"open_r1_math_{language}"
        
        print(f"✓ Loaded {len(self.dataset)} examples from Open-R1 Math")
        return self.dataset
    
    def load_gsm8k(self, split: str = "train"):
        """
        Load GSM8K dataset (alternative math dataset)
        
        Args:
            split: Dataset split to load
            
        Returns:
            Loaded dataset
        """
        print(f"Loading GSM8K dataset (split: {split})...")
        
        self.dataset = load_dataset("openai/gsm8k", split=split)
        self.dataset_name = "gsm8k"
        
        print(f"✓ Loaded {len(self.dataset)} examples from GSM8K")
        return self.dataset
    
    def load_custom_dataset(self, dataset_path: str, split: str = "train"):
        """
        Load custom dataset from local path or HuggingFace
        
        Args:
            dataset_path: Path or HuggingFace dataset identifier
            split: Dataset split to load
            
        Returns:
            Loaded dataset
        """
        print(f"Loading custom dataset from {dataset_path}...")
        
        self.dataset = load_dataset(dataset_path, split=split)
        self.dataset_name = "custom"
        
        print(f"✓ Loaded {len(self.dataset)} examples from custom dataset")
        return self.dataset
    
    def get_dataset_info(self):
        """Get information about the loaded dataset"""
        if self.dataset is None:
            return "No dataset loaded"
        
        return {
            'name': self.dataset_name,
            'num_examples': len(self.dataset),
            'features': self.dataset.features,
            'columns': self.dataset.column_names,
        }
    
    def print_example(self, index: int = 0):
        """Print an example from the dataset"""
        if self.dataset is None:
            print("No dataset loaded")
            return
        
        example = self.dataset[index]
        print("=" * 50)
        print(f"EXAMPLE {index}")
        print("=" * 50)
        for key, value in example.items():
            print(f"{key}: {value}")
        print("=" * 50)
