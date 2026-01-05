"""
Data Preprocessor
Prepares and filters datasets for training
"""

import numpy as np
from typing import Optional, Tuple


class DataPreprocessor:
    """Preprocess and filter datasets for GRPO training"""
    
    def __init__(self, tokenizer, max_length_quantile: float = 0.9):
        """
        Initialize preprocessor
        
        Args:
            tokenizer: HuggingFace tokenizer
            max_length_quantile: Quantile for maximum sequence length (0.9 = remove top 10%)
        """
        self.tokenizer = tokenizer
        self.max_length_quantile = max_length_quantile
    
    def prepare_prompts(self, dataset, system_prompt: str):
        """
        Add system prompt and format conversations
        
        Args:
            dataset: HuggingFace dataset
            system_prompt: System instruction for the model
            
        Returns:
            Formatted dataset
        """
        print("Preparing prompts with chat format...")
        
        def format_example(example):
            return {
                "prompt": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": example["prompt"]}
                ],
                "answer": self._extract_answer(example.get("solution", example.get("answer", "")))
            }
        
        formatted_dataset = dataset.map(format_example)
        print(f"✓ Formatted {len(formatted_dataset)} examples")
        
        return formatted_dataset
    
    def filter_by_length(self, dataset) -> Tuple:
        """
        Remove samples that are too long (top percentage)
        
        Args:
            dataset: Formatted dataset
            
        Returns:
            Tuple of (filtered_dataset, max_length)
        """
        print(f"Filtering dataset by length (keeping {self.max_length_quantile*100}% quantile)...")
        
        # Tokenize to get lengths
        tokenized = dataset.map(
            lambda x: {
                "tokens": self.tokenizer.apply_chat_template(
                    x["prompt"],
                    add_generation_prompt=True,
                    tokenize=True
                )
            },
            batched=True
        )
        
        # Calculate lengths
        tokenized = tokenized.map(lambda x: {"L": len(x["tokens"])})
        
        # Get max length at quantile
        max_length = int(np.quantile(tokenized["L"], self.max_length_quantile))
        
        print(f"✓ Maximum length at {self.max_length_quantile*100}% quantile: {max_length} tokens")
        
        # Filter samples
        filtered_indices = np.where(np.array(tokenized["L"]) <= max_length)[0]
        filtered_dataset = dataset.select(filtered_indices)
        
        print(f"✓ Kept {len(filtered_dataset)}/{len(dataset)} examples ({len(filtered_dataset)/len(dataset)*100:.1f}%)")
        
        return filtered_dataset, max_length
    
    def _extract_answer(self, text: str) -> str:
        """
        Extract answer from solution text
        
        Args:
            text: Solution text (may contain #### marker)
            
        Returns:
            Extracted answer
        """
        # For GSM8K format with ####
        if "####" in text:
            return text.split("####")[1].strip()
        
        # For other formats, return as-is
        return text.strip()
    
    def create_train_test_split(self, dataset, test_size: float = 0.01):
        """
        Create train/test split
        
        Args:
            dataset: Dataset to split
            test_size: Fraction for test set
            
        Returns:
            Dictionary with 'train' and 'test' splits
        """
        print(f"Creating train/test split (test_size={test_size})...")
        
        split_dataset = dataset.train_test_split(test_size=test_size, seed=42)
        
        print(f"✓ Train: {len(split_dataset['train'])} examples")
        print(f"✓ Test: {len(split_dataset['test'])} examples")
        
        return split_dataset
    
    def print_formatted_example(self, dataset, index: int = 0):
        """Print a formatted example from the dataset"""
        example = dataset[index]
        
        print("=" * 50)
        print(f"FORMATTED EXAMPLE {index}")
        print("=" * 50)
        
        # Print chat format
        if "prompt" in example:
            for message in example["prompt"]:
                print(f"{message['role'].upper()}: {message['content'][:200]}...")
        
        # Print answer
        if "answer" in example:
            print(f"ANSWER: {example['answer']}")
        
        print("=" * 50)
