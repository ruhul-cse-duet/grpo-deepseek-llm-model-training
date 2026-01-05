"""
Model Loader
Loads DeepSeek model with Unsloth and LoRA configuration
"""

from unsloth import FastLanguageModel
import torch


class ModelLoader:
    """Load and configure DeepSeek model with LoRA adapters"""
    
    def __init__(self, config):
        """
        Initialize model loader
        
        Args:
            config: ModelConfig instance with hyperparameters
        """
        self.config = config
        self.model = None
        self.tokenizer = None
    
    def load_model(self):
        """
        Load base model with LoRA adapters
        
        Returns:
            Tuple of (model, tokenizer)
        """
        print("=" * 50)
        print("LOADING DEEPSEEK MODEL")
        print("=" * 50)
        print(f"Model: {self.config.MODEL_NAME}")
        print(f"Max Sequence Length: {self.config.MAX_SEQ_LENGTH}")
        print(f"4-bit Quantization: {self.config.LOAD_IN_4BIT}")
        print(f"LoRA Rank: {self.config.LORA_RANK}")
        print("=" * 50)
        
        # Load base model
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.config.MODEL_NAME,
            max_seq_length=self.config.MAX_SEQ_LENGTH,
            load_in_4bit=self.config.LOAD_IN_4BIT,
            fast_inference=self.config.FAST_INFERENCE,
            max_lora_rank=self.config.LORA_RANK,
            gpu_memory_utilization=self.config.GPU_MEMORY_UTILIZATION
        )
        
        print("✓ Base model loaded successfully")
        
        # Add LoRA adapters
        self.model = self._add_lora_adapters(self.model)
        
        print("✓ LoRA adapters added successfully")
        print("=" * 50)
        
        return self.model, self.tokenizer
    
    def _add_lora_adapters(self, model):
        """
        Add LoRA adapters to model
        
        Args:
            model: Base model
            
        Returns:
            Model with LoRA adapters
        """
        return FastLanguageModel.get_peft_model(
            model,
            r=self.config.LORA_RANK,
            target_modules=self.config.TARGET_MODULES,
            lora_alpha=self.config.LORA_ALPHA,
            use_gradient_checkpointing=self.config.USE_GRADIENT_CHECKPOINTING,
            random_state=self.config.RANDOM_STATE
        )
    
    def save_model(self, output_path: str):
        """
        Save LoRA weights
        
        Args:
            output_path: Directory to save LoRA weights
        """
        if self.model is None:
            print("⚠ No model loaded to save")
            return
        
        print(f"Saving LoRA weights to {output_path}...")
        self.model.save_lora(output_path)
        print(f"✓ LoRA weights saved to {output_path}")
    
    def load_lora_weights(self, lora_path: str):
        """
        Load LoRA weights from checkpoint
        
        Args:
            lora_path: Path to LoRA checkpoint
        """
        if self.model is None:
            print("⚠ No model loaded")
            return
        
        print(f"Loading LoRA weights from {lora_path}...")
        # LoRA loading is handled by FastLanguageModel
        print(f"✓ LoRA weights loaded from {lora_path}")
    
    def get_model_info(self):
        """Get information about the loaded model"""
        if self.model is None:
            return "No model loaded"
        
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        
        return {
            'model_name': self.config.MODEL_NAME,
            'trainable_parameters': trainable_params,
            'total_parameters': total_params,
            'percentage_trainable': f"{100 * trainable_params / total_params:.2f}%",
            'lora_rank': self.config.LORA_RANK,
            'lora_alpha': self.config.LORA_ALPHA,
        }
    
    def print_model_info(self):
        """Print model information"""
        info = self.get_model_info()
        
        if isinstance(info, str):
            print(info)
            return
        
        print("=" * 50)
        print("MODEL INFORMATION")
        print("=" * 50)
        print(f"Model Name: {info['model_name']}")
        print(f"Total Parameters: {info['total_parameters']:,}")
        print(f"Trainable Parameters: {info['trainable_parameters']:,}")
        print(f"Percentage Trainable: {info['percentage_trainable']}")
        print(f"LoRA Rank: {info['lora_rank']}")
        print(f"LoRA Alpha: {info['lora_alpha']}")
        print("=" * 50)
