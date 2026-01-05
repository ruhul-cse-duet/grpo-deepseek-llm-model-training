"""
LoRA Configuration
Additional LoRA configuration utilities
"""

from peft import LoraConfig


class LoRAConfigBuilder:
    """Build custom LoRA configurations"""
    
    @staticmethod
    def create_default_config(r=32, lora_alpha=64, target_modules=None):
        """
        Create default LoRA configuration
        
        Args:
            r: LoRA rank
            lora_alpha: LoRA alpha (scaling factor)
            target_modules: List of modules to apply LoRA
            
        Returns:
            LoraConfig instance
        """
        if target_modules is None:
            target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]
        
        return LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
    
    @staticmethod
    def create_high_rank_config():
        """Create high-rank LoRA for better performance"""
        return LoRAConfigBuilder.create_default_config(
            r=64,
            lora_alpha=128
        )
    
    @staticmethod
    def create_low_rank_config():
        """Create low-rank LoRA for faster training"""
        return LoRAConfigBuilder.create_default_config(
            r=16,
            lora_alpha=32
        )
    
    @staticmethod
    def create_attention_only_config():
        """Apply LoRA only to attention layers"""
        return LoRAConfigBuilder.create_default_config(
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
        )
    
    @staticmethod
    def create_mlp_only_config():
        """Apply LoRA only to MLP layers"""
        return LoRAConfigBuilder.create_default_config(
            target_modules=["gate_proj", "up_proj", "down_proj"]
        )
