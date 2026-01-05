"""
Model Configuration
Defines hyperparameters for model loading and LoRA setup
"""


class ModelConfig:
    """Model and tokenizer configuration"""
    
    # Base Model Configuration
    MODEL_NAME = "unsloth/DeepSeek-R1-0528-Qwen3-8B"
    MAX_SEQ_LENGTH = 1024  # Can increase for longer reasoning traces
    LOAD_IN_4BIT = True    # Use 4-bit quantization for memory efficiency
    FAST_INFERENCE = True  # Enable vLLM fast inference
    GPU_MEMORY_UTILIZATION = 0.9  # Reduce if out of memory
    
    # LoRA Configuration
    LORA_RANK = 32  # Larger rank = smarter, but slower
    LORA_ALPHA = 64  # Typically 2 * lora_rank for faster training
    TARGET_MODULES = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ]
    
    # Training Optimization
    USE_GRADIENT_CHECKPOINTING = "unsloth"  # Reduces memory usage
    RANDOM_STATE = 3407
    
    @classmethod
    def get_config_dict(cls):
        """Return configuration as dictionary"""
        return {
            'model_name': cls.MODEL_NAME,
            'max_seq_length': cls.MAX_SEQ_LENGTH,
            'load_in_4bit': cls.LOAD_IN_4BIT,
            'fast_inference': cls.FAST_INFERENCE,
            'gpu_memory_utilization': cls.GPU_MEMORY_UTILIZATION,
            'lora_rank': cls.LORA_RANK,
            'lora_alpha': cls.LORA_ALPHA,
            'target_modules': cls.TARGET_MODULES,
        }
    
    @classmethod
    def print_config(cls):
        """Print current configuration"""
        print("=" * 50)
        print("MODEL CONFIGURATION")
        print("=" * 50)
        print(f"Model Name: {cls.MODEL_NAME}")
        print(f"Max Sequence Length: {cls.MAX_SEQ_LENGTH}")
        print(f"4-bit Quantization: {cls.LOAD_IN_4BIT}")
        print(f"Fast Inference: {cls.FAST_INFERENCE}")
        print(f"GPU Memory Utilization: {cls.GPU_MEMORY_UTILIZATION}")
        print(f"LoRA Rank: {cls.LORA_RANK}")
        print(f"LoRA Alpha: {cls.LORA_ALPHA}")
        print(f"Target Modules: {', '.join(cls.TARGET_MODULES)}")
        print("=" * 50)
