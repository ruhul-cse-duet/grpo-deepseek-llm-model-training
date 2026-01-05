"""
Training Configuration
Defines GRPO training hyperparameters and sampling settings
"""


class TrainingConfig:
    """GRPO training hyperparameters"""
    
    # Learning Rate Settings
    LEARNING_RATE = 5e-6
    WEIGHT_DECAY = 0.001
    WARMUP_RATIO = 0.1
    LR_SCHEDULER_TYPE = "linear"  # Options: "linear", "cosine", "constant"
    OPTIMIZER = "adamw_8bit"  # Memory-efficient optimizer
    
    # Batch Settings
    PER_DEVICE_BATCH_SIZE = 1
    GRADIENT_ACCUMULATION_STEPS = 1  # Increase to 4 for smoother training
    NUM_GENERATIONS = 4  # Number of completions per prompt (decrease if OOM)
    
    # Training Steps
    MAX_STEPS = 100  # Set to None to use num_train_epochs instead
    NUM_TRAIN_EPOCHS = None  # Set to 1 for full training run
    SAVE_STEPS = 100
    LOGGING_STEPS = 1
    
    # Sampling Parameters (vLLM)
    TEMPERATURE = 1.0
    TOP_P = 1.0
    TOP_K = -1  # -1 means disabled
    MIN_P = 0.1
    SEED = 3407
    
    # Output Settings
    OUTPUT_DIR = "outputs"
    REPORT_TO = "none"  # Options: "none", "wandb", "tensorboard"
    
    # Advanced Settings
    FP16_FULL_EVAL = True
    EVAL_ACCUMULATION_STEPS = 1
    EVAL_STRATEGY = "steps"  # Options: "no", "steps", "epoch"
    EVAL_STEPS = 100
    
    @classmethod
    def get_sampling_params(cls, tokenizer):
        """Get vLLM sampling parameters"""
        from vllm import SamplingParams
        
        return SamplingParams(
            min_p=cls.MIN_P,
            top_p=cls.TOP_P,
            top_k=cls.TOP_K,
            temperature=cls.TEMPERATURE,
            seed=cls.SEED,
            stop=[tokenizer.eos_token],
            include_stop_str_in_output=True,
        )
    
    @classmethod
    def get_training_args(cls, max_prompt_length, max_completion_length, tokenizer):
        """Get GRPOConfig training arguments"""
        from trl import GRPOConfig
        
        return GRPOConfig(
            vllm_sampling_params=cls.get_sampling_params(tokenizer),
            temperature=cls.TEMPERATURE,
            learning_rate=cls.LEARNING_RATE,
            weight_decay=cls.WEIGHT_DECAY,
            warmup_ratio=cls.WARMUP_RATIO,
            lr_scheduler_type=cls.LR_SCHEDULER_TYPE,
            optim=cls.OPTIMIZER,
            logging_steps=cls.LOGGING_STEPS,
            per_device_train_batch_size=cls.PER_DEVICE_BATCH_SIZE,
            gradient_accumulation_steps=cls.GRADIENT_ACCUMULATION_STEPS,
            num_generations=cls.NUM_GENERATIONS,
            max_prompt_length=max_prompt_length,
            max_completion_length=max_completion_length,
            max_steps=cls.MAX_STEPS,
            num_train_epochs=cls.NUM_TRAIN_EPOCHS,
            save_steps=cls.SAVE_STEPS,
            report_to=cls.REPORT_TO,
            output_dir=cls.OUTPUT_DIR,
        )
    
    @classmethod
    def print_config(cls):
        """Print current configuration"""
        print("=" * 50)
        print("TRAINING CONFIGURATION")
        print("=" * 50)
        print(f"Learning Rate: {cls.LEARNING_RATE}")
        print(f"Weight Decay: {cls.WEIGHT_DECAY}")
        print(f"Optimizer: {cls.OPTIMIZER}")
        print(f"Batch Size: {cls.PER_DEVICE_BATCH_SIZE}")
        print(f"Gradient Accumulation Steps: {cls.GRADIENT_ACCUMULATION_STEPS}")
        print(f"Num Generations: {cls.NUM_GENERATIONS}")
        print(f"Max Steps: {cls.MAX_STEPS}")
        print(f"Temperature: {cls.TEMPERATURE}")
        print(f"Output Directory: {cls.OUTPUT_DIR}")
        print("=" * 50)
