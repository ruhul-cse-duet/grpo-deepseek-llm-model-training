from trl import GRPOConfig, GRPOTrainer
from vllm import SamplingParams


class GRPOModelTrainer:
    """Main GRPO training orchestrator"""

    def __init__(
            self,
            model,
            tokenizer,
            dataset,
            config,
            reward_functions
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.config = config
        self.reward_functions = reward_functions

    def setup_training(self):
        """Configure GRPO training"""
        vllm_params = SamplingParams(
            min_p=self.config.MIN_P,
            top_p=self.config.TOP_P,
            top_k=self.config.TOP_K,
            temperature=self.config.TEMPERATURE,
            seed=3407,
            stop=[self.tokenizer.eos_token],
            include_stop_str_in_output=True
        )

        training_args = GRPOConfig(
            vllm_sampling_params=vllm_params,
            temperature=self.config.TEMPERATURE,
            learning_rate=self.config.LEARNING_RATE,
            weight_decay=self.config.WEIGHT_DECAY,
            warmup_ratio=self.config.WARMUP_RATIO,
            lr_scheduler_type=self.config.LR_SCHEDULER_TYPE,
            optim=self.config.OPTIMIZER,
            logging_steps=1,
            per_device_train_batch_size=self.config.PER_DEVICE_BATCH_SIZE,
            gradient_accumulation_steps=self.config.GRADIENT_ACCUMULATION_STEPS,
            num_generations=self.config.NUM_GENERATIONS,
            max_steps=self.config.MAX_STEPS,
            save_steps=self.config.SAVE_STEPS,
            report_to="none",
            output_dir="outputs"
        )

        return training_args

    def train(self):
        """Execute GRPO training"""
        training_args = self.setup_training()

        trainer = GRPOTrainer(
            model=self.model,
            processing_class=self.tokenizer,
            reward_funcs=self.reward_functions,
            args=training_args,
            train_dataset=self.dataset
        )

        trainer.train()
        return trainer