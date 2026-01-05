from config.model_config import ModelConfig
from config.training_config import TrainingConfig
from config.reward_config import RewardConfig
from data.dataset_loader import DatasetLoader
from data.data_preprocessor import DataPreprocessor
from data.prompts import PromptTemplates
from models.model_loader import ModelLoader
from rewards.format_rewards import FormatRewardFunction
from rewards.answer_rewards import AnswerRewardFunction
from rewards.language_rewards import LanguageRewardFunction
from training.trainer import GRPOModelTrainer


def main():
    # Load configurations
    model_config = ModelConfig()
    training_config = TrainingConfig()
    reward_config = RewardConfig()

    # Load model
    model_loader = ModelLoader(model_config)
    model, tokenizer = model_loader.load_model()

    # Load and prepare dataset
    dataset_loader = DatasetLoader()
    dataset = dataset_loader.load_openr1_math()

    preprocessor = DataPreprocessor(tokenizer)
    system_prompt = PromptTemplates.SYSTEM_PROMPT
    dataset = preprocessor.prepare_prompts(dataset, system_prompt)
    dataset, max_length = preprocessor.filter_by_length(dataset)

    # Setup reward functions
    reasoning_start, reasoning_end = PromptTemplates.get_reasoning_tokens(
        tokenizer
    )

    format_rewards = FormatRewardFunction(reasoning_end)
    answer_rewards = AnswerRewardFunction(reasoning_end)
    language_rewards = LanguageRewardFunction(target_language="id")

    reward_functions = [
        format_rewards.match_exact,
        format_rewards.match_approximate,
        answer_rewards.check_answer,
        answer_rewards.check_numbers,
        language_rewards.language_reward
    ]

    # Train model
    trainer = GRPOModelTrainer(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        config=training_config,
        reward_functions=reward_functions
    )

    trained_model = trainer.train()

    # Save model
    model.save_lora("grpo_lora")
    print("Training complete! Model saved to grpo_lora/")


if __name__ == "__main__":
    main()