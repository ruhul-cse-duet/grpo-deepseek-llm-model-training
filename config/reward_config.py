"""
Reward Configuration
Defines reward weights for GRPO training
"""


class RewardConfig:
    """Reward function weights and parameters"""
    
    # Format Rewards
    FORMAT_EXACT_MATCH_REWARD = 3.0
    FORMAT_PARTIAL_MATCH_REWARD = 0.5
    FORMAT_MISSING_PENALTY = -1.0
    
    # Answer Rewards
    ANSWER_CORRECT_REWARD = 5.0
    ANSWER_CLOSE_REWARD = 3.5
    ANSWER_RATIO_HIGH_REWARD = 2.0  # When ratio is 0.9-1.1
    ANSWER_RATIO_MID_REWARD = 1.5   # When ratio is 0.8-1.2
    ANSWER_WRONG_PENALTY = -2.5
    ANSWER_VERY_WRONG_PENALTY = -4.5
    ANSWER_MISSING_PENALTY = -2.0
    
    # Number Extraction Rewards
    NUMBER_CORRECT_REWARD = 3.5
    NUMBER_WRONG_PENALTY = -1.5
    NUMBER_MISSING_PENALTY = -2.5
    
    # Language Consistency Rewards
    LANGUAGE_ID_REWARD = 5.0  # Bahasa Indonesia (target language)
    LANGUAGE_EN_PENALTY = -3.0  # English
    LANGUAGE_ZH_PENALTY = -3.0  # Chinese
    LANGUAGE_OTHER_PENALTY = -5.0  # Other languages
    TARGET_LANGUAGE = "id"  # ISO code for Bahasa Indonesia
    
    # Total Possible Reward (approximation)
    MAX_REWARD = 16.5  # Sum of all positive rewards
    
    @classmethod
    def get_reward_summary(cls):
        """Get summary of reward structure"""
        return {
            'format_rewards': {
                'exact_match': cls.FORMAT_EXACT_MATCH_REWARD,
                'partial_match': cls.FORMAT_PARTIAL_MATCH_REWARD,
                'missing_penalty': cls.FORMAT_MISSING_PENALTY,
            },
            'answer_rewards': {
                'correct': cls.ANSWER_CORRECT_REWARD,
                'close': cls.ANSWER_CLOSE_REWARD,
                'ratio_high': cls.ANSWER_RATIO_HIGH_REWARD,
                'ratio_mid': cls.ANSWER_RATIO_MID_REWARD,
                'wrong': cls.ANSWER_WRONG_PENALTY,
                'very_wrong': cls.ANSWER_VERY_WRONG_PENALTY,
                'missing': cls.ANSWER_MISSING_PENALTY,
            },
            'number_rewards': {
                'correct': cls.NUMBER_CORRECT_REWARD,
                'wrong': cls.NUMBER_WRONG_PENALTY,
                'missing': cls.NUMBER_MISSING_PENALTY,
            },
            'language_rewards': {
                'target_language': cls.TARGET_LANGUAGE,
                'id_reward': cls.LANGUAGE_ID_REWARD,
                'en_penalty': cls.LANGUAGE_EN_PENALTY,
                'zh_penalty': cls.LANGUAGE_ZH_PENALTY,
                'other_penalty': cls.LANGUAGE_OTHER_PENALTY,
            },
            'max_reward': cls.MAX_REWARD,
        }
    
    @classmethod
    def print_config(cls):
        """Print current configuration"""
        print("=" * 50)
        print("REWARD CONFIGURATION")
        print("=" * 50)
        print(f"Format Exact Match: +{cls.FORMAT_EXACT_MATCH_REWARD}")
        print(f"Answer Correct: +{cls.ANSWER_CORRECT_REWARD}")
        print(f"Number Correct: +{cls.NUMBER_CORRECT_REWARD}")
        print(f"Language ({cls.TARGET_LANGUAGE}): +{cls.LANGUAGE_ID_REWARD}")
        print(f"Maximum Possible Reward: {cls.MAX_REWARD}")
        print("=" * 50)
