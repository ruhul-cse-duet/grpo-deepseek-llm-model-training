"""
Language Reward Functions
Rewards for language consistency
"""

import langid
from config.reward_config import RewardConfig


class LanguageRewardFunction:
    """Reward function for language consistency"""
    
    def __init__(self, target_language="id"):
        """
        Initialize language reward function
        
        Args:
            target_language: Target language code (e.g., 'id' for Bahasa Indonesia)
        """
        self.target_language = target_language
        self.config = RewardConfig()
        
        # Configure langid for better accuracy
        langid.set_languages(['id', 'en', 'zh'])
    
    def language_reward(self, completions, **kwargs):
        """
        Reward completions in target language
        
        Args:
            completions: List of completion dictionaries
            
        Returns:
            List of reward scores
        """
        scores = []
        
        for completion in completions:
            # Validate completion format
            if not completion or not isinstance(completion[0], dict) or "content" not in completion[0]:
                scores.append(self.config.LANGUAGE_OTHER_PENALTY)
                print(f"Warning: Malformed completion, assigning penalty: {completion}")
                continue
            
            content = completion[0]["content"]
            
            # Detect language
            lang = self._get_language(content)
            
            # Assign reward based on language
            if lang == self.target_language:
                score = self.config.LANGUAGE_ID_REWARD
            elif lang == "en":
                score = self.config.LANGUAGE_EN_PENALTY
            elif lang == "zh":
                score = self.config.LANGUAGE_ZH_PENALTY
            else:
                score = self.config.LANGUAGE_OTHER_PENALTY
            
            scores.append(score)
        
        return scores
    
    def _get_language(self, text):
        """
        Detect language of text
        
        Args:
            text: Input text
            
        Returns:
            Language code (e.g., 'id', 'en', 'zh')
        """
        if not text:
            return "und"  # Undetermined
        
        try:
            lang, confidence = langid.classify(text)
            return lang
        except:
            return "und"
    
    def get_all_language_rewards(self):
        """
        Get all language reward functions
        
        Returns:
            List of reward functions
        """
        return [self.language_reward]
