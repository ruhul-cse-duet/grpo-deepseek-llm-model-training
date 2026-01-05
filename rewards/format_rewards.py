"""
Format Reward Functions
Rewards for output format matching
"""

import re
from config.reward_config import RewardConfig


class FormatRewardFunction:
    """Reward functions for output format matching"""
    
    def __init__(self, reasoning_end_token):
        """
        Initialize format reward function
        
        Args:
            reasoning_end_token: Token that marks end of reasoning (e.g., </think>)
        """
        self.reasoning_end = reasoning_end_token
        self.reasoning_start = reasoning_end_token.replace("/", "")
        
        # Regex to match format: </think>answer
        self.match_format = re.compile(
            rf"{re.escape(reasoning_end_token)}(.*)",
            re.DOTALL
        )
        
        self.config = RewardConfig()
    
    def match_exact(self, completions, **kwargs):
        """
        Reward exact format matches
        
        Args:
            completions: List of completion dictionaries
            
        Returns:
            List of reward scores
        """
        scores = []
        
        for completion in completions:
            response = completion[0]["content"]
            
            # Check if format matches exactly
            if self.match_format.search(response) is not None:
                score = self.config.FORMAT_EXACT_MATCH_REWARD
            else:
                score = 0
            
            scores.append(score)
        
        return scores
    
    def match_approximate(self, completions, **kwargs):
        """
        Reward partial format matches
        
        Args:
            completions: List of completion dictionaries
            
        Returns:
            List of reward scores
        """
        scores = []
        
        for completion in completions:
            response = completion[0]["content"]
            score = 0
            
            # Count reasoning tokens
            # Note: We don't reward <think> since it's prepended automatically
            start_count = response.count(self.reasoning_start)
            end_count = response.count(self.reasoning_end)
            
            # Reward if exactly one of each token
            if start_count == 1:
                score += self.config.FORMAT_PARTIAL_MATCH_REWARD
            elif start_count > 1:
                score += self.config.FORMAT_MISSING_PENALTY
            
            if end_count == 1:
                score += self.config.FORMAT_PARTIAL_MATCH_REWARD
            elif end_count > 1:
                score += self.config.FORMAT_MISSING_PENALTY
            
            scores.append(score)
        
        return scores
    
    def get_all_format_rewards(self):
        """
        Get all format reward functions
        
        Returns:
            List of reward functions
        """
        return [
            self.match_exact,
            self.match_approximate
        ]
