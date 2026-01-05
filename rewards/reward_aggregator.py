"""
Reward Aggregator
Combines all reward functions
"""


class RewardAggregator:
    """Combine all reward functions for GRPO training"""
    
    def __init__(self, reward_functions):
        """
        Initialize reward aggregator
        
        Args:
            reward_functions: List of reward functions
        """
        self.reward_functions = reward_functions
    
    def get_all_rewards(self):
        """
        Get list of all reward functions
        
        Returns:
            List of reward functions
        """
        return self.reward_functions
    
    def add_reward_function(self, reward_func):
        """
        Add a new reward function
        
        Args:
            reward_func: Reward function to add
        """
        self.reward_functions.append(reward_func)
    
    def remove_reward_function(self, index):
        """
        Remove a reward function by index
        
        Args:
            index: Index of reward function to remove
        """
        if 0 <= index < len(self.reward_functions):
            self.reward_functions.pop(index)
