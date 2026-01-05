"""
Answer Reward Functions
Rewards for answer correctness
"""

import re
from config.reward_config import RewardConfig


class AnswerRewardFunction:
    """Reward functions for answer correctness"""
    
    def __init__(self, reasoning_end_token):
        """
        Initialize answer reward function
        
        Args:
            reasoning_end_token: Token that marks end of reasoning
        """
        self.solution_end_regex = rf"{re.escape(reasoning_end_token)}(.*)"
        self.match_format = re.compile(
            self.solution_end_regex,
            re.DOTALL
        )
        
        # Regex to extract numbers from text
        self.match_numbers = re.compile(
            r".*?[\s]{0,}([-]?[\d\.\,]{1,})",
            flags=re.MULTILINE | re.DOTALL
        )
        
        self.config = RewardConfig()
        self.print_counter = 0
        self.print_every = 5  # Print every N samples
    
    def check_answer(self, prompts, completions, answer, **kwargs):
        """
        Check if answer matches ground truth
        
        Args:
            prompts: List of prompt messages
            completions: List of completion dictionaries
            answer: List of ground truth answers
            
        Returns:
            List of reward scores
        """
        question = prompts[0][-1]["content"]
        responses = [completion[0]["content"] for completion in completions]
        
        # Extract responses after </think> token
        extracted_responses = [
            guess.group(1)
            if (guess := self.match_format.search(r)) is not None else None
            for r in responses
        ]
        
        scores = []
        
        for guess, true_answer in zip(extracted_responses, answer):
            score = 0
            
            # Check if answer was extracted
            if guess is None:
                scores.append(self.config.ANSWER_MISSING_PENALTY)
                continue
            
            # Exact match
            if guess == true_answer:
                score += self.config.ANSWER_CORRECT_REWARD
            # Match with whitespace stripped
            elif guess.strip() == true_answer.strip():
                score += self.config.ANSWER_CLOSE_REWARD
            else:
                # Try numerical comparison with ratio
                score += self._calculate_ratio_score(guess, true_answer)
            
            scores.append(score)
        
        return scores
    
    def _calculate_ratio_score(self, guess, true_answer):
        """
        Calculate score based on numerical proximity
        
        Args:
            guess: Model's answer
            true_answer: Ground truth
            
        Returns:
            Reward score
        """
        try:
            ratio = float(guess) / float(true_answer)
            
            # Very close (within 10%)
            if 0.9 <= ratio <= 1.1:
                return self.config.ANSWER_RATIO_HIGH_REWARD
            # Somewhat close (within 20%)
            elif 0.8 <= ratio <= 1.2:
                return self.config.ANSWER_RATIO_MID_REWARD
            else:
                return self.config.ANSWER_WRONG_PENALTY
        except:
            return self.config.ANSWER_VERY_WRONG_PENALTY
    
    def check_numbers(self, prompts, completions, answer, **kwargs):
        """
        Extract and verify numerical answers
        
        Args:
            prompts: List of prompt messages
            completions: List of completion dictionaries
            answer: List of ground truth answers
            
        Returns:
            List of reward scores
        """
        question = prompts[0][-1]["content"]
        responses = [completion[0]["content"] for completion in completions]
        
        # Extract numbers from responses
        extracted_responses = [
            guess.group(1)
            if (guess := self.match_numbers.search(r)) is not None else None
            for r in responses
        ]
        
        scores = []
        
        # Print sample for debugging
        if self.print_counter % self.print_every == 0:
            print("*" * 20 + f" Question:\n{question}")
            print(f"\nAnswer:\n{answer[0]}")
            print(f"\nResponse:\n{responses[0]}")
            print(f"\nExtracted:\n{extracted_responses[0]}")
        self.print_counter += 1
        
        for guess, true_answer in zip(extracted_responses, answer):
            if guess is None:
                scores.append(self.config.NUMBER_MISSING_PENALTY)
                continue
            
            # Convert to numbers and compare
            try:
                true_val = float(true_answer.strip())
                guess_val = float(guess.strip().replace(",", ""))  # Remove commas
                
                if guess_val == true_val:
                    scores.append(self.config.NUMBER_CORRECT_REWARD)
                else:
                    scores.append(self.config.NUMBER_WRONG_PENALTY)
            except:
                scores.append(0)
                continue
        
        return scores
    
    def get_all_answer_rewards(self):
        """
        Get all answer reward functions
        
        Returns:
            List of reward functions
        """
        return [
            self.check_answer,
            self.check_numbers
        ]
