"""
Prompt Templates
System prompts and chat templates for GRPO training
"""


class PromptTemplates:
    """System prompts and chat templates"""
    
    # Default system prompt for Bahasa Indonesia reasoning
    SYSTEM_PROMPT = """You are given a problem.
Think about the problem and provide your working out.
You must think in Bahasa Indonesia."""
    
    # Alternative system prompts
    SYSTEM_PROMPT_ENGLISH = """You are given a problem.
Think about the problem step by step and provide your working out."""
    
    SYSTEM_PROMPT_DETAILED = """You are a mathematical reasoning assistant.
Given a math problem, you must:
1. Think through the problem carefully in Bahasa Indonesia
2. Show your step-by-step reasoning inside <think> tags
3. Provide the final answer after </think>

Example:
User: What is 5 + 3?
Assistant: <think>Saya perlu menjumlahkan 5 dan 3. 5 + 3 = 8.</think>8"""
    
    @staticmethod
    def get_reasoning_tokens(tokenizer):
        """
        Extract special reasoning tokens from tokenizer
        
        Args:
            tokenizer: HuggingFace tokenizer
            
        Returns:
            Tuple of (reasoning_start, reasoning_end, user_token, assistant_token)
        """
        reasoning_start = None
        reasoning_end = None
        user_token = None
        assistant_token = None
        
        for token in tokenizer.get_added_vocab().keys():
            if "think" in token and "/" in token:
                reasoning_end = token
            elif "think" in token:
                reasoning_start = token
            elif "user" in token:
                user_token = token
            elif "assistant" in token:
                assistant_token = token
        
        print(f"✓ Reasoning Start Token: {reasoning_start}")
        print(f"✓ Reasoning End Token: {reasoning_end}")
        
        return reasoning_start, reasoning_end, user_token, assistant_token
    
    @staticmethod
    def format_chat_example(system_prompt: str, user_message: str, assistant_response: str = None):
        """
        Format a chat example
        
        Args:
            system_prompt: System instruction
            user_message: User's question
            assistant_response: Optional assistant response
            
        Returns:
            Formatted chat messages
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        if assistant_response:
            messages.append({"role": "assistant", "content": assistant_response})
        
        return messages
    
    @staticmethod
    def create_reasoning_response(thinking: str, answer: str) -> str:
        """
        Create a properly formatted reasoning response
        
        Args:
            thinking: Reasoning process (in Bahasa Indonesia)
            answer: Final answer
            
        Returns:
            Formatted response with think tags
        """
        return f"<think>{thinking}</think>{answer}"
