import re
from typing import Optional, List, Tuple


class TextProcessor:
    """Text processing utilities for model outputs"""

    def __init__(self, reasoning_start="<think>", reasoning_end="</think>"):
        self.reasoning_start = reasoning_start
        self.reasoning_end = reasoning_end
        self.number_pattern = re.compile(
            r".*?[\s]{0,}([-]?[\d\.\,]{1,})",
            flags=re.MULTILINE | re.DOTALL
        )

    def extract_reasoning(self, text: str) -> Optional[str]:
        """Extract reasoning section from response"""
        pattern = rf"{re.escape(self.reasoning_start)}(.*?){re.escape(self.reasoning_end)}"
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1).strip() if match else None

    def extract_answer(self, text: str) -> Optional[str]:
        """Extract final answer after reasoning"""
        pattern = rf"{re.escape(self.reasoning_end)}(.*)"
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1).strip() if match else None

    def extract_number(self, text: str) -> Optional[str]:
        """Extract first number from text"""
        match = self.number_pattern.search(text)
        return match.group(1) if match else None

    def clean_number(self, number_str: str) -> float:
        """Clean and convert number string to float"""
        try:
            # Remove commas and convert
            cleaned = number_str.strip().replace(",", "")
            return float(cleaned)
        except (ValueError, AttributeError):
            raise ValueError(f"Cannot convert '{number_str}' to number")

    def split_into_steps(self, reasoning: str) -> List[str]:
        """Split reasoning into logical steps"""
        # Split by common step indicators
        step_indicators = [
            r'\n\d+\.',  # 1. 2. 3.
            r'\nStep \d+:',  # Step 1: Step 2:
            r'\n-',  # Bullet points
            r'\n\*',  # Asterisk bullets
        ]

        pattern = '|'.join(step_indicators)
        steps = re.split(pattern, reasoning)
        return [step.strip() for step in steps if step.strip()]

    def calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple word-based similarity between texts"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union)

    def truncate_text(self, text: str, max_tokens: int,
                      tokenizer=None) -> str:
        """Truncate text to maximum token length"""
        if tokenizer is None:
            # Simple word-based truncation
            words = text.split()
            if len(words) <= max_tokens:
                return text
            return ' '.join(words[:max_tokens]) + '...'
        else:
            # Token-based truncation
            tokens = tokenizer.encode(text)
            if len(tokens) <= max_tokens:
                return text
            truncated_tokens = tokens[:max_tokens]
            return tokenizer.decode(truncated_tokens) + '...'

    def count_reasoning_tokens(self, text: str) -> Tuple[int, int, int]:
        """Count tokens in reasoning vs answer sections"""
        reasoning = self.extract_reasoning(text)
        answer = self.extract_answer(text)

        reasoning_words = len(reasoning.split()) if reasoning else 0
        answer_words = len(answer.split()) if answer else 0
        total_words = len(text.split())

        return reasoning_words, answer_words, total_words

    def format_for_display(self, text: str, max_width: int = 80) -> str:
        """Format text for readable console display"""
        import textwrap

        reasoning = self.extract_reasoning(text)
        answer = self.extract_answer(text)

        output = []

        if reasoning:
            output.append("=== REASONING ===")
            wrapped = textwrap.fill(reasoning, width=max_width)
            output.append(wrapped)
            output.append("")

        if answer:
            output.append("=== ANSWER ===")
            wrapped = textwrap.fill(answer, width=max_width)
            output.append(wrapped)

        return '\n'.join(output)


class ChatTemplateFormatter:
    """Format conversations for model input"""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def format_single_turn(self, prompt: str, system_prompt: Optional[str] = None):
        """Format single user prompt"""
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        return self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False
        )

    def format_multi_turn(self, conversation: List[dict]):
        """Format multi-turn conversation"""
        return self.tokenizer.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False
        )

    def add_reasoning_prefix(self, formatted_text: str,
                             reasoning_start: str = "<think>"):
        """Add reasoning token prefix to generation"""
        return formatted_text + reasoning_start