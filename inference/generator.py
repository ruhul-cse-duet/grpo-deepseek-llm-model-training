from vllm import SamplingParams


class TextGenerator:
    """Generate text with trained model"""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate(
            self,
            prompt,
            temperature=1.0,
            max_tokens=1024,
            use_lora=False,
            lora_path=None
    ):
        """Generate completion for prompt"""
        sampling_params = SamplingParams(
            temperature=temperature,
            top_k=50,
            max_tokens=max_tokens
        )

        lora_request = None
        if use_lora and lora_path:
            lora_request = self.model.load_lora(lora_path)

        output = self.model.fast_generate(
            [prompt],
            sampling_params=sampling_params,
            lora_request=lora_request
        )[0].outputs[0].text

        return output