# Advanced GRPO and DeepSeek LLM Underlying Technology



GRPO & DeepSeek: Innovations in

LLM Training and Architecture

Making AI Training More Efficient and Accessible

## What is GRPO?

Group Relative Policy Optimization (GRPO)

- Definition:  A new reinforcement learning algorithm that makes LLM training 50% more

- memory efficient

- Created by:  DeepSeek team in 2024

- Purpose:  Train AI models to reason better (especially in math) without needing massive

### computing resources

- Key Innovation:  Eliminates the need for a "critic" model by using group-based comparisons

### GRPO vs PPO

How GRPO Simplifies Reinforcement Learning

### Traditional PPO:
- Needs 3 models: Actor + Critic + Reference
- High memory usage (2x the main model size)
- Complex training pipeline
- Requires separate value estimation

### GRPO Innovation:

- Needs only 2 models: Actor + Reference
- 50% less memory required

- Simpler, more stable training

- Uses group statistics instead of value

### How GRPO Works - Technical Overview

GRPO in 4 Simple Steps

Process Flow:

1.Generate  - Create 64 responses for each

question

2.Score  - Reward model rates each response

3.Compare  - Calculate which responses are above/below average

○Formula (simplified): Advantage = (score - average) / spread

4.Update  - Increase probability of good responses, decrease bad ones

### Real Results Box:

- DeepSeekMath improvement: 82.9% → 88.2% on GSM8K

- Enabled training on single GPU vs multiple GPUs for PPO

- Now integrated into DeepSeek-R1's training

### DeepSeek Architecture Overview

- DeepSeek: Smart Architecture Beats Brute Force

### Key Components:

- 671 Billion Parameters  - But only uses 37B at a time!

- Mixture of Experts (MoE):  Like having 256 specialist consultants, using only the relevant ones
- Multi-head Latent Attention (MLA):  Compresses memory by 90% while maintaining performance
- Cost to Train:  Only $5.58 million (vs competitors' $100M+)

#### Innovation Highlight:  "Activates different 'expert' networks based on the

#### question - like consulting different professors for different subjects"


How DeepSeek Achieved 90% Memory Reduction

Multi-head Latent Attention (MLA) vs Traditional Multi-head Self Attention

Traditional Multi-head Self Attention:

How it Works:

● Each head stores separate K, V matrices

● Memory: O(n × d × h) where h = number of heads

● Example: 128 heads × 128 dims = 16,384 parameters per token

Problems:

● 🔴

Massive memory usage (KV cache explosion)

● 🔴

Redundant information across heads

● 🔴

Limits model scaling

Multi-head Latent Attention (MLA):

Innovation:

● Compress K, V into shared latent vectors

● All heads share compressed representation

● Memory: O(n × c) where c << d × h

Benefits:

● ✅

90% memory reduction

● ✅

Same performance

● ✅

Enables larger models on same hardware


DeepSeek-R1 Training Pipeline

Revolutionary Training: From Zero to Hero

Two-Stage Process:

Stage 1: R1-Zero (Pure RL)

●First model to learn reasoning using ONLY reinforcement learning

●No human-written examples needed!

●Naturally developed:

○Self-verification  ("Let me check that...")

○Chain-of-thought reasoning

○"Aha moments" - self-correction

Stage 2: R1 (Refined)

1.Cold start with 800K examples  ArXiv

2.Large-scale RL training

3.Generate high-quality reasoning data

4.Final alignment for helpfulness

Advanced GRPO and DeepSeek LLM Underlying Technology

DeepSeek's Coding Prowess

Performance Metrics:

●Codeforces Rating:  2029 (96.3 percentile - better than most human

programmers!)

●Programming Languages:  338+ supported

●Debugging Accuracy:  90% (vs GPT-o1's 80%)

●Cost:  15x cheaper than OpenAI o1

Task DeepSeek-R1 OpenAI o1 Claude 3.5

LiveCodeBench 65.9%  Bind AI IDE - 65.0%  Bind AI IDE

SWE-bench 49.2%  Bind AI IDE 48.9%  Bind AI IDE 50.8%  Bind AI IDE

