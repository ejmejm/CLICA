import os
import logging
import random
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import torch
from torch import nn
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)

from code_env import COMMAND_TOKENS


IGNORE_INDEX = -100


RecurrentState = Any


logger = logging.getLogger(__name__)


class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_tokens: Set[int]):
        self.stop_tokens = stop_tokens

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor, **kwargs) -> bool:
        if input_ids[0][-1] in self.stop_tokens:
            return True
        return False


class BaseAgent(nn.Module):
    """Base class for agents working with text-based environments."""

    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, max_gen_length: int = 128):
        """Initialize the agent."""
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.max_gen_length = max_gen_length

    @property
    def eos_token(self) -> int:
        """Returns the EOS token id."""
        return self.tokenizer.eos_token

    def get_action(self, *args) -> Tuple[RecurrentState, int]:
        """Returns the model's action given the observation string.

        Returns:
            The model's action.
        """
        raise NotImplementedError

    def train_supervised(self, buffer: Tuple[List[int], List[int]]) -> float:
        """Trains the model on the given observations and actions.

        Args:
            buffer: A dictionary containing the observations and actions.

        Returns:
            The loss of the training step.
        """
        raise NotImplementedError

    def save_state(self, path: str):
        """Save the state of the agent."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Save as pretrained model
        self.model.save_pretrained(path)


class DummyAgent(BaseAgent):
    """Dummy agent that returns random actions."""

    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, max_gen_length: int = 128):
        """Initialize the agent."""
        super().__init__(model, tokenizer, max_gen_length)

    def get_action(self, state: RecurrentState, obs: List[int]) -> Tuple[RecurrentState, int]:
        """Returns a random action."""
        return (None, random.randint(0, len(self.tokenizer) - 1))


class TransformerAgent(BaseAgent):
    """Agent that uses a transformer model with limited context to generate actions."""

    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, max_gen_length: int = 128):
        """Initialize the agent."""
        super().__init__(model, tokenizer, max_gen_length)
        
        self.stop_token_ids = set([
            self.tokenizer.encode(token)[0]
            for token in COMMAND_TOKENS.union({self.tokenizer.eos_token})
        ])
        self.stopping_criteria = StoppingCriteriaList([StopOnTokens(self.stop_token_ids)])
        
        self.action_queue = []

    def get_action(self, state: RecurrentState, obs: List[int]) -> Tuple[RecurrentState, int]:
        """Returns a random action."""
        if len(self.action_queue) == 0:
            input_ids = torch.tensor(obs).to(self.model.device)

            # TODO: Add kv caching
            outputs = self.model.generate(
                input_ids = input_ids.unsqueeze(0),
                max_new_tokens = self.max_gen_length,
                temperature = 0.7,
                do_sample = True,
                top_k = 5,
                stopping_criteria = self.stopping_criteria,
            )

            self.action_queue.extend(outputs[0].tolist()[len(obs):])
        
        action = self.action_queue.pop(0)
        return (None, action)
    
    def reset_action_queue(self):
        self.action_queue = []


if __name__ == '__main__':
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Initialize a dummy model and tokenizer
    model = AutoModelForCausalLM.from_pretrained('gpt2')
    tokenizer = AutoTokenizer.from_pretrained('gpt2')

    # Create a DummyAgent
    dummy_agent = DummyAgent(model, tokenizer)

    # Test get_action
    obs = [1, 2, 3, 4, 5]  # Dummy observation
    action = dummy_agent.get_action(obs)
    print(f"Random action: {action}")
    assert 0 <= action < len(tokenizer), "Action should be within tokenizer range"

    # Test multiple calls to get_action
    actions = [dummy_agent.get_action(obs) for _ in range(10)]
    print(f"10 random actions: {actions}")
    assert len(set(actions)) > 1, "Multiple calls should produce different actions"
