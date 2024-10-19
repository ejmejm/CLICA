import os
import logging
import random
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from datasets import Dataset
import torch
from torch import nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)

from clica.code_env import InteractivePythonEnv, COMMAND_TOKENS
from clica.training.training import train_on_sessions, train_on_actions


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

    def get_action(self, state: RecurrentState, obs: List[int]) -> Tuple[RecurrentState, int]:
        """Returns the model's action given the observation string.

        Returns:
            The model's action.
        """
        raise NotImplementedError
    
    def execute_action_loop(self, state: RecurrentState, env: InteractivePythonEnv) -> List[int]:
        """Enacts actions in the environment until the EOS token is generated or the max_gen_length is reached.
        
        Returns:
            The list of actions taken.
        """
        actions = []
        for _ in range(self.max_gen_length):
            state, action = self.get_action(state, env.get_obs())
            env.step(action)
            actions.append(action)
            if action == self.tokenizer.eos_token:
                break
        return actions

    def save(self, path: str):
        """Save the state of the agent."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def load(self, path: str):
        """Load the state of the agent."""
        self.model = AutoModelForCausalLM.from_pretrained(path)
        self.tokenizer = AutoTokenizer.from_pretrained(path)

    def train_on_actions(self, env, actions: List[Tuple[int, str, str]]):
        """
        Trains the agent on a list of actions.

        Args:
            env: The environment to use for generating observations.
            actions: A list of tuples containing (action_id, action, action_type).
        """
        raise NotImplementedError

    def train_on_sessions(self, sessions: List[Dict[str, Any]]):
        """
        Trains the agent on multiple sequences of actions from different sessions.

        Args:
            sessions: A list of dictionaries, each containing session data.
        """
        raise NotImplementedError


class DummyAgent(BaseAgent):
    """Dummy agent that returns random actions."""

    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, max_gen_length: int = 128):
        """Initialize the agent."""
        super().__init__(model, tokenizer, max_gen_length)

    def get_action(self, state: RecurrentState, obs: List[int]) -> Tuple[RecurrentState, int]:
        """Returns a random action."""
        return (None, random.randint(0, len(self.tokenizer) - 1))
    
    def eval(self, path: str) -> Dict[str, Any]:
        """Dummy eval function."""
        print('Example print statement 1')
        time.sleep(0.4)
        print('Example print statement 2')
        time.sleep(1)
        return {'example_key': 1/3, 'example_key_2': 'example_value'}


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

    def train_on_sessions(self, sessions: List[Dict[str, Any]]):
        """
        Trains the agent on multiple sequences of actions from different sessions.

        Args:
            sessions: A list of dictionaries, each containing session data.
        """
        self.model = train_on_sessions(self.model, self.tokenizer, sessions)

    def train_on_actions(self, env, actions: List[Tuple[int, str, str, Optional[bool]]]):
        """
        Trains the agent on a list of actions using the HuggingFace Trainer.

        Args:
            env: The environment to use for generating observations.
            actions: A list of tuples containing (action_id, action, action_type, correct).
        """
        self.model, env = train_on_actions(self.model, self.tokenizer, env, actions)
        return env


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
