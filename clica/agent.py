from dataclasses import dataclass
import os
import logging
import random
import time
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import torch
from torch import nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
    TrainingArguments,
    Trainer,
)
from datasets import Dataset

from clica.code_env import InteractivePythonEnv, COMMAND_TOKENS
from clica.database import ActionType


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


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""
    tokenizer: PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple(
            [instance[key] for instance in instances] for key in ('input_ids', 'labels'))
        input_ids = [torch.tensor(x) for x in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)

        labels = [torch.tensor(x) for x in labels]
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX)

        return dict(
            input_ids = input_ids,
            labels = labels,
            attention_mask = input_ids.ne(self.tokenizer.pad_token_id), # TODO: Is this not also checking for IGNORE_INDEX a bug?
        )


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

    def train_supervised(self, buffer: Tuple[List[int], List[int]]) -> float:
        """Trains the model on the given observations and actions.

        Args:
            buffer: A dictionary containing the observations and actions.

        Returns:
            The loss of the training step.
        """
        raise NotImplementedError

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

    def train_on_actions(self, env, actions: List[Tuple[int, str, str, Optional[bool]]]):
        """
        Trains the agent on a list of actions using the HuggingFace Trainer.

        Args:
            env: The environment to use for generating observations.
            actions: A list of tuples containing (action_id, action, action_type, correct).
        """
        train_data = []

        for action_idx, action, action_type, correct in actions:
            obs = env.get_obs()
            action_token_ids = self.tokenizer.encode(action, add_special_tokens=False)

            # Handle setting the instruction, code, exec output actions
            if action_type == ActionType.SET_INSTRUCTION.value:
                env.set_instruction(action_token_ids)
            elif action_type == ActionType.SET_CODE.value:
                env.set_code(action_token_ids)
            elif action_type == ActionType.SET_EXEC_OUTPUT.value:
                env.set_exec_output(action_token_ids)
            
            # Handle input actions
            elif action_type == ActionType.INPUT.value:
                # Add to training data if not incorrect
                if correct != False:
                    input_ids = obs + action_token_ids
                    labels = [IGNORE_INDEX] * len(obs) + action_token_ids

                    train_data.append({
                        "input_ids": input_ids,
                        "labels": labels,
                    })
                
                # Enact each action in the environment
                for token_id in action_token_ids:
                    env.step(token_id)

        # Create a Dataset from the training data
        dataset = Dataset.from_list(train_data)

        # Define training arguments
        training_args = TrainingArguments(
            output_dir = './results',
            num_train_epochs = 2,
            per_device_train_batch_size = 4,
            logging_dir = './logs',
            logging_steps = 10,
        )

        # Create and run the Trainer
        trainer = Trainer(
            model = self.model,
            args = training_args,
            train_dataset = dataset,
            data_collator = DataCollatorForSupervisedDataset(self.tokenizer),
        )

        # Set up logging to file
        trainer.train()

        # Update the model reference
        self.model = trainer.model
        
        del trainer
        torch.cuda.empty_cache()
        
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
