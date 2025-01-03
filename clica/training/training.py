from typing import Any, Dict, List, Optional, Sequence, Tuple
from dataclasses import dataclass

from datasets import Dataset
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import PreTrainedTokenizer, Trainer, TrainingArguments

from clica.code_env import COMMAND_TOKENS, InteractivePythonEnv
from clica.database import ActionType
from clica.training.trainer import MultiSequenceDataloader, MultiSequenceDataset


IGNORE_INDEX = -100
EOS_TOKENS = {'<|EOS|>', '<|end_of_sentence|>', '<|end_of_text|>'}

DEFAULT_TRAIN_CONFIG = {
    'n_train_epochs': 20,
    'batch_size': 4,
    'logging_steps': 10,
    'gradient_accumulation_steps': 1,
    'learning_rate': 5e-5,
    'max_grad_norm': 1.0,
}


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""
    tokenizer: PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple(
            [instance[key] for instance in instances] for key in ('input_ids', 'labels'))

        input_ids = [torch.tensor(x, dtype=torch.long) for x in input_ids]
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)

        labels = [torch.tensor(x, dtype=torch.long) for x in labels]
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)

        return dict(
            input_ids = input_ids,
            labels = labels,
            attention_mask = input_ids.ne(self.tokenizer.pad_token_id), # TODO: This may be a problem because when pad = eos token, the final eos token may not be trained on. Check this.
        )


def create_session_dataset(session: Dict[str, Any], tokenizer: PreTrainedTokenizer):
    """
    Creates a dataset from a single session.

    Args:
        session: A dictionary containing session data.
        tokenizer: The tokenizer to use.

    Returns:
        A list of dictionaries containing input_ids and labels for each step in the session.
    """
    non_text_tokens = COMMAND_TOKENS.union({tokenizer.eos_token})
    
    # First pass: Combine consecutive input actions
    combined_actions = []
    was_last_action_text = False
    for action_id, action, action_type, correct in session['actions']:
        if action in EOS_TOKENS:
            action = tokenizer.eos_token
        
        if correct == False:
            continue
        
        is_action_text = action_type == ActionType.INPUT.value and action not in non_text_tokens

        # Combine consecutive input actions if the previous one was text
        if combined_actions and action_type == ActionType.INPUT.value and was_last_action_text:
            combined_actions[-1] = (action_id, combined_actions[-1][1] + action, action_type, None)
        else:
            combined_actions.append((action_id, action, action_type, correct))

        was_last_action_text = is_action_text

    # Second pass: Create input_ids and labels
    env = InteractivePythonEnv(tokenizer)
    env.set_instruction(tokenizer.encode(session['initial_instruction'], add_special_tokens=False))
    env.set_code(tokenizer.encode(session['initial_code'], add_special_tokens=False))
    env.set_exec_output(tokenizer.encode(session['initial_exec_output'], add_special_tokens=False))

    sequence_data = []

    for action_id, action, action_type, correct in combined_actions:
        obs = env.get_obs()
        action_token_ids = tokenizer.encode(action, add_special_tokens=False)

        if action_type == ActionType.INPUT.value:
            input_ids = obs + action_token_ids
            labels = [IGNORE_INDEX] * len(obs) + action_token_ids
            sequence_data.append({
                'input_ids': input_ids,
                'labels': labels,
            })
            
            for token_id in action_token_ids:
                env.step(token_id)
        else:
            if action_type == ActionType.SET_INSTRUCTION.value:
                env.set_instruction(action_token_ids)
            elif action_type == ActionType.SET_CODE.value:
                env.set_code(action_token_ids)
            elif action_type == ActionType.SET_EXEC_OUTPUT.value:
                env.set_exec_output(action_token_ids)

    return sequence_data


def calculate_loss(logits, labels):
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index = IGNORE_INDEX,
    )
    return loss


def train_on_sessions(
    model: nn.Module,
    tokenizer: PreTrainedTokenizer,
    sessions: List[Dict[str, Any]],
    train_config: Dict[str, Any],
    optimizer: Optional[torch.optim.Optimizer] = None,
):
    """
    Trains the agent on multiple sequences of actions from different sessions.

    Args:
        model: The model to train.
        tokenizer: The tokenizer to use.
        sessions: A list of dictionaries, each containing session data.
        train_config: A dictionary containing training configuration.
    """
    def get_config_value(key):
        return train_config.get(key, DEFAULT_TRAIN_CONFIG[key])
    
    device = next(model.parameters()).device
    
    all_sequences = [create_session_dataset(session, tokenizer) for session in sessions.values()]
    max_sequence_length = max([len(seq) for seq in all_sequences])
    dataset = MultiSequenceDataset.from_nested_list(all_sequences)

    dataloader = MultiSequenceDataloader(
        dataset,
        batch_size = get_config_value('batch_size'),
        shuffle = True,
        collate_fn = DataCollatorForSupervisedDataset(tokenizer),
    )

    n_train_epochs = get_config_value('n_train_epochs')
    batch_size = get_config_value('batch_size')
    logging_steps = get_config_value('logging_steps')
    gradient_accumulation_steps = get_config_value('gradient_accumulation_steps')
    max_grad_norm = get_config_value('max_grad_norm')
    
    n_samples = sum([len(seq) for seq in all_sequences])
    max_batches_per_epoch = n_samples // batch_size + max_sequence_length
    # max_updates_per_epoch = max_batches_per_epoch // gradient_accumulation_steps

    optimizer = optimizer or torch.optim.Adam(
        model.parameters(),
        lr = get_config_value('learning_rate'),
    )

    recurrent_states = [None for _ in range(batch_size)]
    losses = []

    curr_iter = 0
    for epoch_idx in range(n_train_epochs):
        for batch_idx, train_batch in enumerate(dataloader):
            # TODO: Don't forget to use the recurrent states here in the future
            train_batch = {k: v.to(device) for k, v in train_batch.items()}
            outputs = model(input_ids=train_batch['input_ids'], attention_mask=train_batch['attention_mask'])

            loss = calculate_loss(outputs.logits, train_batch['labels'])
            loss = loss / gradient_accumulation_steps
            loss.backward()
            losses.append(loss.item())
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

            curr_iter += 1
            if curr_iter % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            if curr_iter % logging_steps == 0:
                current_epoch = epoch_idx + (batch_idx + 1) / max_batches_per_epoch
                avg_loss = sum(losses) / len(losses)
                print(f'Epoch {current_epoch:.2f}, Iter {curr_iter}: Training loss: {avg_loss:.4f}')
                losses = []
        
        # Step the optimizer after the epoch if the last step was not a gradient accumulation step
        if curr_iter % gradient_accumulation_steps != 0:
            optimizer.step()
            optimizer.zero_grad()

    if curr_iter % logging_steps != 0:
        avg_loss = sum(losses) / len(losses)
        print(f'Epoch {n_train_epochs}, Iter {curr_iter}: Training loss: {avg_loss:.4f}')
        losses = []

    torch.cuda.empty_cache()

    return model, optimizer


def train_on_actions(model, tokenizer, env, actions: List[Tuple[int, str, str, Optional[bool]]]):
    """
    Trains the agent on a list of actions using the HuggingFace Trainer.

    Args:
        model: The model to train.
        tokenizer: The tokenizer to use.
        env: The environment to use for generating observations.
        actions: A list of tuples containing (action_id, action, action_type, correct).
    """
    train_data = []

    for action_idx, action, action_type, correct in actions:
        obs = env.get_obs()
        action_token_ids = tokenizer.encode(action, add_special_tokens=False)

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
        model = model,
        args = training_args,
        train_dataset = dataset,
        data_collator = DataCollatorForSupervisedDataset(tokenizer),
    )

    # Set up logging to file
    trainer.train()

    # Update the model reference
    model = trainer.model
    
    del trainer
    torch.cuda.empty_cache()
    
    return model, env
