from typing import Any, Dict, List, Optional, Sequence, Tuple
from dataclasses import dataclass

from accelerate import Accelerator
from datasets import Dataset, IterableDataset
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import PreTrainedTokenizer, Trainer, TrainingArguments

from clica.code_env import InteractivePythonEnv
from clica.database import ActionType
from clica.training.trainer import SequentialDataloader


IGNORE_INDEX = -100


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""
    tokenizer: PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple(
            [instance[key] for instance in instances] for key in ('input_ids', 'labels'))

        input_ids = [torch.tensor(x) for x in input_ids]
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)

        labels = [torch.tensor(x) for x in labels]
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)

        return dict(
            input_ids = input_ids,
            labels = labels,
            attention_mask = input_ids.ne(self.tokenizer.pad_token_id), # TODO: Is this not also checking for IGNORE_INDEX a bug?
        )


def create_session_dataset(session, tokenizer):
    """
    Creates a dataset from a single session.

    Args:
        session: A dictionary containing session data.
        tokenizer: The tokenizer to use.

    Returns:
        A list of dictionaries containing input_ids and labels for each step in the session.
    """
    # First pass: Combine consecutive input actions
    combined_actions = []
    for action_id, action, action_type, correct in session['actions']:
        if action_type == ActionType.INPUT.value:
            if correct == False:
                continue
            
            if combined_actions and combined_actions[-1][2] == ActionType.INPUT.value:
                combined_actions[-1] = (action_id, combined_actions[-1][1] + action, ActionType.INPUT.value, None)
            else:
                combined_actions.append((action_id, action, ActionType.INPUT.value, None))
        else:
            combined_actions.append((action_id, action, action_type, correct))

    # Second pass: Create input_ids and labels
    env = InteractivePythonEnv()
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
):
    """
    Trains the agent on multiple sequences of actions from different sessions.

    Args:
        model: The model to train.
        tokenizer: The tokenizer to use.
        sessions: A list of dictionaries, each containing session data.
    """
    all_sequences = [create_session_dataset(session, tokenizer) for session in sessions]
    dataset = Dataset.from_list(all_sequences)
    dataset = IterableDataset(dataset)

    dataloader = SequentialDataloader(
        dataset,
        batch_size = 4,
        shuffle = True,
        collate_fn = DataCollatorForSupervisedDataset(tokenizer),
    )

    n_train_epochs = 2
    batch_size = 4
    logging_steps = 10
    gradient_accumulation_steps = 1
    
    n_samples = sum([len(seq) for seq in all_sequences])
    batches_per_epoch = n_samples // batch_size

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr = 5e-5,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=n_train_epochs * batches_per_epoch)

    accelerator = Accelerator(
        gradient_accumulation_steps = gradient_accumulation_steps,
    )
    dataloader, model, optimizer, scheduler = accelerator.prepare(
        dataloader, model, optimizer, scheduler)
    
    recurrent_states = [None for _ in range(batch_size)]

    curr_iter = 0
    for epoch_idx in range(n_train_epochs):
        for batch_idx, train_batch in enumerate(dataloader):
            with accelerator.accumulate(model):
                # TODO: Don't forget to use the recurrent states here in the future
                outputs = model(input_ids=train_batch['input_ids'], attention_mask=train_batch['attention_mask'])
                loss = calculate_loss(outputs.logits, train_batch['labels'])
                
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
            curr_iter += 1

            if batch_idx % logging_steps == 0:
                # total_batches = n_train_epochs * batches_per_epoch
                current_epoch = epoch_idx + (batch_idx + 1) / batches_per_epoch
                print(f'Epoch {current_epoch:.2f}, Iter {curr_iter}: Training loss: {loss.item():.4f}')

    torch.cuda.empty_cache()

    return model


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
