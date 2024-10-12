from io import StringIO
from contextlib import redirect_stdout, redirect_stderr
from typing import Dict, List, Tuple
import sys
import warnings
import traceback

import gymnasium as gym
import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, PreTrainedTokenizer, PreTrainedModel


def format_command_token(key):
    return COMMAND_OPENER + key + COMMAND_CLOSER
  
def format_special_token(key):
    return SPECIAL_TOKEN_OPENER + key + SPECIAL_TOKEN_CLOSER


COMMAND_OPENER = '<｜'
COMMAND_CLOSER = '｜>'
SPECIAL_TOKEN_OPENER = '<|'
SPECIAL_TOKEN_CLOSER = '|>'

# Command special tokens
KEY_BACKSPACE_TOKEN = format_command_token('key_backspace')
KEY_ENTER_TOKEN = format_command_token('key_enter')
KEY_LEFT_TOKEN = format_command_token('key_left')
KEY_RIGHT_TOKEN = format_command_token('key_right')
KEY_UP_TOKEN = format_command_token('key_up')
KEY_DOWN_TOKEN = format_command_token('key_down')
CURSOR_TOKEN = format_command_token('cursor')
RUN_CODE_TOKEN = format_command_token('run_code')

COMMAND_TOKENS = set([
  KEY_BACKSPACE_TOKEN,
  KEY_ENTER_TOKEN,
  KEY_LEFT_TOKEN,
  KEY_RIGHT_TOKEN,
  KEY_UP_TOKEN,
  KEY_DOWN_TOKEN,
  RUN_CODE_TOKEN,
])

# Header special tokens
HEADER_INSTRUCTION_TOKEN = format_special_token('instruction')
HEADER_CODE_TOKEN = format_special_token('code')
HEADER_EXEC_OUTPUT_TOKEN = format_special_token('execution_output')
HEADER_TEXT_QUEUE_TOKEN = format_special_token('text_queue')

HEADER_TOKENS = set([
  HEADER_INSTRUCTION_TOKEN,
  HEADER_CODE_TOKEN,
  HEADER_EXEC_OUTPUT_TOKEN,
  HEADER_TEXT_QUEUE_TOKEN,
])

ENV_SPECIAL_TOKENS = COMMAND_TOKENS.union(HEADER_TOKENS).union(set([CURSOR_TOKEN]))


class InteractivePythonEnv(gym.Env):
  """
  A gym-like environment that allows for interaction with a Python script.
  All observations are lists of tokens, and all actions are single tokens.
  
  The agent's observations are split into:
    - The most recent instruction prompt
    - The most recent output of executing the code
    - The current code
    - Current action queue
    
  Actions are split into categories:
    - Commands: move the cursor around, run the code, backspace, etc.
    - Text input: insert text into the code
    
  When taking a text action, new text tokens are appended to the action queue.
  Text will continue to accumulate until the ENTER command is received, at which point
  the text will be inserted into the code document. Any other command attempted while
  the text is still queued will be ignored.
  """

  metadata = {'render.modes': ['human']}

  def __init__(self, tokenizer: PreTrainedTokenizer, vocab: Dict[str, int]):
    super(InteractivePythonEnv, self).__init__()

    self._tokenizer = tokenizer
    self._vocab = vocab # Maps token string to token id
    
    self._check_command_tokens_in_vocab()
    # Maps command token id to the token string
    self._rev_vocab = {v: k for k, v in self._vocab.items()}
    self._command_token_ids = set([self._vocab[token] for token in COMMAND_TOKENS])
    self._new_line_ids = set(self._get_new_line_ids())
    
    self.reset()

  def reset(self) -> List[int]:
    """
    Resets the environment to its initial state.

    Returns:
      str: The current text observation.
    """
    self._instruction = []
    self._code = []
    self._exec_output = []
    self._text_queue = []
    self._cursor_pos = 0

    return self.get_obs()

  def _check_command_tokens_in_vocab(self):
    """
    Checks if all command tokens are in the vocab.
    """
    for token in COMMAND_TOKENS:
      assert token in self._vocab, f'Command token not in vocab: {token}'

  def step(self, action: int) -> Tuple[List[int], float, bool, dict]:
    """
    Interprets the token and it applies an appropriate change to the environment,
    then returns the new observation, reward, termination flag, and extra info.

    Args:
      action (int): The token (command or text) action to take in the environment.

    Returns:
      tuple: A 4-element tuple containing:
        - The new observation
        - The reward
        - The termination flag
        - The extra info
    """
    
    if action in self._command_token_ids:
      self._apply_command(action)
    else:
      self._text_queue.append(action)

    return self.get_obs(), 0, False, {}
    
  def _get_new_line_ids(self):
    """Returns the token ids for all tokens that contain a newline character."""
    ids = [i for token, i in self._vocab.items() if '\n' in token]
    return ids

  def _get_previous_new_line_idx(self):
    """Returns the index of the last newline character prior to the cursor."""
    for i in range(self._cursor_pos - 1, -1, -1):
      if self._code[i] in self._new_line_ids:
        return i
    return -1
  
  def _get_next_new_line_idx(self):
    """Returns the index of the next newline character after the cursor."""
    for i in range(self._cursor_pos, len(self._code)):
      if self._code[i] in self._new_line_ids:
        return i
    return -1

  def _apply_command(self, command_token_id: int):
    """
    Applies an action to the environment.

    Args:
      command_token_id (int): The id of the command token to apply.
    """
    command_token = self._rev_vocab[command_token_id]
      
    if command_token == KEY_ENTER_TOKEN:
      self._code = self._code[:self._cursor_pos] + self._text_queue + self._code[self._cursor_pos:]
      self._cursor_pos += len(self._text_queue)
      self._text_queue = []
      
    elif len(self._text_queue) > 0:
      pass
    
    elif command_token == KEY_LEFT_TOKEN:
      self._cursor_pos = max(0, self._cursor_pos - 1)
    
    elif command_token == KEY_RIGHT_TOKEN:
      self._cursor_pos = min(len(self._code), self._cursor_pos + 1)
      
    elif command_token == KEY_UP_TOKEN:
      new_line_idx = self._get_previous_new_line_idx()
      self._cursor_pos = new_line_idx if new_line_idx != -1 else 0
    
    elif command_token == KEY_DOWN_TOKEN:
      new_line_idx = self._get_next_new_line_idx()
      self._cursor_pos = new_line_idx if new_line_idx != -1 else len(self._code)
    
    elif command_token == KEY_BACKSPACE_TOKEN:
      if self._cursor_pos > 0:
        self._code = self._code[:self._cursor_pos - 1] + self._code[self._cursor_pos:]
        self._cursor_pos -= 1
    
    elif command_token == RUN_CODE_TOKEN:
      self._execute_code()
      
    else:
      raise ValueError(f'Invalid command token: {command_token}')

  def _execute_code(self):
    """
    Executes the code in the code buffer and captures both stdout and the full stack trace if an error occurs.
    """
    code_text = self._tokenizer.decode(self._code, skip_special_tokens=True)
    
    stdout_capture = StringIO()
    error_trace = ''
    
    with redirect_stdout(stdout_capture):
        try:
            exec(code_text)
        except Exception:
            error_trace = traceback.format_exc()
    
    output = stdout_capture.getvalue() + error_trace
    self._exec_output = self._tokenizer.encode(output)

  def set_instruction(self, instruction: List[int]):
    """
    Sets the instruction to the given list of token ids.
    """
    self._instruction = instruction

  def set_code(self, code: List[int]):
    """
    Sets the code to the given list of token ids.
    """
    self._code = code

  def set_exec_output(self, exec_output: List[int]):
    """
    Sets the execution output to the given list of token ids.
    """
    self._exec_output = exec_output

  def get_obs(self, include_cursor: bool = True) -> List[int]:
    """
    Get the observation, which includes the current instruction, most recent execution output, 
    the current code, and the current text queue.
    """
    code = self._code
    if include_cursor:
      code = code[:self._cursor_pos] + [self._vocab[CURSOR_TOKEN]] + code[self._cursor_pos:]
      
    obs_token_ids = [self._vocab[HEADER_INSTRUCTION_TOKEN]] + self._instruction \
        + [self._vocab[HEADER_EXEC_OUTPUT_TOKEN]] + self._exec_output \
        + [self._vocab[HEADER_CODE_TOKEN]] + code \
        + [self._vocab[HEADER_TEXT_QUEUE_TOKEN]] + self._text_queue
  
    return obs_token_ids
  
  def get_dict_obs(self, include_cursor: bool = True):
    """
    Get the observation as a dictionary, with keys for:
      the instruction, code, execution output, and text queue.
    """
    code = self._code
    if include_cursor:
      code = code[:self._cursor_pos] + [self._vocab[CURSOR_TOKEN]] + code[self._cursor_pos:]
    
    return {
      'instruction': self._instruction,
      'code': code,
      'exec_output': self._exec_output,
      'text_queue': self._text_queue,
    }
    
  def render(self, mode: str = 'human'):
    """
    Renders the current terminal screen.

    Args:
      mode (str): The mode to use for rendering. Currently only 'human' is supported.
    """
    if mode != 'human':
      raise ValueError(f'Invalid mode: {mode}')

    print(self.get_obs())
    
  def set_instruction(self, instruction: List[int]):
    """
    Sets the instruction to the given list of token ids.
    """
    self._instruction = instruction


def add_and_init_special_token(token: str, tokenizer: PreTrainedTokenizer, model: PreTrainedModel = None):
  """Add a custom token to the tokenizer and model."""
  
  # Add the new token to the tokenizer
  
  if model is not None:
    # Tokenize the components of the custom token
    components = tokenizer.tokenize(token)  # Remove <|...|>
    
    # Get the embeddings of the component tokens
    with torch.no_grad():
      component_ids = tokenizer.convert_tokens_to_ids(components)
      component_embeddings = model.get_input_embeddings()(torch.tensor(component_ids))
    
    # Calculate the sum of component embeddings
    new_embedding = component_embeddings.mean(dim=0)
    
    # Normalize the new embedding to match the expected magnitude
    with torch.no_grad():
      existing_embeddings = model.get_input_embeddings().weight.data
      avg_norm = existing_embeddings.norm(dim=1).mean()
      new_embedding = F.normalize(new_embedding, dim=0) * avg_norm

  num_added_tokens = tokenizer.add_special_tokens({"additional_special_tokens": [token]})
  
  if model is not None:
    # Resize the model's token embeddings
    model.resize_token_embeddings(len(tokenizer))
    
    # Set the embedding for the new token
    model.get_input_embeddings().weight.data[-1] = new_embedding

  return num_added_tokens


### Some simple tests ###

if __name__ == '__main__':
    from transformers import AutoTokenizer
    # Initialize tokenizer and vocab
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    # Add special tokens
    for token in ENV_SPECIAL_TOKENS:
      add_and_init_special_token(token, tokenizer, model)
      
    vocab = tokenizer.get_vocab()

    # Create environment
    env = InteractivePythonEnv(tokenizer, vocab)

    # Test reset
    obs = env.reset()
    print("Reset observation:")
    print(tokenizer.decode(obs))
    print("\n" + "-" * 50 + "\n")

    # Test adding some code
    code = "print('Hello, World!')"
    for token_id in tokenizer.encode(code):
      print('Taking action:', tokenizer.decode([token_id]))
      obs, _, _, _ = env.step(token_id)
    
    print("Observation after adding code:")
    print(tokenizer.decode(obs))
    print("\n" + "-" * 50 + "\n")
    
    obs, _, _, _ = env.step(vocab[KEY_ENTER_TOKEN])
    print("Observation after entering code:")
    print(tokenizer.decode(obs))
    print("\n" + "-" * 50 + "\n")

    # Test running code
    run_token = vocab[RUN_CODE_TOKEN]
    obs, _, _, _ = env.step(run_token)
    print("Observation after running code:")
    print(tokenizer.decode(obs))
    print("\n" + "-" * 50 + "\n")

    # Test cursor movement
    left_token = vocab[KEY_LEFT_TOKEN]
    obs, _, _, _ = env.step(left_token)
    print("Observation after moving cursor left:")
    print(tokenizer.decode(obs))
    print("\n" + "-" * 50 + "\n")

    # Test backspace
    backspace_token = vocab[KEY_BACKSPACE_TOKEN]
    obs, _, _, _ = env.step(backspace_token)
    print("Observation after backspace:")
    print(tokenizer.decode(obs))
    print("\n" + "-" * 50 + "\n")

    # Test adding text to queue
    new_text = "Python"
    for char in new_text:
        token = tokenizer.encode(char)[0]
        obs, _, _, _ = env.step(token)
    print("Observation after adding text to queue:")
    print(tokenizer.decode(obs))
    print("\n" + "-" * 50 + "\n")

    # Test entering queued text
    enter_token = vocab[KEY_ENTER_TOKEN]
    obs, _, _, _ = env.step(enter_token)
    print("Observation after entering queued text:")
    print(tokenizer.decode(obs))
    print("\n" + "-" * 50 + "\n")

    # Run the modified code
    obs, _, _, _ = env.step(run_token)
    print("Final observation after running modified code:")
    print(tokenizer.decode(obs))
    print("\n" + "-" * 50 + "\n")