import difflib
from typing import Optional, List

import litellm
from transformers import PreTrainedTokenizer

from code_env import KEY_LEFT_TOKEN, KEY_RIGHT_TOKEN, KEY_BACKSPACE_TOKEN


def generate_solution(instruction: str, code: str = '', exec_output: str = '') -> str:
    """
    Generates a solution for the given instruction using LiteLLM.

    Args:
        instruction (str): The problem or task description.
        code (str, optional): Existing code, if any. Defaults to ''.
        exec_output (str, optional): Execution output of existing code, if any. Defaults to ''.

    Returns:
        str: Generated solution code.
    """
    prompt = create_solution_prompt(instruction, code, exec_output)
    
    response = litellm.completion(
        model='gpt-3.5-turbo',
        messages=[{'role': 'system', 'content': prompt}],
        temperature=0.7,
        max_tokens=1000,
    )

    response_text = response.choices[0].message.content.strip()
    response_code = extract_code(response_text)

    return response_code


def create_solution_prompt(instruction: str, code: str, exec_output: str) -> str:
    """
    Creates a prompt for generating a solution based on the given inputs.

    Args:
        instruction (str): The problem or task description.
        code (str): Existing code, if any.
        exec_output (str): Execution output of existing code, if any.

    Returns:
        str: Formatted prompt for solution generation.
    """
    prompt = (
        'Generate a Python solution for the following problem:\n\n'
        f'{instruction}\n\n'
    )

    if code:
        prompt += (
            '\n\nExisting code is provided below. Build upon it if possible, '
            f'but rewrite anything you use:\n\n```python\n{code}\n```'
        )

    if exec_output:
        prompt += (
            f'\n\nThe following output was produced when running the last version '
            f'of the code:\n\n```\n{exec_output}\n```'
        )

    prompt += (
        '\n\nProvide only the code for the solution, without any additional explanation. '
        'Ensure the solution is written in Python and includes all necessary '
        'code to solve the problem. Respond with only the code, and no other text.'
    )

    return prompt


def extract_code(string: str) -> str:
    """Extract code blocks from a string.

    Uses very simple heuristics to try to extract a block of code when
    a string may contain more than just code.
    This is really only necessary because we are working with pretty
    limited models for now so they don't always follow instructions
    perfectly.

    Args:
        string (str): The string to extract code blocks from.

    Returns:
        str: The extracted code block.
    """
    # First try to find a block that starts and ands with ```...``` or ```python...```
    start = string.find('```python\n')
    start = start if start == -1 else start + len('```python\n')
    if not (0 <= start <= 100):
        start = string.find('```\n')
        start = start if start == -1 else start + len('```\n')
    if 0 <= start <= 100:
        end = string.find('\n```', start + 1)
        end = end if end != -1 else len(string)
        return string[start:end]
    
    # If that fails, just return the whole string
    return string


def get_actions_from_diff(
        start_ids: List[int],
        end_ids: List[int],
        tokenizer: PreTrainedTokenizer,
        cursor_pos: Optional[int] = 0
    ) -> List[int]:
    """Determine which actions to take to transform input_ids into output_ids.
    
    Args:
        start_ids: A list of token IDs representing the initial code.
        end_ids: A list of token IDs representing the target code.
        tokenizer: A HuggingFace tokenizer.
        cursor_pos: The initial cursor position (default is 0).

    Returns:
        A list of token IDs representing the actions to take.
    """
    # Get token IDs for special tokens
    key_left_token_id = tokenizer.convert_tokens_to_ids(KEY_LEFT_TOKEN)
    key_right_token_id = tokenizer.convert_tokens_to_ids(KEY_RIGHT_TOKEN)
    backspace_token_id = tokenizer.convert_tokens_to_ids(KEY_BACKSPACE_TOKEN)
    enter_token_id = tokenizer.convert_tokens_to_ids(KEY_ENTER_TOKEN)

    # Use difflib to get the differences as operations
    s = difflib.SequenceMatcher(None, start_ids, end_ids)
    opcodes = s.get_opcodes()

    # Initialize the list of action ids
    action_ids = []
    current_pos = cursor_pos
    
    # Process each change
    for tag, i1, i2, j1, j2 in opcodes:
        current_pos = i1
        if tag == 'replace':
            # Move cursor to the position of change
            action_ids.extend([key_right_token_id] * (i2 - current_pos))
            # Perform deletions
            action_ids.extend([backspace_token_id] * (i2 - current_pos))
            # Add replacement tokens
            action_ids.extend(end_ids[j1:j2])
            action_ids.extend([enter_token_id])
            # current_pos = i2 # i2 + (j2 - j1)
        elif tag == 'delete':
            # Move cursor to the position of change
            action_ids.extend([key_right_token_id] * (i2 - current_pos))
            # Perform deletions
            action_ids.extend([backspace_token_id] * (i2 - current_pos))
            # current_pos = i1
        elif tag == 'insert':
            # Move cursor to the position of insertion
            action_ids.extend([key_right_token_id] * (i1 - current_pos))
            # Add new tokens
            action_ids.extend(end_ids[j1:j2])
            action_ids.extend([enter_token_id])
            # current_pos = i1
        elif tag == 'equal':
            # Move cursor to the end of the equal section
            action_ids.extend([key_right_token_id] * (i2 - current_pos))
            # current_pos = i2

    # Adjust the start of the sequence to work from the initial cursor position
    if cursor_pos > 0:
        # Count the number of initial right movements
        n_start_rights = 0
        for action in action_ids:
            if action == key_right_token_id:
                n_start_rights += 1
            else:
                break

        # Calculate the required movement
        movement = n_start_rights - cursor_pos

        if movement < 0:
            # Move left
            nav_tokens = [key_left_token_id] * -movement
        elif movement > 0:
            # Move right
            nav_tokens = [key_right_token_id] * movement
        else:
            nav_tokens = []

        # Replace the initial movements with the calculated nav_tokens
        action_ids = nav_tokens + action_ids[n_start_rights:]
        
    # Remove any trailing navigation tokens
    while action_ids and action_ids[-1] in [key_left_token_id, key_right_token_id]:
        action_ids.pop()

    return action_ids


if __name__ == '__main__':
    # Test the generate_solution function
    print('Testing solution generation:')

    # Test 1: Generate a solution for a simple problem
    print('\n1. Generating a solution for a simple problem:')
    instruction = 'Write a function that calculates the factorial of a given number.'
    solution = generate_solution(instruction)
    print(solution)

    # Test 2: Generate a solution with existing code
    print('\n2. Generating a solution with existing code:')
    instruction = 'Modify the factorial function to use iteration instead of recursion.'
    code = 'def factorial(n):\n    return 1 if n == 0 else n * factorial(n - 1)'
    solution = generate_solution(instruction, code)
    print(solution)

    # Test 3: Generate a solution with execution output
    print('\n3. Generating a solution with execution output:')
    instruction = 'Fix the bug in the following code that calculates the sum of even numbers in a list.'
    code = 'def sum_even(numbers):\n    return sum(num for num in numbers if num % 2 == 0)'
    exec_output = 'TypeError: \'int\' object is not iterable'
    solution = generate_solution(instruction, code, exec_output)
    print(solution)


    ### Test get_actions_from_diff function ###

    print('\nTesting get_actions_from_diff function:')
    from transformers import AutoTokenizer
    from code_env import add_and_init_special_token, InteractivePythonEnv, KEY_ENTER_TOKEN, ENV_SPECIAL_TOKENS

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    for token in ENV_SPECIAL_TOKENS:
        add_and_init_special_token(token, tokenizer)
    vocab = tokenizer.get_vocab()

    # Create environment
    env = InteractivePythonEnv(tokenizer, vocab)

    # Define start and end code snippets
    start_code = """def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1): # The goal will be to remove this comment
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j], arr[j+1] # And swap these
    return arr
"""

    end_code = """def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr
"""

    # Reset environment and enter initial code
    env.reset()
    for token_id in tokenizer.encode(start_code):
        env.step(token_id)
    env.step(vocab[KEY_ENTER_TOKEN])

    # Get actions from diff
    start_ids = env.get_dict_obs(include_cursor=False)['code']
    print(tokenizer.decode(start_ids))
    
    assert start_ids == tokenizer.encode(start_code)
    end_ids = tokenizer.encode(end_code)
    actions = get_actions_from_diff(start_ids, end_ids, tokenizer, env._cursor_pos)
    print('Actions to take:\n', tokenizer.decode(actions))
    print()

    # Apply actions to environment
    for action in actions:
        env.step(action)

    # Get final code from environment
    final_code = tokenizer.decode(env.get_dict_obs(include_cursor=False)['code'])
    print('Final code:\n', final_code)

    assert final_code == end_code, 'Final code does not match target code'