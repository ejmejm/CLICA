from typing import Optional
import litellm


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
