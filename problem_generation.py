from enum import Enum
from typing import List, Optional
import litellm
import random


RANDOM_TOPICS = [
    "string manipulation", "list comprehension", "file I/O",
    "data structures", "algorithms", "object-oriented programming",
    "error handling", "regular expressions", "recursion",
    "sorting and searching", "mathematical operations",
    "functional programming", "generators",
    "context managers", "multithreading", "multiprocessing",
    "cryptography", "text processing",
]


class DifficultyLevel(Enum):
    BEGINNER = 0
    INTERMEDIATE = 1
    ADVANCED = 2


def generate_problems(n: int = 1, topic: Optional[str] = None, level: Optional[DifficultyLevel] = None, existing_code: Optional[str] = None) -> List[str]:
    prompt = _create_problem_prompt(topic, level, existing_code)
    
    response = litellm.completion(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": prompt}],
        temperature=0.9,
        max_tokens=500,
        n=n,
    )

    problems = [choice.message.content.strip() for choice in response.choices]
    return problems


def generate_problem(topic: Optional[str] = None, level: Optional[DifficultyLevel] = None, existing_code: Optional[str] = None) -> str:
    return generate_problems(n=1, topic=topic, level=level, existing_code=existing_code)[0]


def _create_problem_prompt(topic: Optional[str], level: Optional[DifficultyLevel], existing_code: Optional[str]) -> str:
    base_prompt = (
        "Generate a concise Python programming problem that is self-contained "
        "and solvable using only default Python libraries. The problem should "
        "be clear and require a solution of no more than 40 lines of code. "
    )

    if existing_code:
        base_prompt += (
            f"Use the following existing code as a basis for the problem:\n\n{existing_code}\n\n"
            "The problem should be related to this code. It could involve fixing a bug, "
            "renaming variables, changing the implementation, or any other relevant task. "
        )
    if topic:
        base_prompt += f"The problem should be related to the topic: {topic}. "

    if level:
        difficulty_descriptions = {
            DifficultyLevel.BEGINNER: "suitable for beginners, focusing on basic concepts",
            DifficultyLevel.INTERMEDIATE: "moderately challenging, involving more complex logic",
            DifficultyLevel.ADVANCED: "challenging, requiring advanced problem-solving skills",
        }
        base_prompt += f"The difficulty level should be {difficulty_descriptions[level]}. "
    else:
        base_prompt += "The difficulty level should vary. "

    base_prompt += (
        "Present only the problem statement without any additional explanation or solution. "
        "Describe the task directly without any prefix."
    )

    return base_prompt


if __name__ == '__main__':
    """Run simple tests for problem generation functions."""
    print("Testing problem generation:")
    
    # Test 1: Generate a single problem
    print("\n1. Generating a single problem:")
    problem = generate_problem()
    print(problem)

    # Test 2: Generate multiple problems
    print("\n2. Generating 3 problems:")
    problems = generate_problems(n=3)
    for i, prob in enumerate(problems, 1):
        print(f"\nProblem {i}:")
        print(prob)

    # Test 3: Generate a problem with a specific topic
    print("\n3. Generating a problem about list comprehension:")
    problem = generate_problem(topic="list comprehension")
    print(problem)

    # Test 4: Generate a problem with a specific difficulty level
    print("\n4. Generating an advanced problem:")
    problem = generate_problem(level=DifficultyLevel.ADVANCED)
    print(problem)

    # Test 5: Generate a problem based on existing code
    print("\n5. Generating a problem based on existing code:")
    existing_code = "def factorial(n):\n    return 1 if n == 0 else n * factorial(n - 1)"
    problem = generate_problem(existing_code=existing_code)
    print(problem)