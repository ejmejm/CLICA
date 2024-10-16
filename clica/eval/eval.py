import json
import os
from typing import Any, Dict, Optional

from clica.agent import BaseAgent
from clica.code_env import InteractivePythonEnv
from clica.eval.human_eval import run_human_eval


def run_human_eval_from_task_path(agent: BaseAgent, task_path: str) -> Dict[str, Any]:
    # If task_path is a file, check if it's a .jsonl file
    if os.path.isfile(task_path):
        if not task_path.endswith('.jsonl'):
            raise ValueError(f"Human eval file {task_path} is not a .jsonl file!")
        else:
            data_path = task_path
    # If task_path is a directory, get the first .jsonl file in the directory
    else:
        jsonl_files = [f for f in os.listdir(task_path) if f.endswith('.jsonl')]
        if not jsonl_files:
            raise ValueError(f"No .jsonl file found for human eval in directory {task_path}")
        data_path = os.path.join(task_path, jsonl_files[0])

    env = InteractivePythonEnv(
        tokenizer = agent.tokenizer,
        vocab = agent.tokenizer.get_vocab(),
    )

    # Run human eval
    eval_results = run_human_eval(data_path, agent, env)
    return eval_results


def run_agent_eval(agent: BaseAgent, task_path: str, eval_type: Optional[str] = None) -> Dict[str, Any]:
    """Determines which eval to run based on the task_path, and runs it.
    
    Returns:
        The evaluation results.
    """
    # If the task_path is a folder, get the metadata.json file
    if not eval_type:
        if os.path.isdir(task_path):
            metadata_path = os.path.join(task_path, 'metadata.json')
            if not os.path.exists(metadata_path):
                raise ValueError(f"No metadata.json file found in {task_path}!")

            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            eval_type = metadata.get('eval_type', 'human_eval')
            if eval_type is None:
                raise ValueError("No eval_type found in metadata.json!")
        else:
            eval_type = 'human_eval'

    # Run the appropriate eval
    if eval_type == 'human_eval':
        eval_results = run_human_eval_from_task_path(agent, task_path)
    else:
        raise ValueError(f"Eval type '{eval_type}' not supported!")

    return eval_results