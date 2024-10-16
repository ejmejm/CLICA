import os
from typing import Dict, Any, List, Optional
from multiprocessing import Manager, Process
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter, defaultdict
import numpy as np
import tqdm

from clica.agent import BaseAgent
from clica.code_env import InteractivePythonEnv, KEY_ENTER_TOKEN
from human_eval.data import stream_jsonl, read_problems, write_jsonl
from human_eval.evaluation import evaluate_functional_correctness, estimate_pass_at_k, HUMAN_EVAL
from human_eval.execution import create_tempdir, reliability_guard, swallow_io, time_limit, TimeoutException


BENCHMARK_OUTPUT_DIR = '.benchmark_outputs/'
N_WORKERS = 4


def generate_human_eval_completion(prompt: str, agent: BaseAgent, env: InteractivePythonEnv):
    env.reset()
    
    # Start by entering the code in the prompt
    prompt_token_ids = agent.tokenizer.encode(prompt, add_special_tokens=False)
    for token_id in prompt_token_ids:
        env.step(token_id)
    env.step(agent.tokenizer.convert_tokens_to_ids(KEY_ENTER_TOKEN))
    
    # Set the instruction
    instruction_ids = agent.tokenizer.encode("Finish writing the function")
    env.set_instruction(instruction_ids)
    
    # Now prompt the agent to generate the rest of the code
    # TODO: Note that there is a big difference between the agent starting with an empty recurrent state and
    # passing all the prompt actions into the recurrent state.
    # In practice, the agent should probably be trained with every action being passed into the recurrent state,
    # even when it is taken by a human, because it gives the agent context on what has happened previously.
    # Currently, that is not being done here to save a little on compute, but it really should be done in the future.
    state = None
    agent.execute_action_loop(state, env)
    
    solution_token_ids = env.get_dict_obs(include_cursor=False)['code']
    solution_text = agent.tokenizer.decode(solution_token_ids)
    
    return solution_text


def run_human_eval(data_path: str, agent: BaseAgent, env: InteractivePythonEnv, n_samples_per_task: int = 1):
    problems = read_problems(data_path)
    samples = [
        dict(task_id=task_id, completion=generate_human_eval_completion(problems[task_id]['prompt'], agent, env))
        for task_id in problems
        for _ in range(n_samples_per_task)
    ]
    
    os.makedirs(BENCHMARK_OUTPUT_DIR, exist_ok=True)

    # Write the results to a file
    filepath = os.path.join(BENCHMARK_OUTPUT_DIR, 'human_eval_samples.jsonl')
    write_jsonl(filepath, samples)

    evaluate_functional_correctness(filepath, k=[1], n_workers=N_WORKERS, timeout=20, problem_file=data_path)

    # Read the results
    results = list(stream_jsonl(filepath + '_results.jsonl'))
    passed = [r['passed'] for r in results]
    passed_frac = sum(passed) / len(passed)

    return {'passed': f'{sum(passed)}/{len(passed)}', 'passed_frac': passed_frac}


### Sourced from the human_eval repo ###


def unsafe_execute(problem: Dict[str, Any], completion: str, timeout: float, result: List):
    with create_tempdir():

        # These system calls are needed when cleaning up tempdir.
        import os
        import shutil
        rmtree = shutil.rmtree
        rmdir = os.rmdir
        chdir = os.chdir

        # Disable functionalities that can make destructive changes to the test.
        reliability_guard()

        # Construct the check program and run it.
        check_program = (
                completion + "\n" +
                problem["test"] + "\n" +
                f"check({problem['entry_point']})"
        )

        try:
            exec_globals = {}
            with swallow_io():
                with time_limit(timeout):
                    exec(check_program, exec_globals)
            result.append("passed")
        except TimeoutException:
            result.append("timed out")
        except BaseException as e:
            result.append(f"failed: {e}")

        # Needed for cleaning up.
        shutil.rmtree = rmtree
        os.rmdir = rmdir
        os.chdir = chdir


def check_correctness(problem: Dict, completion: str, timeout: float,
                      completion_id: Optional[int] = None) -> Dict:
    """
    Evaluates the functional correctness of a completion by running the test
    suite provided in the problem. 

    :param completion_id: an optional completion ID so we can match
        the results later even if execution finishes asynchronously.
    """
    with Manager() as manager:
        result = manager.list()

        p = Process(target=unsafe_execute, args=(problem, completion, timeout, result))
        p.start()
        p.join(timeout=timeout + 1)
        if p.is_alive():
            p.kill()

        if not result:
            result.append("timed out")

        return dict(
            task_id=problem["task_id"],
            passed=result[0] == "passed",
            result=result[0],
            completion_id=completion_id,
        )

def evaluate_functional_correctness(
    sample_file: str,
    k: List[int] = [1, 10, 100],
    n_workers: int = 4,
    timeout: float = 3.0,
    problem_file: str = HUMAN_EVAL,
    ignore_incomplete: bool = False
):
    """
    Evaluates the functional correctness of generated samples, and writes
    results to f"{sample_file}_results.jsonl.gz"
    """

    problems = read_problems(problem_file)

    # Check the generated samples against test suites.
    with ThreadPoolExecutor(max_workers=n_workers) as executor:

        futures = []
        completion_id = Counter()
        n_samples = 0
        results = defaultdict(list)

        print("Reading samples...")
        for sample in tqdm.tqdm(stream_jsonl(sample_file)):
            task_id = sample["task_id"]
            completion = sample["completion"]
            args = (problems[task_id], completion, timeout, completion_id[task_id])
            future = executor.submit(check_correctness, *args)
            futures.append(future)
            completion_id[task_id] += 1
            n_samples += 1

        if not ignore_incomplete:
            assert len(completion_id) == len(problems), "Some problems are not attempted."

        print("Running test suites...")
        for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            results[result["task_id"]].append((result["completion_id"], result))

    # Calculate pass@k.
    total, correct = [], []
    for result in results.values():
        result.sort()
        passed = [r[1]["passed"] for r in result]
        total.append(len(passed))
        correct.append(sum(passed))
    total = np.array(total)
    correct = np.array(correct)

    ks = k
    pass_at_k = {f"pass@{k}": estimate_pass_at_k(total, correct, k).mean()
                 for k in ks if (total >= k).all()}

    # Finally, save the results in one file:
    def combine_results():
        for sample in stream_jsonl(sample_file):
            task_id = sample["task_id"]
            result = results[task_id].pop(0)
            sample["result"] = result[1]["result"]
            sample["passed"] = result[1]["passed"]
            yield sample

    out_file = sample_file + "_results.jsonl"
    print(f"Writing results to {out_file}...")
    write_jsonl(out_file, tqdm.tqdm(combine_results(), total=n_samples))

    return pass_at_k
