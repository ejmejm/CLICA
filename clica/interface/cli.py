import curses
from typing import Callable, Optional
import copy
import logging
import os

import hydra
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer

from clica.agent import BaseAgent
from clica.database import ActionSource, ActionType, InteractionDatabase
from clica.interface.curses_utils import LineWriter
from clica.interface.inputs import *
from clica.interface.states import MenuState


class InteractiveCLI():
    """CLI to interactively train an agent."""

    def __init__(
            self,
            agent: BaseAgent,
            make_env: Callable,
            env_kwargs: Optional[dict] = None,
            supervised_train_mode: str = 'multi-token', # 'multi-token' or 'single-token'
            db_path: Optional[str] = 'data/interactions.db',
            model_save_dir: Optional[str] = None,
            eval_data_path: Optional[str] = None,
        ):
        """Initializes the interactive CLI.

        Args:
            env_kwargs (dict, optional): Keyword arguments for the TextEditEnv. Defaults to None.
        """
        # agent. -> eos_token, max_gen_length, get_action(obs), train_supervised({obs, act})...
        self.agent = agent
        self.supervised_train_mode = supervised_train_mode
        self.eval_data_path = eval_data_path

        env_kwargs = env_kwargs or {}
        self.env = make_env(**env_kwargs)
        self.env.reset()
        self.user_feedback = ''
        self.curr_reward = 0
        self.state = MenuState
        
        self._use_db = db_path is not None
        if self._use_db:
            self.db = InteractionDatabase(db_path)
            self.db.create_session()
    
        self.last_trained_step = 0
        self.last_trained_env = copy.deepcopy(self.env)  # Initialize with the starting environment
        
        self.model_save_dir = model_save_dir
        self.loaded_model_name = None
    
    # TODO: Add setting the prompt actions to the database too
    def _env_enact(self, action: int, source: ActionSource = ActionSource.HUMAN, correct: Optional[bool] = None):
        """Enacts an action in the environment and updates the database."""
        self.env.step(action)
        if self._use_db:
            action_str = self.agent.tokenizer.decode([action])
            self.db.add_action(action_str, ActionType.INPUT, source, correct)

    def _write_obs_to_screen(self, writer: LineWriter):
        prompt_header = "=" * 5 + " User Instruction " + "=" * 5
        prompt_lines = self._get_env_instruction().splitlines()
        
        code_header = "=" * 5 + " Project Code " + "=" * 5
        code_lines = self._get_env_code(include_cursor=False).splitlines()

        text_queue_header = "=" * 5 + " Text Queue " + "=" * 5
        text_queue_lines = self._get_env_text_queue().splitlines()

        exec_output_header = "=" * 5 + " Execution Output " + "=" * 5
        exec_output_lines = self._get_env_exec_output().splitlines()

        ### Prompt text ###
        writer.write(prompt_header, curses.A_ITALIC)
        writer.write(prompt_lines)
        writer.skip_lines(2)
        
        ### Code text ###
        writer.write(code_header, curses.A_ITALIC)
        writer.write(code_lines)
        writer.skip_lines(2)

        ### Text queue ###
        writer.write(text_queue_header, curses.A_ITALIC)
        writer.write(text_queue_lines)
        writer.skip_lines(2)

        ### Execution output ###
        writer.write(exec_output_header, curses.A_ITALIC)
        writer.write(exec_output_lines)

    def _write_command_menu_to_screen(self, writer: LineWriter):
        """Writes the command menu to the screen.
        
        Example of format:
            Available commands:
            [p]rompt | [e]xample | [+] reward | [-] reward | [ENTER] end turn
        """
        commands_header = "Available commands:"
        available_commands = self.state._get_available_commands()
        commands_text = " | ".join(available_commands.values())
        writer.write(commands_header)
        writer.write(commands_text)

    def _get_env_code(self, include_cursor: bool = False):
        """Returns the code from the environment as a string."""
        dict_obs = self.env.get_dict_obs(include_cursor=include_cursor)
        code = self.agent.tokenizer.decode(dict_obs['code'])
        return code

    def _get_env_text_queue(self):
        """Returns the text queue from the environment as a string."""
        dict_obs = self.env.get_dict_obs()
        text_queue = self.agent.tokenizer.decode(dict_obs['text_queue'])
        return text_queue

    def _get_env_instruction(self):
        """Returns the instruction from the environment as a string."""
        dict_obs = self.env.get_dict_obs()
        instruction = self.agent.tokenizer.decode(dict_obs['instruction'])
        return instruction

    def _get_env_exec_output(self):
        """Returns the execution output from the environment as a string."""
        dict_obs = self.env.get_dict_obs()
        exec_output = self.agent.tokenizer.decode(dict_obs['exec_output'])
        return exec_output

    def _query_and_parse_user_key(self):
        key = self.stdscr.getkey()

        # If the key is ESC, wait for the next key to get the full escape sequence
        if key == '\x1b':
            self.stdscr.nodelay(True)
            while True:
                try:
                    key += self.stdscr.getkey()
                except curses.error:
                    break
            self.stdscr.nodelay(False)

        # Convert escape sequences to commands (e.g. '\033[A' -> '<|KEY_UP|>')
        key = parse_escape_key(key)
        
        return key

    def _interaction_loop(self, stdscr):
        """Main loop that passes control to the appropriate handler function."""
        self.stdscr = stdscr
        self.stdscr.keypad(True)
        
        while self.state != None:
            self.state = self.state.handle_execution(self)

        self.db.close()

    def run(self):
        """Call externally to start the interactive CLI."""
        curses.wrapper(self._interaction_loop)
        self.stdscr = None

    def reset_session(self):
        """Resets the session, wiping the environment and starting a new session."""
        self.env.reset()
        self.curr_reward = 0
        if self._use_db:
            self.db.create_session()
        self.last_trained_step = 0
        self.last_trained_env = copy.deepcopy(self.env)


def suppress_cli_warnings():
    # Suppress logger warnings
    logging.getLogger('LiteLLM').setLevel(logging.ERROR)
    logging.getLogger('Pydantic').setLevel(logging.ERROR)

    # Suppress TensorFlow warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


@hydra.main(version_base=None, config_path='../../conf', config_name='default')
def test_cli(config: DictConfig):
    from clica.agent import DummyAgent

    suppress_cli_warnings()
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    config = OmegaConf.create(config)

    tokenizer = AutoTokenizer.from_pretrained(config.get('tokenizer_name', config.model_name))
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Add special tokens
    for token in ENV_SPECIAL_TOKENS:
      add_and_init_special_token(token, tokenizer)

    vocab = tokenizer.get_vocab()
    
    db_path = os.path.join(
        os.path.dirname(__file__),
        '../../data/test_interactions.db',
    )

    agent = DummyAgent(None, tokenizer, max_gen_length=8)
    cli = InteractiveCLI(
        agent = agent,
        make_env = InteractivePythonEnv,
        db_path = db_path,
        model_save_dir = config.get('model_save_dir'),
        eval_data_path = config.get('eval_data_path'),
        env_kwargs = dict(
            tokenizer = agent.tokenizer,
            vocab = vocab,
        ),
    )
    cli.run()


if __name__ == '__main__':
    test_cli()
