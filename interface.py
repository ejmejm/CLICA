import curses
from curses import ascii
import enum
from typing import Callable, Optional, Union

import hydra
from omegaconf import DictConfig, OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer

from agent import BaseAgent, DummyAgent
from code_env import *


# # Map of escape sequences to special keys (e.g. '\033[A' -> 'up')
# ESCAPE_SEQ_MAP = {}
# COMMAND_CHARS_LEN = len(COMMAND_OPENER) + len(COMMAND_CLOSER)
# for key, value in SPECIAL_KEYS.items():
#     # For duplicate values, only keep the shortest key
#     if value in ESCAPE_SEQ_MAP and \
#        len(key) + COMMAND_CHARS_LEN < len(ESCAPE_SEQ_MAP[value]):
#         continue
#     ESCAPE_SEQ_MAP[value] = COMMAND_OPENER + key + COMMAND_CLOSER


KEY_CTRL_RIGHT_TOKEN = format_command_token('key_ctrl_right')
KEY_CTRL_R_TOKEN = RUN_CODE_TOKEN

SPECIAL_KEY_TO_TOKEN_MAP = {
  '\033[A': KEY_UP_TOKEN,
  '\033[B': KEY_DOWN_TOKEN,
  '\033[C': KEY_RIGHT_TOKEN,
  '\033[D': KEY_LEFT_TOKEN,
  '\x7f': KEY_BACKSPACE_TOKEN,
  '\033[1;5C': KEY_CTRL_RIGHT_TOKEN,
  ascii.ctrl('r'): KEY_CTRL_R_TOKEN,
}


def parse_escape_key(key: str) -> str:
    """Converts an escape sequences to a command string, otherwise does nothing.

    Args:
        key (str): The key to parse.

    Returns:
        str: The parsed key.
    """
    return SPECIAL_KEY_TO_TOKEN_MAP.get(key, key)


class CLIState(enum.Enum):
    """Enum for the different states of the CLI."""
    MENU = 0
    PROMPT = 1
    EXAMPLE = 2
    AGENT_TURN = 3


class LineWriter():
    """Class for writing lines to the terminal with curses.
    
    Use this class instead of curses.addstr() to write to the terminal.
    This handles the line index automatically and ensures that the text
    does not go out of bounds (which would cause an error).

    Example:
        writer = LineWriter(stdscr)
        writer.write('Hello world!')
        writer.write('This is a new line.', x=10)
        writer.write('This is another new line.')
    """

    def __init__(self, stdscr: curses.window):
        """Initializes the LineWriter.
        
        Args:
            stdscr (curses.window): The curses window to write to.
        """
        self.stdscr = stdscr
        self.reset()

    def reset(self):
        """Resets the line index to 0."""
        self.line_idx = 0
        self.max_line_idx = self.stdscr.getmaxyx()[0] - 1

    def write(self, string: Union[str, list[str]], *args, x: int = 0, **kwargs):
        """Writes a string or a list of strings to the terminal.

        Args:
            string (str): String to write.
            x (int, optional): X coordinate. Defaults to 0.
        """
        if isinstance(string, str):
            string = string.splitlines() # addstr() does not support multiline strings

        for line in string:
            if self.line_idx < self.max_line_idx:
                self.stdscr.addstr(self.line_idx, x, line, *args, **kwargs)
            else:
                self.stdscr.addstr(self.max_line_idx, x, '...', *args, **kwargs)
            self.line_idx += 1
            
    def skip_lines(self, n: int):
        """Skips n lines.
        
        Args:
            n (int): Number of lines to skip.
        """
        self.line_idx += n


class InteractiveCLI():
    """CLI to interactively train an agent."""

    def __init__(
            self,
            agent: BaseAgent,
            make_env: Callable,
            env_kwargs: Optional[dict] = None,
            supervised_train_mode: str = 'multi-token' # 'multi-token' or 'single-token'
        ):
        """Initializes the interactive CLI.

        Args:
            env_kwargs (dict, optional): Keyword arguments for the TextEditEnv. Defaults to None.
        """

        # agent. -> eos_token, max_gen_length, get_action(obs), train_supervised({obs, act})...
        self.agent = agent
        self.supervised_train_mode = supervised_train_mode

        env_kwargs = env_kwargs or {}
        self.env = make_env(**env_kwargs)
        self.env.reset()
        self.user_feedback = ''
        self.curr_reward = 0
        self.state = CLIState.MENU

    def _get_available_commands(self):
        """Returns a dictionary of available commands for the current state."""
        if self.state == CLIState.MENU:
            return {
                'p': '[p]rompt',
                'e': '[e]xample',
                '+': '[+] reward',
                '-': '[-] reward',
                'ENTER': '[ENTER] end turn'
            }
        elif self.state == CLIState.PROMPT:
            return {'ESC': '[ESC] menu'}
        elif self.state == CLIState.EXAMPLE:
            return {
                '^r': '[ctrl+r] run code',
                '^KEY_RIGHT': '[ctrl+right] insert text',
                'ESC': '[ESC] menu',
            }
        return {}

    def _write_obs_to_screen(self, writer: LineWriter):
        prompt_header = "=" * 5 + " User Instruction " + "=" * 5
        prompt_lines = self._get_env_instruction().splitlines()
        
        code_header = "=" * 5 + " Project Code " + "=" * 5
        code_lines = self._get_env_code(include_cursor=True).splitlines()

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
        # TODO: Write warning if the code window is smaller than the env window
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
        available_commands = self._get_available_commands()
        commands_text = " | ".join(available_commands.values())
        writer.write(commands_header)
        writer.write(commands_text)

    def _render_menu(self):
        """Renders the main menu to the terminal."""

        # Write to screen
        self.stdscr.clear()

        writer = LineWriter(self.stdscr)
        
        ### Env observation ###
        self._write_obs_to_screen(writer)
        writer.skip_lines(3)

        ### Commands text ###
        self._write_command_menu_to_screen(writer)
        writer.skip_lines(2)

        writer.write(f"Current reward: {self.curr_reward}")

        self.stdscr.refresh()

    def _handle_menu(self):
        """Handles user input in the main menu."""
        self._render_menu()
        key = self.stdscr.getkey()
        if key == 'p':
            self.state = CLIState.PROMPT
        elif key == 'e':
            self.state = CLIState.EXAMPLE
        elif key in ('+', '='):
            self.curr_reward += 1
        elif key == '-':
            self.curr_reward -= 1
        elif key in ('\n', '\r'):
            self.state = CLIState.AGENT_TURN
        elif key in ('\x1b', '\x03'): # ESC, ctrl+c
            return True
        return False

    def _render_prompt(self):
        """Renders the prompt editor to the terminal."""
        self.stdscr.clear()
        self.stdscr.addstr(0, 0, "Write a new prompt:\n", curses.A_BOLD)
        self.stdscr.refresh()

    def _handle_prompt(self):
        """Handles user input in the prompt editor."""
        self._render_prompt()
        curses.echo()
        self.user_prompt = self.stdscr.getstr(1, 0).decode('utf-8')
        curses.noecho()
        
        self.state = CLIState.MENU

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

    def _render_example(self, insert_queue: str):
        """Renders the project code to the terminal."""
        self.stdscr.clear()

        writer = LineWriter(self.stdscr)
        self._write_obs_to_screen(writer)
        
        writer.write(insert_queue)
        writer.skip_lines(3)
        
        self._write_command_menu_to_screen(writer)

        self.stdscr.refresh()
        
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

    def _handle_example(self):
        """Handles user input in the code editor."""
        # Use standard key escape sequences
        self.stdscr.keypad(False)

        # example_buffer = {'prompt': [], 'code': [], 'act': []}
        insert_queue = ''
        queued_actions = []
        
        self._render_example(insert_queue)
        
        while True:
            
            # If there are actions in the queue, execute them and rerender the environment
            if len(queued_actions) > 0:
                action_id = queued_actions.pop(0)
                self.env.step(action_id)
                self._render_example(insert_queue)
                continue
            
            # Otherwise, get the next action from the user
            key = self._query_and_parse_user_key()
            
            # Handle escape / resize events
            if key == '\x1b':
                # Enact all actions waiting in the queue before exiting
                if len(insert_queue) > 0:
                    actions = self.agent.tokenizer.encode(insert_queue)
                    for act in actions:
                        self.env.step(act)
                break
            elif key == 'KEY_RESIZE':
                continue

            # If enter key, take actions for the queued text, and then take the enter key action
            if key == KEY_CTRL_RIGHT_TOKEN:
                queued_actions.extend(self.agent.tokenizer.encode(insert_queue))
                queued_actions.append(self.agent.tokenizer.convert_tokens_to_ids(KEY_ENTER_TOKEN))
                insert_queue = ''
                
            # Handle normal text
            elif key not in COMMAND_TOKENS:
                insert_queue += key
            
            # Otherwise it is just a normal command token
            # This isn't a great solution, but for now we will ignore it if text is queued
            elif len(insert_queue) > 0:
                continue

            else:
                self.env.step(self.agent.tokenizer.convert_tokens_to_ids(key))
            
            self._render_example(insert_queue)
            
            # example_buffer['prompt'].append(self.user_prompt)
            # example_buffer['code'].append(code)
            # example_buffer['act'].append(key)

        # Back to simplified key representations (e.g. 'KEY_UP')
        # self.agent.train_supervised(example_buffer)
        self.stdscr.keypad(True)
        self.state = CLIState.MENU

    def _handle_agent_turn(self):
        """Handles the agent's turn, prompting it to generate text."""
        action = None
        act_idx = 0
        stop_flag = False
        while act_idx < self.agent.max_gen_length and not stop_flag:
            self._render_menu()
            actions = [self.agent.get_action(self.env.get_obs())]
            for action in actions:
                if action == self.agent.eos_token or act_idx >= self.agent.max_gen_length:
                    stop_flag = True
                    break
                
                self.env.step(action)
                act_idx += 1

        # for _ in range(self.agent.max_gen_length):
        #     self._render_menu()
        #     action = self.agent.get_action(self.user_prompt, self.env.get_obs())
        #     if action == self.agent.eos_token:
        #         self.agent.clear_action_queue()
        #         break
        #     self.env.step(action)

        self.state = CLIState.MENU

    def _interaction_loop(self, stdscr):
        """Main loop that passes control to the appropriate handler function."""
        self.stdscr = stdscr
        self.stdscr.keypad(True)
        
        while True:
            if self.state == CLIState.MENU:
                terminate = self._handle_menu()
                if terminate:
                    break
            elif self.state == CLIState.PROMPT:
                self._handle_prompt()
            elif self.state == CLIState.EXAMPLE:
                self._handle_example()
            elif self.state == CLIState.AGENT_TURN:
                self._handle_agent_turn()
            else:
                raise ValueError(f'Invalid state: {self.state}')

    def run(self):
        """Call externally to start the interactive CLI."""
        curses.wrapper(self._interaction_loop)
        self.stdscr = None


@hydra.main(version_base=None, config_path='conf', config_name='default')
def run_cli(config: DictConfig):
    config = OmegaConf.create(config)
    
    model = AutoModelForCausalLM.from_pretrained('gpt2')
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    
    # Add special tokens
    for token in ENV_SPECIAL_TOKENS:
      add_and_init_special_token(model, tokenizer, token)
      
    vocab = tokenizer.get_vocab()
    
    agent = DummyAgent(model, tokenizer, max_gen_length=8)
    cli = InteractiveCLI(
        agent = agent,
        make_env = InteractivePythonEnv,
        env_kwargs = dict(
            tokenizer = agent.tokenizer,
            vocab = vocab,
        ),
    )
    cli.run()


if __name__ == '__main__':
    run_cli()