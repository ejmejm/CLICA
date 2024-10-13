from curses import ascii

from clica.code_env import *


def parse_escape_key(key: str) -> str:
    """Converts an escape sequences to a command string, otherwise does nothing.

    Args:
        key (str): The key to parse.

    Returns:
        str: The parsed key.
    """
    return SPECIAL_KEY_TO_TOKEN_MAP.get(key, key)


def control_key(key: str) -> str:
    """
    Converts a string character to its corresponding control character.

    Args:
        key (str): A single character string representing the key to be converted.

    Returns:
        str: A string representing the corresponding control character.
    """
    return chr(ord(key.upper()) - 64)


def alt_key(key: str) -> str:
    """
    Prefixes a string character with the escape character (\\x1b).

    Args:
        key (str): A single character string representing the key to be prefixed.

    Returns:
        str: A string representing the corresponding alt character.
    """
    return '\x1b' + key


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

KEY_CTRL_P = control_key('p')
KEY_CTRL_E = control_key('e')