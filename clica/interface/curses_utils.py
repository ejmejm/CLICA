import curses
from curses.textpad import Textbox
from typing import Optional, Union


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
        self.max_y, self.max_x = self.stdscr.getmaxyx()
        self.max_line_idx = self.max_y - 1
        self.curr_x = 0


    def write(self, string: Union[str, list[str]], *args, x: Optional[int] = None, new_line: bool = True, **kwargs):
        """Writes a string or a list of strings to the terminal.

        Args:
            string (str): String to write.
            x (int, optional): X coordinate. Defaults to 0.
        """
        if isinstance(string, str):
            string = string.splitlines()
            
        if x is not None:
            self.curr_x = x

        written = False
        for line in string:
            if len(line) == 0:
                self.line_idx += 1
                self.curr_x = 0
                continue
            
            remaining = line
            while remaining:
                written = True
                space_left = self.max_x - self.curr_x
                to_write = remaining[:space_left]
                if self.line_idx < self.max_line_idx:
                    self.stdscr.addstr(self.line_idx, self.curr_x, to_write, *args, **kwargs)
                else:
                    self.stdscr.addstr(self.max_line_idx, self.curr_x, '...', *args, **kwargs)
                    self.curr_x = 0
                    return
                
                remaining = remaining[space_left:]
                
                self.line_idx += 1
                self.curr_x = 0  # Reset x for wrapped lines
        
        # Keep the cursor at the end of the line if new_line is false
        if written and not new_line and len(to_write) < space_left:
            self.line_idx -= 1
            self.curr_x = len(to_write)


    def skip_lines(self, n: int):
        """Skips n lines.
        
        Args:
            n (int): Number of lines to skip.
        """
        self.line_idx = min(self.line_idx + n, self.max_line_idx)
        self.curr_x = 0

    def get_current_line(self):
        return self.line_idx


def get_text_input(window: curses.window, initial_text: str = '') -> Optional[str]:
    """Gets text input from the user.
    
    Args:
        window (curses.window): The curses window to write to.
        initial_text (str, optional): Initial text in the textbox. Pass nothing for an empty textbox.
    
    Returns:
        Optional[str]: The text input by the user, or None if the user pressed the escape key to exit.
    """
    textbox = Textbox(window)
    for char in initial_text:
        textbox.do_command(char)
    
    escaped = False

    # Source: https://stackoverflow.com/questions/47481955/python-curses-detecting-the-backspace-key/75950899#75950899
    def validate(char: str) -> str:
        nonlocal escaped

        # Exit input with the escape key
        escape = 27
        if char == escape:
            escaped = True
            char = curses.ascii.BEL # Control-G
        
        # Delete the character to the left of the cursor
        elif char in (curses.KEY_BACKSPACE, curses.ascii.DEL):
            char = curses.KEY_BACKSPACE

        return char

    textbox.edit(validate)
    text = textbox.gather().strip()
    
    return text if not escaped else None