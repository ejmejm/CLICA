import sys
import logging


class LoggingRedirector:
    def __init__(self, filename):
        self.filename = filename
        self.file_handler = None
        self.original_stdout = None
        self.original_stderr = None

    def __enter__(self):
        self.file = open(self.filename, 'a')
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        sys.stdout = self.file
        sys.stderr = self.file

        self.file_handler = logging.FileHandler(self.filename)
        self.file_handler.setLevel(logging.INFO)
        self.file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logging.getLogger().addHandler(self.file_handler)

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        self.file.close()
        logging.getLogger().removeHandler(self.file_handler)