from enum import Enum
import threading
import uuid

import pandas as pd
from datasets import Dataset
import sqlite3
import time
from typing import Optional, Generator


class ActionSource(Enum):
    HUMAN = 'human'
    AI = 'ai'
    DATASET = 'dataset'


class ActionType(Enum):
    INPUT = 'input'
    SET_INSTRUCTION = 'set_instruction'
    SET_CODE = 'set_code'
    SET_EXEC_OUTPUT = 'set_exec_output'


class InteractionDatabase:
    def __init__(self, db_path: str = 'data/interactions.db', autosave_interval: int = 15):
        self.db_path: str = db_path
        self.conn: sqlite3.Connection = sqlite3.connect(db_path, check_same_thread=False)
        self.cursor: sqlite3.Cursor = self.conn.cursor()
        self.lock: threading.Lock = threading.Lock()
        self.create_tables()
        self.current_session_id: Optional[int] = None
        self.current_action_id: Optional[int] = None
        self.autosave_interval: int = autosave_interval
        self.start_autosave()

    def create_tables(self) -> None:
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id INTEGER PRIMARY KEY,
                initial_instruction TEXT,
                initial_code TEXT,
                initial_exec_output TEXT,
                verified BOOLEAN
            )
        """)
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS actions (
                action_uuid TEXT PRIMARY KEY,
                session_id INTEGER,
                action_id INTEGER,
                action_type TEXT,
                action TEXT,
                correct BOOLEAN DEFAULT NULL,
                source TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions (session_id)
            )
        """)
        self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_session_id ON actions (session_id)")
        self.conn.commit()

    def create_session(self, initial_instruction: str = "", initial_code: str = "", initial_exec_output: str = "", verified: bool = False) -> int:
        with self.lock:
            self.cursor.execute("""
                INSERT INTO sessions (initial_instruction, initial_code, initial_exec_output, verified)
                VALUES (?, ?, ?, ?)
            """, (initial_instruction, initial_code, initial_exec_output, verified))
            self.conn.commit()
            self.current_session_id = self.cursor.lastrowid
            self.current_action_id = 0
        return self.current_session_id

    def add_action(
        self,
        action: str,
        action_type: ActionType,
        source: ActionSource = ActionSource.HUMAN,
        correct: Optional[bool] = None,
        session_id: Optional[int] = None,
    ) -> None:
        if session_id is None:
            session_id = self.current_session_id
        
        with self.lock:
            if session_id == self.current_session_id:
                self.current_action_id += 1
                action_id = self.current_action_id
            else:
                self.cursor.execute("""
                    SELECT COALESCE(MAX(action_id), 0) + 1
                    FROM actions
                    WHERE session_id = ?
                """, (session_id,))
                action_id = self.cursor.fetchone()[0]

            self.cursor.execute("""
                INSERT INTO actions (action_uuid, session_id, action_id, action_type, action, correct, source)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (str(uuid.uuid4()), session_id, action_id, action_type.value, action, correct, source.value))
            self.conn.commit()

    def set_session_verified(self, session_id: Optional[int] = None) -> None:
        if session_id is None:
            session_id = self.current_session_id
        
        with self.lock:
            self.cursor.execute("UPDATE sessions SET verified = ? WHERE session_id = ?", (True, session_id))
            self.conn.commit()

    def get_total_sessions(self) -> int:
        self.cursor.execute("SELECT COUNT(*) FROM sessions")
        return self.cursor.fetchone()[0]

    def get_total_actions(self) -> int:
        self.cursor.execute("SELECT COUNT(*) FROM actions")
        return self.cursor.fetchone()[0]

    def sample_random_session_id(self) -> Optional[int]:
        self.cursor.execute("SELECT session_id FROM sessions ORDER BY RANDOM() LIMIT 1")
        result = self.cursor.fetchone()
        return result[0] if result else None

    def get_session_actions(self, session_id: int) -> pd.DataFrame:
        query = """
            SELECT *
            FROM actions
            WHERE session_id = ?
            ORDER BY action_id
        """
        return pd.read_sql_query(query, self.conn, params=(session_id,))

    def action_generator(self, batch_size: int = 32) -> Generator[Dataset, None, None]:
        query = """
            SELECT s.*, a.*
            FROM sessions s
            JOIN actions a ON s.session_id = a.session_id
            ORDER BY s.session_id, a.action_id
        """
        for chunk in pd.read_sql_query(query, self.conn, chunksize=batch_size):
            yield Dataset.from_pandas(chunk)

    def create_hf_dataset(self, output_path: str = "ai_chat_dataset") -> None:
        sessions_query = "SELECT * FROM sessions ORDER BY session_id"
        actions_query = "SELECT * FROM actions ORDER BY session_id, action_id"
        
        sessions_df = pd.read_sql_query(sessions_query, self.conn)
        actions_df = pd.read_sql_query(actions_query, self.conn)
        
        sessions_dataset = Dataset.from_pandas(sessions_df)
        actions_dataset = Dataset.from_pandas(actions_df)
        
        sessions_dataset.save_to_disk(f"{output_path}/sessions")
        actions_dataset.save_to_disk(f"{output_path}/actions")

    def autosave(self) -> None:
        while self.run_autosave_thread:
            with self.lock:
                self.conn.commit()
            time.sleep(self.autosave_interval)

    def start_autosave(self) -> None:
        self.run_autosave_thread = True
        self.autosave_thread = threading.Thread(target=self.autosave, daemon=True)
        self.autosave_thread.start()

    def close(self) -> None:
        # Send signal to stop autosave thread
        self.run_autosave_thread = False
        # Save any remaining data
        with self.lock:
            self.conn.commit()
        # Then close the connection to the database
        self.conn.close()


# Usage example:
if __name__ == '__main__':
    db = InteractionDatabase(autosave_interval=15)
    
    # Create a new session
    session_id = db.create_session('Initial instruction', 'Initial code', 'Initial output')
    
    # Add actions to the session
    db.add_action('User input', ActionType.INPUT, source=ActionSource.HUMAN)
    db.add_action('Updated code', ActionType.SET_CODE, source=ActionSource.AI)
    
    # Set the session as verified
    db.set_session_verified()
    
    # Get session actions
    actions_df = db.get_session_actions(session_id)
    print(actions_df)
    
    # Create HuggingFace dataset
    db.create_hf_dataset()
    
    db.close()