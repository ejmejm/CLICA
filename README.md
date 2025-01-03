# CLICA

CLICA (Continual Learning Interactive Code Agent) is a command-line tool for interactively training a coding agent. It allows users to provide coding instructions, receive generated solutions, and continuously improve the agent's capabilities through interaction. The system uses a database to track sessions and actions, enabling the agent to adapt to user preferences over time.

## Data Format

### Sessions Table

| Column Name         | Data Type | Constraints    | Description                                                                                                           |
|---------------------|-----------|----------------|-----------------------------------------------------------------------------------------------------------------------|
| session_id          | INT       | PRIMARY KEY    | Unique identifier for each session                                                                                    |
| initial_instruction | STRING    |                | The initial instruction given for the session                                                                         |
| initial_code        | STRING    |                | The initial code state for the session                                                                                |
| initial_exec_output | STRING    |                | The initial execution output for the session                                                                          |
| verified            | BOOLEAN   |                | Indicates whether the session was verified by the user. Online sessions should be marked as verified for training use |

### Actions Table

| Column Name | Data Type | Constraints                 | Description                                                                                                        |
|-------------|-----------|-----------------------------|--------------------------------------------------------------------------------------------------------------------|
| action_uuid | STRING    | PRIMARY KEY                 | Unique identifier for each action                                                                                  |
| session_id  | INT       | FOREIGN KEY, SECONDARY INDEX| References the session this action belongs to                                                                      |
| action_id   | INT       |                             | Chronological order of actions within a session                                                                    |
| action_type | ENUM      |                             | Type of action: {input, set_instruction, set_code, set_exec_output}                                                |
| action      | STRING    |                             | The content of the action                                                                                          |
| correct     | BOOLEAN   | OPTIONAL                    | Indicates if the action is correct. Incorrect actions can be used for setup but should not be trained on directly |
| source      | ENUM      |                             | Source of the action: {human, ai, dataset}                                                                         |

## License

This project is licensed under the MIT License with Commons Clause. This means you are free to use, modify, and distribute the software for any purpose, but selling the software is prohibited.

For more details, see the [LICENSE](LICENSE) file in the project repository.

The MIT License is available at: https://opensource.org/license/mit
The Commons Clause is available at: https://commonsclause.com/
