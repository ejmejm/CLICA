from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional


if TYPE_CHECKING:
    from clica.interface.cli import InteractiveCLI


class CLIState(ABC):
    state_id: str = NotImplemented

    @staticmethod
    @abstractmethod
    def handle_execution(cli_context: InteractiveCLI) -> Optional[str]:
        pass
    
    @staticmethod
    def _get_available_commands() -> dict[str, str]:
        return {}
