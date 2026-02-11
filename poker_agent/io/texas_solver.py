import subprocess
from pathlib import Path
from typing import Final
from pamiq_core import Agent
from torch import Tensor
class TexasSolver:
    """Minimal wrapper for TexasSolver executable."""
    def __init__(self, solver_path: str | Path) -> None:
        """Initialize the TexasSolver wrapper.
        Args:
            solver_path: Path to the TexasSolver executable.
        """
        self.solver_path = Path(solver_path)
        if not self.solver_path.exists():
            raise FileNotFoundError(f"TexasSolver executable not found at {solver_path}")
    def solve(
        self,
        board: str,
        range_ip: str,
        range_oop: str,
        pot: float,
        stack: float,
        accuracy: float = 1e-3,
        max_iteration: int = 100,
        thread_num: int = 1,
    ) -> str:
        """Run the solver and return the output.
        Args:
            board: Board cards (e.g., "Ah,Ks,2d").
            range_ip: IP player range.
            range_oop: OOP player range.
            pot: Pot size.
            stack: Effective stack size.
            accuracy: Target accuracy.
            max_iteration: Maximum iterations.
            thread_num: Number of threads.
        Returns:
            The raw output from the solver.
        """
        commands = [
            f"set_pot {pot}",
            f"set_effective_stack {stack}",
            f"set_board {board}",
            f"set_range_ip {range_ip}",
            f"set_range_oop {range_oop}",
            f"set_accuracy {accuracy}",
            f"set_max_iteration {max_iteration}",
            f"set_thread_num {thread_num}",
            "start_solve",
        ]
        input_str = "\n".join(commands) + "\n"
        process = subprocess.run(
            [str(self.solver_path), "-m", "holdem"],
            input=input_str,
            text=True,
            capture_output=True,
            check=True,
        )
        return process.stdout
