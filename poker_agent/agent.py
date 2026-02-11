from pathlib import Path
from typing import override
from pamiq_core import Agent
from torch import Tensor
from .io.texas_solver import TexasSolver
class TexasSolverAgent(Agent[Tensor, Tensor]):
    """Agent that uses TexasSolver to select actions."""
    def __init__(self, solver_path: str | Path) -> None:
        """Initialize the TexasSolverAgent.
        Args:
            solver_path: Path to the TexasSolver executable.
        """
        super().__init__()
        self.solver = TexasSolver(solver_path)
    @override
    def step(self, observation: Tensor) -> Tensor:
        """Select an action based on the observation.
        Args:
            observation: The input observation tensor.
        Returns:
            The selected action tensor.
        """
        import torch
        return torch.zeros(5, dtype=observation.dtype, device=observation.device)
