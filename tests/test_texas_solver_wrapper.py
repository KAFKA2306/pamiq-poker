import sys
import os
from pathlib import Path
from unittest.mock import MagicMock, patch
sys.path.append(str(Path(__file__).parent.parent))
from poker_agent.io import TexasSolver
def test_texas_solver():
    solver_path = Path("mock_solver")
    with open("mock_solver", "w") as f:
        f.write("dummy")
    try:
        solver = TexasSolver(solver_path)
        with patch("subprocess.run") as mock_run:
            mock_run.return_value.stdout = "Mock Output"
            board = "Ah,Ks,2d"
            range_ip = "AA,KK"
            range_oop = "QQ,JJ"
            pot = 10.0
            stack = 100.0
            output = solver.solve(
                board=board,
                range_ip=range_ip,
                range_oop=range_oop,
                pot=pot,
                stack=stack,
            )
            assert output == "Mock Output"
            mock_run.assert_called_once()
            args, kwargs = mock_run.call_args
            cmd_list = args[0]
            assert cmd_list == ["mock_solver", "-m", "holdem"]
            input_str = kwargs["input"]
            expected_commands = [
                "set_pot 10.0",
                "set_effective_stack 100.0",
                "set_board Ah,Ks,2d",
                "set_range_ip AA,KK",
                "set_range_oop QQ,JJ",
                "set_accuracy 0.001",
                "set_max_iteration 100",
                "set_thread_num 1",
                "start_solve",
            ]
            for cmd in expected_commands:
                assert cmd in input_str
            print("Test passed!")
    finally:
        if os.path.exists("mock_solver"):
            os.remove("mock_solver")
if __name__ == "__main__":
    test_texas_solver()
