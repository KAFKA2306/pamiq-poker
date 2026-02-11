import pytest
import torch
from pamiq_core.testing import (
    connect_components,
    create_mock_buffer,
    create_mock_models,
)
from pytest_mock import MockerFixture
from torch.distributions import Normal
from exp.agents.curiosity.meta import MetaCuriosityAgent, create_surprisal_coefficients
from exp.data import BufferName, DataKey
from exp.models import ModelName
OBSERVATION_DIM = 16
ACTION_DIM = 4
HIDDEN_DIM = 32
DEPTH = 2
NUM_LEVELS = 3
class TestCreateSurprisalCoefficients:
    """Tests for create_surprisal_coefficients function."""
    @pytest.mark.parametrize(
        "method, num, expected",
        [
            ("maximize", 1, [1.0]),
            ("maximize", 3, [1.0, 1.0, 1.0]),
            ("maximize", 5, [1.0, 1.0, 1.0, 1.0, 1.0]),
            ("minimize", 1, [-1.0]),
            ("minimize", 3, [-1.0, -1.0, -1.0]),
            ("minimize", 5, [-1.0, -1.0, -1.0, -1.0, -1.0]),
            ("maximize_top", 1, [1.0]),
            ("maximize_top", 3, [-1.0, -1.0, 1.0]),
            ("maximize_top", 5, [-1.0, -1.0, -1.0, -1.0, 1.0]),
            ("top_only", 1, [1.0]),
            ("top_only", 3, [0.0, 0.0, 1.0]),
            ("top_only", 5, [0.0, 0.0, 0.0, 0.0, 1.0]),
        ],
    )
    def test_fixed_coefficients(self, method: str, num: int, expected: list[float]):
        """Test methods that return fixed coefficient patterns."""
        coeffs = create_surprisal_coefficients(method, num)
        assert coeffs == expected
    @pytest.mark.parametrize(
        "method, num, expected",
        [
            ("lerp_min_max", 1, [-1.0]),
            ("lerp_min_max", 3, [-1.0, 0.0, 1.0]),
            ("lerp_min_max", 5, [-1.0, -0.5, 0.0, 0.5, 1.0]),
            ("lerp_max_min", 1, [1.0]),
            ("lerp_max_min", 3, [1.0, 0.0, -1.0]),
            ("lerp_max_min", 5, [1.0, 0.5, 0.0, -0.5, -1.0]),
        ],
    )
    def test_interpolation_coefficients(
        self, method: str, num: int, expected: list[float]
    ):
        """Test methods that return interpolated coefficient patterns."""
        coeffs = create_surprisal_coefficients(method, num)
        assert len(coeffs) == num
        for i, (coef, exp) in enumerate(zip(coeffs, expected)):
            assert coef == pytest.approx(exp), f"Coefficient {i} mismatch"
    @pytest.mark.parametrize("num", [0, -1, -10])
    def test_invalid_num(self, num: int):
        """Test error raised for invalid num parameter."""
        with pytest.raises(ValueError, match="`num` must be >= 1"):
            create_surprisal_coefficients("maximize", num)
    def test_invalid_method(self):
        """Test error raised for unknown method."""
        with pytest.raises(ValueError, match="Unknown method"):
            create_surprisal_coefficients("unknown", 3)
class TestMetaCuriosityAgent:
    """Tests for the MetaCuriosityAgent class."""
    @pytest.fixture
    def models(self):
        models = {}
        for i in range(NUM_LEVELS):
            fd_model, _ = create_mock_models()
            pred_obs = torch.randn(OBSERVATION_DIM)
            hidden = torch.randn(DEPTH, HIDDEN_DIM)
            fd_model.inference_model.return_value = (pred_obs, hidden)
            models[ModelName.FORWARD_DYNAMICS + str(i)] = fd_model
        policy_value_model, _ = create_mock_models()
        action_dist = Normal(torch.zeros(ACTION_DIM), torch.ones(ACTION_DIM))
        value = torch.tensor(0.5)
        policy_hidden = torch.randn(DEPTH, HIDDEN_DIM)
        policy_value_model.inference_model.return_value = (
            action_dist,
            value,
            policy_hidden,
        )
        models[ModelName.POLICY_VALUE] = policy_value_model
        return models
    @pytest.fixture
    def buffers(self):
        buffers = {str(BufferName.POLICY): create_mock_buffer()}
        for i in range(NUM_LEVELS):
            buffers[BufferName.FORWARD_DYNAMICS + str(i)] = create_mock_buffer()
        return buffers
    @pytest.fixture
    def mock_aim_run(self, mocker: MockerFixture):
        return mocker.patch("exp.agents.curiosity.meta.get_global_run")
    @pytest.fixture
    def agent(self, models, buffers, mock_aim_run):
        coefficients = create_surprisal_coefficients("maximize_top", NUM_LEVELS)
        agent = MetaCuriosityAgent(
            surprisal_coefficients=coefficients,
            log_every_n_steps=5,
        )
        connect_components(agent, buffers=buffers, models=models)
        return agent
    def test_empty_coefficients(self):
        """Test that agent works with empty coefficients list."""
        agent = MetaCuriosityAgent(surprisal_coefficients=[])
        assert agent.num_meta_levels == 0
        assert len(agent.surprisal_coefficients) == 0
    def test_custom_coefficients(self):
        """Test agent with custom coefficients."""
        custom_coeffs = [0.5, -0.5, 1.0]
        agent = MetaCuriosityAgent(surprisal_coefficients=custom_coeffs)
        assert agent.num_meta_levels == 3
        assert agent.surprisal_coefficients == custom_coeffs
    def test_setup_step_teardown(
        self, agent: MetaCuriosityAgent, mocker: MockerFixture
    ):
        """Test the main interaction loop of the agent."""
        agent.setup()
        assert agent.policy_hidden_state is None
        assert len(agent.forward_dynamics_hiddens) == NUM_LEVELS
        assert all(h is None for h in agent.forward_dynamics_hiddens)
        assert agent.predicted_obses is None
        assert isinstance(agent.step_data_policy, dict)
        assert len(agent.step_data_fd) == NUM_LEVELS
        observation = torch.randn(OBSERVATION_DIM)
        spy_policy_collect = mocker.spy(agent.collector_policy, "collect")
        spy_fd_collects = []
        for i in range(NUM_LEVELS):
            spy = mocker.spy(agent.collectors_fd[i], "collect")
            spy_fd_collects.append(spy)
        action = agent.step(observation)
        assert action.shape == (ACTION_DIM,)
        assert agent.global_step == 1
        assert agent.predicted_obses is not None
        assert len(agent.predicted_obses) == NUM_LEVELS
        assert spy_policy_collect.call_count == 0
        for spy in spy_fd_collects:
            assert spy.call_count == 0
        observation2 = torch.randn(OBSERVATION_DIM)
        agent.step(observation2)
        assert agent.global_step == 2
        assert spy_policy_collect.call_count == 0
        for spy in spy_fd_collects:
            assert spy.call_count == 0
        observation3 = torch.randn(OBSERVATION_DIM)
        agent.step(observation3)
        assert agent.global_step == 3
        for i, spy in enumerate(spy_fd_collects):
            assert spy.call_count == 1
            fd_data = spy.call_args[0][0]
            assert DataKey.OBSERVATION in fd_data
            assert DataKey.ACTION in fd_data
            assert DataKey.HIDDEN in fd_data
            assert DataKey.TARGET in fd_data
        assert spy_policy_collect.call_count == 1
        policy_data = spy_policy_collect.call_args[0][0]
        assert DataKey.OBSERVATION in policy_data
        assert DataKey.ACTION in policy_data
        assert DataKey.ACTION_LOG_PROB in policy_data
        assert DataKey.VALUE in policy_data
        assert DataKey.REWARD in policy_data
        assert DataKey.HIDDEN in policy_data
        assert len(agent.surprisal_coefficients) == NUM_LEVELS
    def test_logging(self, agent: MetaCuriosityAgent, mock_aim_run):
        """Test metrics logging at specified intervals."""
        mock_run = mock_aim_run.return_value
        agent.setup()
        observation = torch.randn(OBSERVATION_DIM)
        for _ in range(6):
            agent.step(observation)
        mock_run.track.assert_called()
        track_calls = mock_run.track.call_args_list
        tracked_metrics = {call[1]["name"] for call in track_calls}
        for i in range(NUM_LEVELS):
            assert f"reward_{i}" in tracked_metrics
        assert "reward_sum" in tracked_metrics
        for call in track_calls:
            assert call[1]["context"]["curiosity_type"] == "meta"
    def test_save_and_load_state(self, agent: MetaCuriosityAgent, tmp_path):
        """Test state saving and loading functionality."""
        agent.setup()
        observation = torch.randn(OBSERVATION_DIM)
        agent.step(observation)
        agent.step(observation)
        agent.global_step = 42
        agent.policy_hidden_state = torch.randn(DEPTH, HIDDEN_DIM)
        for i in range(NUM_LEVELS):
            agent.forward_dynamics_hiddens[i] = torch.randn(DEPTH, HIDDEN_DIM)
        save_path = tmp_path / "agent_state"
        agent.save_state(save_path)
        assert (save_path / "policy_hidden_state.pt").exists()
        assert (save_path / "global_step").exists()
        for i in range(NUM_LEVELS):
            assert (save_path / f"forward_dynamics_hidden_{i}.pt").exists()
        coefficients = create_surprisal_coefficients("maximize_top", NUM_LEVELS)
        new_agent = MetaCuriosityAgent(
            surprisal_coefficients=coefficients,
        )
        new_agent.load_state(save_path)
        assert new_agent.global_step == 42
        assert new_agent.policy_hidden_state is not None
        assert agent.policy_hidden_state is not None
        assert torch.allclose(new_agent.policy_hidden_state, agent.policy_hidden_state)
        for i in range(NUM_LEVELS):
            assert agent.forward_dynamics_hiddens[i] is not None
            assert new_agent.forward_dynamics_hiddens[i] is not None
            assert torch.allclose(
                new_agent.forward_dynamics_hiddens[i],
                agent.forward_dynamics_hiddens[i],
            )
