import pytest
import torch
from torch.utils.data import TensorDataset
from exp.trainers.sampler import RandomTimeSeriesSampler
class TestRandomTimeSeriesSampler:
    @pytest.fixture
    def sample_dataset(self):
        data = torch.arange(100)
        return TensorDataset(data)
    def test_init(self, sample_dataset):
        sampler = RandomTimeSeriesSampler(
            sample_dataset, sequence_length=10, max_samples=5
        )
        assert sampler.sequence_length == 10
        assert sampler.max_samples == 5
        with pytest.raises(ValueError):
            RandomTimeSeriesSampler(
                TensorDataset(torch.arange(5)), sequence_length=10, max_samples=5
            )
    def test_len(self, sample_dataset):
        sampler1 = RandomTimeSeriesSampler(
            sample_dataset, sequence_length=10, max_samples=5
        )
        assert len(sampler1) == 5
        sampler2 = RandomTimeSeriesSampler(
            sample_dataset, sequence_length=10, max_samples=1000
        )
        assert len(sampler2) == 91
    def test_iter(self, sample_dataset):
        sampler = RandomTimeSeriesSampler(
            sample_dataset, sequence_length=5, max_samples=10
        )
        batches = list(iter(sampler))
        assert len(batches) == 10
        for batch in batches:
            assert len(batch) == 5
            assert all(0 <= idx < 100 for idx in batch)
            assert batch[-1] - batch[0] == 4
