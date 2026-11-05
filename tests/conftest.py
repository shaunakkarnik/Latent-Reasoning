import pytest
import torch

from helpers import tiny_example, tiny_model


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.set_num_threads(1)
    torch.manual_seed(123)


@pytest.fixture
def model():
    result = tiny_model()
    result.oracle.set_statistics(torch.zeros(8), torch.ones(8))
    return result


@pytest.fixture
def example():
    return tiny_example()
