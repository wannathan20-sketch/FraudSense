import torch
from torch import nn

from train.train_ae import move_batch_to_model_device


def test_move_batch_to_model_device_matches_first_model_parameter():
    model = nn.Linear(3, 2)
    batch = (torch.ones(4, 3),)

    moved = move_batch_to_model_device(batch, model)

    assert moved.device == next(model.parameters()).device
