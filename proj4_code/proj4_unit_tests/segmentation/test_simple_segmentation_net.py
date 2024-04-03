
from types import SimpleNamespace

import torch
from torch import nn

from proj4_code.segmentation.simple_segmentation_net import SimpleSegmentationNet

def test_check_output_shapes():
    """ """
    num_classes = 11
    criterion=nn.CrossEntropyLoss(ignore_index=255)
    model = SimpleSegmentationNet(pretrained=False, num_classes = num_classes, criterion=criterion)

    batch_size = 5
    H = 201
    W = 201
    x = torch.rand(batch_size,3,H,W).type(torch.float32)
    y = torch.ones(batch_size,H,W).type(torch.int64) * 255

    logits, yhat, main_loss, aux_loss = model(x, y)

    assert logits.shape == (batch_size, num_classes, H, W)

    # make sure that the output shape is correct
    assert yhat.shape == (batch_size, H, W)

    assert isinstance(logits, torch.Tensor)
    assert isinstance(yhat, torch.Tensor)
    assert isinstance(main_loss, torch.Tensor)
    assert isinstance(aux_loss, torch.Tensor)

    # check loss with all ground truth set to ignore index
    # assert torch.allclose(main_loss, torch.Tensor([0.]))
    # assert torch.allclose(aux_loss, torch.Tensor([0.]))


def test_check_output_shapes_testtime():
    """When y is not provided to the model, losses should be None"""
    num_classes = 11
    criterion=nn.CrossEntropyLoss(ignore_index=255)
    model = SimpleSegmentationNet(pretrained=False, num_classes = num_classes, criterion=criterion)

    batch_size = 5
    H = 201
    W = 201
    x = torch.rand(batch_size,3,H,W).type(torch.float32)
    y = torch.ones(batch_size,H,W).type(torch.int64) * 255

    logits, yhat, main_loss, aux_loss = model(x)

    assert logits.shape == (batch_size, num_classes, H, W)

    # make sure that the output shape is correct
    assert yhat.shape == (batch_size, H, W)

    assert isinstance(logits, torch.Tensor)
    assert isinstance(yhat, torch.Tensor)

    assert main_loss is None
    assert aux_loss is None
