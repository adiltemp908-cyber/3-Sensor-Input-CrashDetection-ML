"""Padding and optimization checks; synthetic fixtures are not crash evidence."""
import copy

import torch

from crashdetect.vzcrash.inception import InceptionNet, MaskedBatchNorm


def test_masked_normalization_matches_unpadded_batchnorm():
    torch.manual_seed(4)
    x = torch.randn(3, 8, 57)
    valid = torch.ones(3, 1, 57, dtype=torch.bool)
    norm = MaskedBatchNorm(8)
    reference = torch.nn.BatchNorm1d(8)
    torch.testing.assert_close(norm(x, valid), reference(x))
    torch.testing.assert_close(norm.running_mean, reference.running_mean)
    torch.testing.assert_close(norm.running_var, reference.running_var)


def test_training_padding_does_not_change_logits_statistics_or_gradients():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    net = InceptionNet(7).train()
    padded_net = copy.deepcopy(net)
    x = torch.randn(2, 101, 7)
    lengths = torch.tensor([73, 101])
    padded = torch.randn(2, 139, 7) * 100  # Padding must be ignored, even if nonzero.
    padded[:, :101] = x
    a, b = net(x, lengths), padded_net(padded, lengths)
    torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)
    a.sum().backward()
    b.sum().backward()
    assert net.blocks[0].branches[0].weight.grad.abs().sum() > 0
    for (name, parameter), (_, padded_parameter) in zip(net.named_parameters(), padded_net.named_parameters()):
        torch.testing.assert_close(parameter.grad, padded_parameter.grad, atol=2e-5, rtol=2e-4, msg=name)
    for name, value in net.named_buffers():
        torch.testing.assert_close(value, dict(padded_net.named_buffers())[name])
