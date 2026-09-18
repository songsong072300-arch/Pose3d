"""Coarse heatmap and fine coordinate losses."""
import torch
import torch.nn.functional as F


def heatmap_loss(pred, target, peak_weight=100.0):
    weights = 1.0 + peak_weight * target
    return (F.smooth_l1_loss(pred, target, reduction="none") * weights).mean()


def spatial_soft_argmax(heatmaps, temperature=50.0):
    """Convert BxCxHxW heatmaps to differentiable BxCx2 pixel coordinates."""
    batch, channels, height, width = heatmaps.shape
    probabilities = F.softmax(
        heatmaps.reshape(batch, channels, -1) * temperature, dim=-1,
    ).reshape(batch, channels, height, width)
    xs = torch.linspace(0, width - 1, width, device=heatmaps.device,
                        dtype=heatmaps.dtype)
    ys = torch.linspace(0, height - 1, height, device=heatmaps.device,
                        dtype=heatmaps.dtype)
    x = (probabilities.sum(dim=2) * xs).sum(dim=-1)
    y = (probabilities.sum(dim=3) * ys).sum(dim=-1)
    return torch.stack((x, y), dim=-1)


def corner_coordinate_loss(pred, corners, temperature=50.0):
    height, width = pred.shape[-2:]
    scale = pred.new_tensor((width - 1, height - 1))
    predicted_corners = spatial_soft_argmax(pred, temperature=temperature)
    return F.smooth_l1_loss(predicted_corners / scale, corners / scale)


def corner_heatmap_loss(pred, target, corners, coordinate_weight=2.0):
    coarse = heatmap_loss(pred, target)
    fine = corner_coordinate_loss(pred, corners)
    return coarse + coordinate_weight * fine, coarse.detach(), fine.detach()
