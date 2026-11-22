import unittest

import numpy as np
import torch
from torch import nn

from common import I20CNN, ROW_CENTRES, to_input_grid


class GradCamGeometryTest(unittest.TestCase):
    def test_parameter_count_and_shapes_match_paper(self):
        model = I20CNN().eval()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 708_866)
        logits, acts = model(torch.zeros(2, 1, 64, 60), keep=True)
        self.assertEqual(tuple(logits.shape), (2, 2))
        self.assertEqual([tuple(a.shape[2:]) for a in acts], [(27, 60), (13, 60), (6, 60)])

    def test_row_centres_equal_receptive_field_midpoints(self):
        # Use a tall input so interior rows are free of edge clipping, positive weights and
        # average pooling so every input in the receptive field gets a positive gradient.
        model = I20CNN().eval()
        model.pool = nn.AvgPool2d((2, 1))
        for block in model.blocks:
            conv, bn, _ = block
            nn.init.constant_(conv.weight, 1.0)
            bn.eval()
        x = torch.ones(1, 1, 400, 60, requires_grad=True)
        acts, _ = model.features(x)
        for layer, a in enumerate(acts):
            for row in (5, 8):
                x.grad = None
                a[0, :, row, 30].sum().backward(retain_graph=True)
                support = np.flatnonzero(x.grad[0, 0, :, 30].abs().numpy() > 0)
                midpoint = (support.min() + support.max()) / 2
                expected = ROW_CENTRES[layer][0] + row * np.diff(ROW_CENTRES[layer][:2])[0]
                self.assertAlmostEqual(midpoint, expected, places=6, msg=f"layer {layer + 1}")

    def test_mapping_preserves_columns_and_constants(self):
        for layer, centres in enumerate(ROW_CENTRES):
            cam = np.ones((1, len(centres), 60))
            np.testing.assert_allclose(to_input_grid(cam, layer), 1.0)
            ramp = np.tile(np.arange(60.0), (1, len(centres), 1))
            np.testing.assert_allclose(to_input_grid(ramp, layer)[0, 10], np.arange(60.0))


if __name__ == "__main__":
    unittest.main()
