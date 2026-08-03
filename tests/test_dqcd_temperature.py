import unittest


try:
    import torch

    from rfdetr.models.lwdetr import compute_dqcd_temperatures

    HAS_TORCH = True
except (ImportError, RuntimeError):
    HAS_TORCH = False


@unittest.skipUnless(HAS_TORCH, "PyTorch model dependencies are unavailable")
class DqcdTemperatureTests(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.gates = [
            torch.tensor([[[[0.2]]], [[[0.8]]]]),
            torch.tensor([[[[0.4]]], [[[0.6]]]]),
        ]

    def compute(self, mode):
        return compute_dqcd_temperatures(
            0.2, self.gates, mode, 2, self.device, torch.float32
        )

    def test_adaptive_uses_per_image_mean_gate(self):
        self.assertTrue(torch.allclose(self.compute("adaptive"), torch.tensor([0.16, 0.24])))

    def test_fixed_uses_base_temperature(self):
        self.assertTrue(torch.allclose(self.compute("fixed"), torch.tensor([0.2, 0.2])))

    def test_shuffled_preserves_temperature_distribution(self):
        adaptive = self.compute("adaptive").sort().values
        shuffled = self.compute("shuffled").sort().values
        self.assertTrue(torch.allclose(shuffled, adaptive))

    def test_random_stays_in_declared_range(self):
        random_temperatures = self.compute("random")
        self.assertTrue(torch.all(random_temperatures >= 0.1))
        self.assertTrue(torch.all(random_temperatures < 0.3))

    def test_unknown_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            self.compute("unsupported")


if __name__ == "__main__":
    unittest.main()
