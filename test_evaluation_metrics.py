import unittest

from evaluation_metrics import threshold_for_sensitivity


class ThresholdTests(unittest.TestCase):
    def test_maintains_full_recall_without_flagging_every_case(self):
        labels = [1, 0, 1, 0]
        scores = [0.9, 0.8, 0.7, 0.1]
        cutoff = threshold_for_sensitivity(labels, scores, 1.0)
        self.assertAlmostEqual(cutoff, 0.7)
        self.assertEqual([score >= cutoff for score in scores],
                         [True, True, True, False])

    def test_keeps_highest_qualifying_cutoff_even_on_roc_plateau(self):
        cutoff = threshold_for_sensitivity(
            [1, 1, 1, 0], [0.9, 0.8, 0.7, 0.6], 2 / 3
        )
        self.assertAlmostEqual(cutoff, 0.8)

    def test_rejects_missing_class(self):
        with self.assertRaises(ValueError):
            threshold_for_sensitivity([1, 1], [0.9, 0.8], 0.95)

    def test_rejects_invalid_target(self):
        for target in (0, 1.01, float("nan")):
            with self.subTest(target=target), self.assertRaises(ValueError):
                threshold_for_sensitivity([0, 1], [0.2, 0.8], target)


if __name__ == "__main__":
    unittest.main()
