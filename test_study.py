import unittest

import numpy as np

from study import paired_split, without_volume


class StudyTest(unittest.TestCase):
    def test_volume_mask_preserves_price_and_source_images(self):
        images = np.full((2, 64, 60), 255, dtype=np.uint8)
        masked = without_volume(images)
        np.testing.assert_array_equal(masked[:, :52], images[:, :52])
        self.assertFalse(masked[:, 52:].any())
        self.assertTrue(images[:, 52:].all())

    def test_split_excludes_december_embargo_and_nonfinite_returns(self):
        dates = np.array(
            ["1998-01-30"] * 10 + ["1999-12-31"] * 2 + ["2000-01-31"] * 2,
            dtype="datetime64[D]",
        )
        returns = np.array([0., .1, -.1, .4, -.4, .2, -.2, .3, -.3, .5, .1, .2, np.nan, .1])
        train, validation, test = paired_split(dates, returns, 2000, 123, 0)
        self.assertEqual(len(train), 7)
        self.assertEqual(len(validation), 3)
        np.testing.assert_array_equal(np.sort(np.r_[train, validation]), np.arange(10))
        np.testing.assert_array_equal(test, [13])


if __name__ == "__main__":
    unittest.main()
