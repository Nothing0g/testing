import unittest

from india_stock_llm import TinyIndiaStockLLM, build_training_windows


class TestIndiaStockLLM(unittest.TestCase):
    def test_build_training_windows(self):
        prices = [100, 101, 102, 103, 104, 105]
        X, y = build_training_windows(prices, context_window=3, horizon=1)
        self.assertEqual(X, [[100, 101, 102], [101, 102, 103], [102, 103, 104]])
        self.assertEqual(y, [103, 104, 105])

    def test_model_learns_simple_trend(self):
        prices = [100 + i for i in range(20)]
        model = TinyIndiaStockLLM(context_window=4)
        model.train(prices, epochs=600)

        pred = model.predict_next([116, 117, 118, 119])
        self.assertTrue(119.5 <= pred <= 121.5)

        horizon = model.predict_days([116, 117, 118, 119], days=3)
        self.assertEqual(len(horizon), 3)
        self.assertTrue(horizon[0] <= horizon[1] <= horizon[2])


if __name__ == "__main__":
    unittest.main()
