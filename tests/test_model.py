import unittest
from model import train_model
from ingestion import load_data, preprocess_data

class TestModel(unittest.TestCase):
    def test_train_model(self):
        df = preprocess_data(load_data('data/train.csv'))
        model, accuracy = train_model(df)
        self.assertGreater(accuracy, 0.7)  # Ensuring model accuracy is acceptable

if __name__ == '__main__':
    unittest.main()

