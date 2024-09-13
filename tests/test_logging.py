import unittest
from unittest.mock import patch
import logging
from model import predict_survival  # Import the function to test

class TestLogging(unittest.TestCase):

    @patch('model.logging')  # Mock the logging module in the model
    def test_logging_for_prediction(self, mock_logging):
        # Define test input
        input_data = {'age': 8, 'sex': 'male', 'class': 'third'}
        
        # Call the function to test
        predict_survival(input_data)
        
        # Check that logging.info was called with the expected messages
        mock_logging.info.assert_any_call(f"Received input data: {input_data}")
        mock_logging.info.assert_any_call("Prediction: 1")

if __name__ == '__main__':
    unittest.main()

