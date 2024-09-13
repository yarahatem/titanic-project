import unittest
from app import app

class TestAPI(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
    
    def test_predict(self):
        response = self.app.post('/predict', json={
            'Pclass': 3, 'Age': 22, 'Fare': 7.25, 'Sex_male': 1, 'Embarked_Q': 0, 'Embarked_S': 1
        })
        self.assertEqual(response.status_code, 200)

if __name__ == '__main__':
    unittest.main()

