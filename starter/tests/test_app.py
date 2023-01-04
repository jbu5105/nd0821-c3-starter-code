""" Tests API """
import json
import string
import random
from main import app
from fastapi.testclient import TestClient

client = TestClient(app)

def test_get():
    r = client.get("/")
    assert r.status_code == 200


def test_wrong_path():
    """ Test wrong paths """
    # Init variables
    letters = string.ascii_lowercase
    length = random.randint(1, 10)
    random_string = 'inference'
    # Generate random_string that does not match any existing paths
    while random_string == 'inference':
        random_string = ''.join(random.choice(letters) for _ in range(length))

    r = client.get(f'/{random_string}')

    assert r.status_code != 200


def test_post():
    r = client.post(
        "/inference",
        json={
            'age': 32,
            'workclass': 'State-gov',
            'fnlgt': 141297,
            'education': 'Bachelors',
            'education_num': 13,
            'marital_status': 'Married-civ-spouse',
            'occupation': 'Exec-managerial',
            'relationship': 'Wife',
            'race': 'White',
            'sex': 'Female',
            'capital_gain': 10000,
            'capital_loss': 5000,
            'hours_per_week': 35,
            'native_country': 'China'
         }
    )

    assert r.status_code == 200
