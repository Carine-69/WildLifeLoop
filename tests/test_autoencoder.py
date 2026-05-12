import pytest
import numpy as np
import torch
from src.brain.autoencoder import LSTMAutoencoder, AnomalyDetector

AUTOENCODER_CONFIG = "configs/autoencoder.yaml"

@pytest.fixture
def detector():
    return AnomalyDetector(config_path=AUTOENCODER_CONFIG)

@pytest.fixture
def normal_sequences():
    np.random.seed(42)
    return np.random.normal(loc=0.3, scale=0.05, size=(50, 24, 5)).astype(np.float32)

@pytest.fixture
def anomalous_sequence():
    np.random.seed(99)
    return np.random.normal(loc=0.9, scale=0.3, size=(24, 5)).astype(np.float32)

def test_import():
    from src.brain.autoencoder import LSTMAutoencoder, AnomalyDetector
    assert LSTMAutoencoder is not None
    assert AnomalyDetector is not None

def test_lstm_autoencoder_forward():
    model = LSTMAutoencoder(input_size=5, hidden_size=32, num_layers=2).to("cuda")
    x = torch.randn(2, 24, 5).to("cuda")
    output = model(x)
    assert output.shape == x.shape

def test_detector_loads(detector):
    assert detector.models == {}
    assert detector.input_size == 5
    assert detector.sequence_length == 24

def test_train_individual(detector, normal_sequences):
    detector.train_individual("LION_001", normal_sequences)
    assert "LION_001" in detector.models

def test_score_normal(detector, normal_sequences):
    detector.train_individual("LION_002", normal_sequences)
    normal_seq = normal_sequences[0]
    score = detector.score("LION_002", normal_seq)
    assert isinstance(score, float)
    print(f"Normal score: {score:.4f}")

def test_score_anomaly_higher_than_normal(detector, normal_sequences, anomalous_sequence):
    detector.train_individual("LION_003", normal_sequences)
    normal_score = detector.score("LION_003", normal_sequences[0])
    anomaly_score = detector.score("LION_003", anomalous_sequence)
    print(f"Normal: {normal_score:.4f}, Anomaly: {anomaly_score:.4f}")
    assert anomaly_score > normal_score