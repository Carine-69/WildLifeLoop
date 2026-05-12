import torch
import torch.nn as nn
import numpy as np
import os
import yaml
from pathlib import Path


class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()

        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.output_layer = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        seq_len = x.shape[1]
        encoder_output, (hidden, cell) = self.encoder(x)
        decoder_input = hidden[-1].unsqueeze(1).repeat(1, seq_len, 1)
        decoder_output, _ = self.decoder(decoder_input)
        reconstruction = self.output_layer(decoder_output)
        return reconstruction


class AnomalyDetector:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.models = {}
        self.input_size = self.config['architecture']['features']
        self.hidden_size = self.config['architecture']['hidden_size']
        self.num_layers = self.config['architecture']['num_layers']
        self.sequence_length = self.config['architecture']['sequence_length']
        self.epochs = self.config['training']['epochs']
        self.learning_rate = self.config['training']['learning_rate']
        self.batch_size = self.config['training']['batch_size']
        self.anomaly_threshold = self.config['score']['anomaly_threshold']
        self.model_dir = Path(self.config['storage']['model_dir'])
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def train_individual(self, individual_id, sequences):
        model = LSTMAutoencoder(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers
        ).to("cuda")

        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()

        x = torch.tensor(sequences, dtype=torch.float32).to("cuda")

        model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            reconstruction = model(x)
            loss = criterion(reconstruction, x)
            loss.backward()
            optimizer.step()

        self.models[individual_id] = model
        self.save(individual_id)

    def score(self, individual_id, sequence):
        if individual_id not in self.models:
            return 0.0

        model = self.models[individual_id]
        model.eval()

        x = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to("cuda")

        with torch.no_grad():
            reconstruction = model(x)

        error = nn.MSELoss()(reconstruction, x).item()
        return error

    def save(self, individual_id):
        path = self.model_dir / f"{individual_id}.pt"
        torch.save(self.models[individual_id].state_dict(), path)

    def load(self, individual_id):
        path = self.model_dir / f"{individual_id}.pt"
        if path.exists():
            model = LSTMAutoencoder(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers
            ).to("cuda")
            model.load_state_dict(torch.load(path))
            self.models[individual_id] = model