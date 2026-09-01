import torch
import torch.nn as nn


class BiLSTMIntentPredictor(nn.Module):
    """
    Input  : (batch, seq_len, 5)  — x1, y1, x2, y2, vehicle_speed (normalised)
    Output : (batch, 1)           — raw logit; apply sigmoid for probability
    """

    def __init__(self, input_dim: int = 5, proj_dim: int = 64,
                 hidden_dim: int = 128, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, proj_dim),
            nn.ReLU(),
        )
        self.bilstm = nn.LSTM(
            input_size=proj_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=True,
            batch_first=True,
        )
        # 128 hidden * 2 directions = 256
        self.head = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, 5)
        projected = self.input_proj(x)               # (batch, seq_len, 64)
        out, _ = self.bilstm(projected)               # (batch, seq_len, 256)
        last = out[:, -1, :]                          # (batch, 256)
        return self.head(last)                        # (batch, 1)
