from typing import Optional

import torch
import torch.nn as nn


class RULLSTM(nn.Module):
    """
    Remaining Useful Life (RUL) 회귀를 위한 간단한 LSTM 기반 모델.

    입력:  (batch, seq_len, input_dim)
    출력:  (batch,)  - 각 시퀀스에 대한 RUL 스칼라 예측값
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            shape (B, T, F)의 입력 시퀀스.

        Returns
        -------
        torch.Tensor
            shape (B,) 의 RUL 예측값.
        """
        # LSTM 출력: out shape (B, T, H), (h_n, c_n)는 마지막 hidden/cell state
        out, _ = self.lstm(x)
        # 마지막 타임스텝의 hidden state만 사용
        last_out = out[:, -1, :]        # (B, H)
        y_hat = self.fc(last_out)       # (B, 1)
        return y_hat.squeeze(-1)        # (B,)


def count_parameters(model: nn.Module) -> int:
    """
    학습 가능한 파라미터 개수를 반환하는 편의 함수.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
