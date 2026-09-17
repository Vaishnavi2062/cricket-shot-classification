import torch
import torch.nn as nn

class CricketShotGRU(nn.Module):
    def __init__(self, input_size=1280, hidden_size=256, num_layers=2, num_classes=10, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        output, hidden = self.gru(x)
        last_output = output[:, -1, :]
        prediction = self.fc(last_output)
        return prediction

if __name__ == "__main__":
    model = CricketShotGRU()

    x = torch.randn(1, 24, 1280)

    output = model(x)

    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
    print("GRU model created successfully!")
