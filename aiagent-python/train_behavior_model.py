import torch
import torch.nn as nn
import json
import os

ACTIONS = ["move_forward", "jump", "look_left", "look_right", "chat", "none"]

# Simple neural net
class AIModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 64),
            nn.ReLU(),
            nn.Linear(64, len(ACTIONS))
        )

    def forward(self, x):
        return self.net(x)

def encode_input(entry):
    return [
        entry["x"], entry["y"], entry["z"],
        entry["yaw"], entry["pitch"],
        1.0 if "pickaxe" in entry["holding"].lower() else 0.0
    ]

def encode_action(action):
    return ACTIONS.index(action) if action in ACTIONS else ACTIONS.index("none")

def load_data(path):
    X, y = [], []
    with open(path, "r") as f:
        for line in f:
            entry = json.loads(line)
            X.append(encode_input(entry))
            y.append(encode_action(entry["action"]))
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

def train():
    model = AIModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    X, y = load_data("player_actions.jsonl")

    for epoch in range(20):
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch}, loss: {loss.item():.4f}")

    torch.save(model.state_dict(), "ai_model.pt")
    print("Model saved as ai_model.pt")

if __name__ == "__main__":
    train()
