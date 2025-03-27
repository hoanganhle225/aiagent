import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder

class MinecraftDataset(Dataset):
    def __init__(self, folder_path):
        self.samples = []
        self.label_encoder = LabelEncoder()
        all_actions = []

        for file in os.listdir(folder_path):
            if not file.endswith(".jsonl"):
                continue
            with open(os.path.join(folder_path, file), 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        x = [
                            float(data.get("x", 0)),
                            float(data.get("y", 0)),
                            float(data.get("z", 0)),
                            float(data.get("yaw", 0)),
                            float(data.get("pitch", 0))
                        ]
                        holding = data.get("holding", "[Air]")
                        holding_id = hash(holding) % 10000
                        x.append(holding_id)

                        self.samples.append((x, data["action"]))
                        all_actions.append(data["action"])
                    except Exception:
                        continue

        self.label_encoder.fit(all_actions)
        self.samples = [(x, self.label_encoder.transform([y])[0]) for x, y in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

class AgentModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.net(x)

def train():
    dataset = MinecraftDataset("cleaned_environments")
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = AgentModel(input_size=6, num_classes=len(dataset.label_encoder.classes_))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(10):
        total_loss = 0
        for inputs, targets in dataloader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"[Epoch {epoch+1}] Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), "ai_model.pt")
    print("✅ Đã huấn luyện xong và lưu model tại ai_model.pt")

if __name__ == "__main__":
    train()
