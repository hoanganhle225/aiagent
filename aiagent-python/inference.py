import torch
import torch.nn as nn

class BehaviorCloningModel(nn.Module):
    def __init__(self):
        super(BehaviorCloningModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 5)  # Giữ nguyên số class như khi huấn luyện
        )

    def forward(self, x):
        return self.net(x)

# Tạo model
model = BehaviorCloningModel()

# Load state_dict
model.load_state_dict(torch.load("ai_model.pt"))
model.eval()

# Hàm dự đoán hành động
def predict_action(state):
    x = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
        action_idx = torch.argmax(logits, dim=1).item()
    return action_idx

# Test
if __name__ == "__main__":
    test_state = [-18.5, 71.0, -147.5, 0.0, 0.0, 0]
    action = predict_action(test_state)
    print("Predicted action index:", action)
