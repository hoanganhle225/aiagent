import os
import minerl
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib
from model import BehaviorCloningModel



DATASET_DIR = "datasets"
ENVIRONMENTS = [
    "MineRLTreechop-v0",
    "MineRLNavigate-v0",
    "MineRLObtainDiamond-v0",
    "MineRLObtainIronPickaxe-v0"
]

def extract_features(obs, action):
    x = obs.get("XPos", 0)
    y = obs.get("YPos", 0)
    z = obs.get("ZPos", 0)
    pitch = 0.0
    yaw = 0.0

    if "camera" in action:
        cam = action["camera"]
        if not isinstance(cam, np.ndarray):
            cam = np.array(cam)
        if cam.ndim == 3:
            pitch = cam[0][-1][1]
            yaw = cam[0][-1][0]
        elif cam.ndim == 2:
            pitch = cam[-1][1]
            yaw = cam[-1][0]

    holding = 0.0
    return [x, y, z, yaw, pitch, holding]

    

def action_to_label(action_dict):
    # Pick the first active action
    for key in ["forward", "back", "left", "right", "jump", "attack"]:
        if key in action_dict and np.any(action_dict[key]):
            return f"move_{key}" if key != "attack" else "attack"
    return "idle"

def load_data():
    X, y = [], []
    for env_name in ENVIRONMENTS:
        print(f"[DATA] Loading from: {env_name}")
        data = minerl.data.make(env_name, data_dir=DATASET_DIR)
        for obs, action, _, _, _ in data.batch_iter(batch_size=1, num_epochs=1, seq_len=1):
            obs = {k: v[0] if isinstance(v, (list, np.ndarray)) and len(v) > 0 else v for k, v in obs.items()}
            action = {k: v[0] if isinstance(v, (list, np.ndarray)) and len(v) > 0 else v for k, v in action.items()}
            features = extract_features(obs, action)
            label = action_to_label(action)
            X.append(features)
            y.append(label)
    return np.array(X), np.array(y)

if __name__ == "__main__":
    X, y = load_data()
    print(f"[DATA] Samples: {len(X)}")

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    input_dim = X.shape[1]
    output_dim = len(label_encoder.classes_)

    model = BehaviorCloningModel(input_dim=input_dim, output_dim=output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y_encoded, dtype=torch.long)

    print("[TRAIN] Starting training...")
    for epoch in range(30):
        optimizer.zero_grad()
        output = model(X_tensor)
        loss = criterion(output, y_tensor)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1} - Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), "ai_model.pt")
    joblib.dump(label_encoder, "label_encoder.pkl")
    print("[✅] Saved model to ai_model.pt and label_encoder.pkl")
