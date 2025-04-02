import minerl
import numpy as np
from collections import defaultdict

ENVIRONMENTS = [
    'MineRLTreechop-v0',
    'MineRLNavigate-v0',
    'MineRLNavigateDense-v0',
    'MineRLNavigateExtreme-v0',
    'MineRLNavigateExtremeDense-v0',
    'MineRLObtainDiamond-v0',
    'MineRLObtainDiamondDense-v0',
    'MineRLObtainIronPickaxe-v0',
    'MineRLObtainIronPickaxeDense-v0',

    'MineRLTreechopVectorObf-v0',
    'MineRLNavigateVectorObf-v0',
    'MineRLNavigateDenseVectorObf-v0',
    'MineRLNavigateExtremeVectorObf-v0',
    'MineRLNavigateExtremeDenseVectorObf-v0',
    'MineRLObtainDiamondVectorObf-v0',
    'MineRLObtainDiamondDenseVectorObf-v0',
    'MineRLObtainIronPickaxeVectorObf-v0',
    'MineRLObtainIronPickaxeDenseVectorObf-v0',
    
    'MineRLBasaltFindCave-v0',
    'MineRLBasaltCreateVillageAnimalPen-v0',
    'MineRLBasaltMakeWaterfall-v0',
    'MineRLBasaltBuildVillageHouse-v0'
    
]

def extract_features(obs, action):
    x = obs.get("xpos", 0)
    y = obs.get("ypos", 0)
    z = obs.get("zpos", 0)
    yaw, pitch = 0.0, 0.0

    cam = action.get("camera")
    if isinstance(cam, (np.ndarray, list)) and len(cam) == 2:
        yaw, pitch = float(cam[0]), float(cam[1])

    holding = 1.0 if "mainhand" in obs.get("inventory", {}) else 0.0
    return [x, y, z, yaw, pitch, holding]

def get_label(action):
    if action.get("forward"):
        return "move_forward"
    elif action.get("back"):
        return "move_back"
    elif action.get("jump"):
        return "move_jump"
    elif action.get("left"):
        return "move_left"
    elif action.get("right"):
        return "move_right"
    elif action.get("attack"):
        return "attack"
    return "idle"

def load_data():
    X = []
    y = []
    for env_id in ENVIRONMENTS:
        print(f"[DATA] Loading from {env_id}")
        data = minerl.data.make(env_id)
        count = 0
        for trajectory in data.batch_iter(batch_size=1, num_epochs=1, seq_len=1):
            for obs, action, reward, done, next_obs in trajectory:
                try:
                    obs_features = extract_features(obs, action)
                    label = get_label(action)
                    X.append(obs_features)
                    y.append(label)
                    count += 1
                except Exception as e:
                    print(f"[WARN] Skipping sample due to error: {e}")
        print(f"[DATA] {env_id} → Loaded {count} samples")
    print(f"[DATA] Total: {len(X)} samples")
    return X, y

# Huấn luyện và lưu model
if __name__ == '__main__':
    from sklearn.preprocessing import LabelEncoder
    from sklearn.linear_model import LogisticRegression
    import joblib

    print("[TRAIN] Loading data from environments...")
    X, y = load_data()

    print("[TRAIN] Encoding labels...")
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    print("[TRAIN] Training model...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y_encoded)

    print("[SAVE] Saving model and label encoder...")
    joblib.dump(model, 'ai_model.pt')
    joblib.dump(label_encoder, 'label_encoder.pkl')

    print("[DONE] Model training complete ✅")

