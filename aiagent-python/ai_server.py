import torch
import socket
import json
import joblib
from model import BehaviorCloningModel
import numpy as np

HOST = '127.0.0.1'
PORT = 2107

label_encoder = joblib.load("label_encoder.pkl")

model = BehaviorCloningModel(input_dim=6, output_dim=len(label_encoder.classes_))
model.load_state_dict(torch.load("ai_model.pt"))
model.eval()

def predict_action(data):
    try:
        yaw = float(data["yaw"])
        pitch = float(data["pitch"])

        # Clamp yaw and pitch to reasonable bounds
        yaw = max(min(yaw, 360), -360)
        pitch = max(min(pitch, 90), -90)

        features = [
            float(data["x"]),
            float(data["y"]),
            float(data["z"]),
            yaw,
            pitch,
            1.0 if data.get("holding", "") != "[Air]" else 0.0
        ]
        state = torch.tensor(features, dtype=torch.float32)
        with torch.no_grad():
            output = model(state)
            action_idx = torch.argmax(output).item()
            action_label = label_encoder.inverse_transform([action_idx])[0]
            print("[DEBUG] Model output:", output)
            print("[DEBUG] Action idx:", action_idx, "→", action_label)
            return action_label
    except Exception as e:
        print("[AI SERVER] Prediction error:", e)
        return "idle"


def start_server():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        print("[AI SERVER] Listening on port", PORT)
        while True:
            conn, addr = s.accept()
            print("[AI SERVER] Connected by", addr)
            with conn:
                while True:
                    try:
                        data = conn.recv(1024)
                        if not data:
                            break
                        json_data = json.loads(data.decode())
                        print("[AI SERVER] Received:", json_data)

                        action = predict_action(json_data)
                        #conn.sendall((action + "\n").encode())
                        response_obj = {"move": action}
                        conn.sendall((json.dumps(response_obj) + "\n").encode())

                        print("[AI SERVER] Sent action:", action)
                    except Exception as e:
                        print("[AI SERVER] Error:", e)
                        conn.sendall(b"idle\n")

if __name__ == "__main__":
    start_server()
