import torch
import socket
import json
from model import BehaviorCloningModel

HOST = '127.0.0.1'
PORT = 2107

model = BehaviorCloningModel(input_dim=6, output_dim=5)
model.load_state_dict(torch.load("ai_model.pt"))
model.eval()

def predict_action(data):
    try:
        features = [
            float(data["x"]),
            float(data["y"]),
            float(data["z"]),
            float(data["yaw"]),
            float(data["pitch"]),
            1.0 if data.get("holding", "") != "[Air]" else 0.0
        ]
        state = torch.tensor(features, dtype=torch.float32)
        with torch.no_grad():
            output = model(state)
            action_idx = torch.argmax(output).item()
        return action_idx
    except Exception as e:
        print("[AI SERVER] ⚠️ Dữ liệu lỗi:", data)
        print("[AI SERVER] ❌ Lỗi khi predict:", e)
        return 0  # Trả về hành động mặc định (vd: đứng yên)



def start_server():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        print("[AI SERVER] Waiting for connections...")
        while True:
            conn, addr = s.accept()
            print(f"[AI SERVER] Connected by {addr}")
            with conn:
                try:
                    data = conn.recv(1024)
                    if not data:
                        continue
                    json_data = json.loads(data.decode())
                    action_idx = predict_action(json_data)
                    conn.sendall(str(action_idx).encode())
                except Exception as e:
                    print("[AI SERVER] ❌ Error:", e)
                    conn.sendall(str(0).encode())  # trả về hành động mặc định

if __name__ == "__main__":
    start_server()
