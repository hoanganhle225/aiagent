import socket
import os
import re
from datetime import datetime

HOST = "0.0.0.0"
PORT = 2001
SAVE_DIR = "./environments/"

os.makedirs(SAVE_DIR, exist_ok=True)

def sanitize_filename(name):
    # Loại bỏ các ký tự không hợp lệ
    return re.sub(r'[<>:"/\\|?*\n\r]', '_', name)

def is_valid_filename(name):
    # Kiểm tra xem tên file có phần mở rộng .jsonl không
    return name.endswith(".jsonl") and len(name) < 100

def start_server():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        print(f"[SERVER] Listening on port {PORT}...")

        while True:
            conn, addr = s.accept()
            with conn:
                print(f"[SERVER] Connection from {addr}")
                try:
                    raw_name = conn.recv(1024).decode().strip()
                    filename = sanitize_filename(raw_name)

                    if not is_valid_filename(filename):
                        print(f"[SERVER] ❌ Rejected invalid filename: {filename}")
                        continue

                    full_path = os.path.join(SAVE_DIR, filename)

                    with open(full_path, "a", encoding="utf-8") as f:
                        while True:
                            data = conn.recv(4096)
                            if not data:
                                break
                            f.write(data.decode("utf-8").strip() + "\n")

                    print(f"[SERVER] ✅ Received and saved file: {filename}")
                except Exception as e:
                    print(f"[SERVER] ❌ Error handling connection from {addr}: {e}")

if __name__ == "__main__":
    start_server()
