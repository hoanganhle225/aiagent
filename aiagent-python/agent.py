import socket
import json
import time

HOST = "localhost"
PORT = 2001

def rl_decision(state):
    # Simple rule: nếu nhìn xuống dưới thì nhảy
    if state["pitch"] > 30:
        return "jump"
    elif "pickaxe" in state["holding"].lower():
        return "chat"
    else:
        return "look_right"

def main():
    print("[AI Agent] Connecting to Minecraft socket...")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        while True:
            try:
                s.connect((HOST, PORT))
                print("[AI Agent] Connected.")
                break
            except ConnectionRefusedError:
                print("[AI Agent] Waiting for server...")
                time.sleep(1)

        s_file = s.makefile("r")

        while True:
            try:
                # Nhận trạng thái từ Minecraft
                state_line = s_file.readline()
                if not state_line:
                    print("[AI Agent] Connection closed.")
                    break

                state = json.loads(state_line)
                print(f"[AI Agent] Received state: {state}")

                # Tính toán hành động
                action = rl_decision(state)

                # Gửi hành động trở lại Minecraft
                s.sendall((action + "\n").encode("utf-8"))

            except Exception as e:
                print(f"[AI Agent] Error: {e}")
                break

if __name__ == "__main__":
    main()
