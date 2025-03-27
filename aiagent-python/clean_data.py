import os
import json

def is_valid_action(action):
    required_keys = ["x", "y", "z", "yaw", "pitch", "holding", "action"]
    return all(key in action for key in required_keys)

def clean_jsonl_files(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(".jsonl"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            with open(output_path, "w", encoding="utf-8") as out_f:
                with open(input_path, "r", encoding="utf-8") as in_f:
                    buffer = ""
                    for line_num, line in enumerate(in_f, start=1):
                        buffer += line.strip()
                        try:
                            action = json.loads(buffer)
                            if is_valid_action(action):
                                out_f.write(json.dumps(action) + "\n")
                            else:
                                print(f"[WARN] ❗ Thiếu key trong dòng {line_num} của {filename}, bỏ qua.")
                            buffer = ""
                        except json.JSONDecodeError:
                            continue
                    # Xử lý dòng cuối nếu còn buffer
                    try:
                        action = json.loads(buffer)
                        if is_valid_action(action):
                            out_f.write(json.dumps(action) + "\n")
                        else:
                            print(f"[WARN] ❗ Dòng cuối của {filename} thiếu key.")
                    except json.JSONDecodeError:
                        print(f"[ERROR] ❌ Dòng cuối của {filename} không thể phân tích JSON.")

if __name__ == "__main__":
    input_folder = "environments"  # thư mục chứa file jsonl gốc
    output_folder = "cleaned_environments"  # thư mục lưu file đã xử lý
    clean_jsonl_files(input_folder, output_folder)
    print(f"✅ Dữ liệu đã được làm sạch và lưu tại: {output_folder}")
