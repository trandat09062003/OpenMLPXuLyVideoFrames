import cv2
import numpy as np
import time
import os
import sys

# Ensure UTF-8 output on Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

def create_sample_video(filename="sample_video.mp4", duration_sec=8, fps=30, width=1280, height=720):
    """
    Tạo video mẫu với các hình khối chuyển động, gradient màu và hạt nhiễu
    để dùng làm dữ liệu thử nghiệm cho thuật toán xử lý video song song.
    """
    print(f"[*] Đang tạo video mẫu '{filename}' ({width}x{height}, {fps} FPS, {duration_sec}s)...")
    total_frames = duration_sec * fps
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print(f"[!] Lỗi: Không thể khởi tạo VideoWriter cho file {filename}")
        return False

    for i in range(total_frames):
        # Tạo ảnh nền gradient chuyển màu theo thời gian
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        t = i / float(total_frames)
        # Nền màu chuyển động
        r_val = int(127 + 127 * np.sin(2 * np.pi * t))
        g_val = int(127 + 127 * np.cos(2 * np.pi * t))
        b_val = int(127 + 127 * np.sin(4 * np.pi * t))
        frame[:] = (b_val, g_val, r_val)
        
        # Vẽ các hình tròn chuyển động
        cx1 = int(width / 2 + (width / 3) * np.cos(4 * np.pi * t))
        cy1 = int(height / 2 + (height / 3) * np.sin(4 * np.pi * t))
        cv2.circle(frame, (cx1, cy1), 60, (0, 255, 255), -1)
        
        cx2 = int(width / 2 + (width / 4) * np.sin(6 * np.pi * t))
        cy2 = int(height / 2 + (height / 4) * np.cos(6 * np.pi * t))
        cv2.rectangle(frame, (cx2 - 50, cy2 - 50), (cx2 + 50, cy2 + 50), (255, 0, 128), -1)
        
        # Thêm text số thứ tự frame
        text = f"Frame: {i:04d} / {total_frames:04d} | OpenMP Parallel Demo"
        cv2.putText(frame, text, (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA)
        
        # Thêm một chút texture nhiễu ngẫu nhiên để bộ lọc xử lý bận rộn hơn
        noise = np.random.randint(0, 30, (height, width, 3), dtype=np.uint8)
        frame = cv2.add(frame, noise)
        
        out.write(frame)

    out.release()
    file_size_mb = os.path.getsize(filename) / (1024 * 1024)
    print(f"[+] Đã tạo xong video: {filename} ({total_frames} frames, {file_size_mb:.2f} MB)")
    return True

if __name__ == "__main__":
    create_sample_video()
