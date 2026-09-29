import subprocess
import re
import matplotlib.pyplot as plt
import os
import sys

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

def compile_code():
    print("[*] Đang biên dịch mã nguồn C++ OpenMP...")
    # Thử biên dịch với pkg-config opencv4 hoặc opencv
    cmd = "g++ -O3 -fopenmp video_openmp.cpp -o video_omp $(pkg-config --cflags --libs opencv4 2>/dev/null || pkg-config --cflags --libs opencv 2>/dev/null || echo -lopencv_core -lopencv_imgproc -lopencv_videoio -lopencv_highgui)"
    res = subprocess.run(cmd, shell=True, executable='/bin/bash' if os.name != 'nt' else None)
    if res.returncode == 0:
        print("[+] Biên dịch thành công: ./video_omp")
        return True
    else:
        print("[!] Lỗi khi biên dịch!")
        return False

def run_benchmark():
    if not os.path.exists("sample_video.mp4"):
        print("[*] Chưa thấy file sample_video.mp4, đang tự động tạo video mẫu...")
        import generate_sample_video
        generate_sample_video.create_sample_video()

    exe = "./video_omp" if os.name != 'nt' else "video_omp.exe"
    if not os.path.exists(exe):
        if not compile_code():
            print("[!] Không tìm thấy file thực thi video_omp.")
            return

    threads_list = [1, 2, 4, 8]
    times = []
    fps_list = []
    speedups = []

    print("\n=================== BẮT ĐẦU BENCHMARK ===================")
    base_time = None

    for t in threads_list:
        cmd = f"OMP_NUM_THREADS={t} {exe} sample_video.mp4 test_out.mp4 1"
        print(f"\n[*] Đang chạy với {t} luồng...")
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, executable='/bin/bash' if os.name != 'nt' else None)
        stdout, stderr = process.communicate()
        
        # Trích xuất thời gian và FPS từ output
        # Ví dụ output: Hoàn tất! Thời gian: 5.12 s | Tốc độ: 46.87 FPS
        match_time = re.search(r"Thời gian:\s*([\d\.]+)\s*s", stdout)
        match_fps = re.search(r"Tốc độ:\s*([\d\.]+)\s*FPS", stdout)

        if match_time:
            elapsed = float(match_time.group(1))
            fps = float(match_fps.group(1)) if match_fps else 0.0
            times.append(elapsed)
            fps_list.append(fps)
            if base_time is None:
                base_time = elapsed
            speedup = base_time / elapsed
            speedups.append(speedup)
            print(f" -> {t} Luồng: Thời gian = {elapsed:.2f}s | FPS = {fps:.1f} | Speedup = {speedup:.2f}x")
        else:
            print(f"[!] Output không hợp lệ hoặc lỗi:\n{stdout}\n{stderr}")

    if not times:
        print("[!] Không thu thập được dữ liệu benchmark.")
        return

    # Vẽ biểu đồ kết quả
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Biểu đồ 1: Thời gian xử lý & Speedup
    color = 'tab:blue'
    ax1.set_xlabel('Số lượng luồng (Threads)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Thời gian thực thi (giây) [Thấp hơn là tốt hơn]', color=color, fontsize=11)
    bars = ax1.bar([str(t) for t in threads_list], times, color=color, alpha=0.6, width=0.4, label='Thời gian (s)')
    ax1.tick_params(axis='y', labelcolor=color)
    for bar in bars:
        height = bar.get_height()
        ax1.annotate(f'{height:.2f}s',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3),  
                     textcoords="offset points",
                     ha='center', va='bottom', fontweight='bold')

    ax1_twin = ax1.twinx()
    color = 'tab:red'
    ax1_twin.set_ylabel('Tốc độ tăng tốc (Speedup - Lần)', color=color, fontsize=11)
    ax1_twin.plot([str(t) for t in threads_list], speedups, color=color, marker='o', linewidth=2.5, label='Speedup')
    ax1_twin.tick_params(axis='y', labelcolor=color)
    for i, txt in enumerate(speedups):
        ax1_twin.annotate(f'{txt:.2f}x', (str(threads_list[i]), speedups[i]), textcoords="offset points", xytext=(0,10), ha='center', fontweight='bold', color='darkred')

    ax1.set_title('Thời gian xử lý & Tăng tốc (Speedup)', fontsize=13, fontweight='bold')
    ax1.grid(axis='y', linestyle='--', alpha=0.5)

    # Biểu đồ 2: Tốc độ khung hình (FPS)
    ax2.set_xlabel('Số lượng luồng (Threads)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Khung hình / Giây (FPS) [Cao hơn là tốt hơn]', color='tab:green', fontsize=11)
    bars2 = ax2.bar([str(t) for t in threads_list], fps_list, color='tab:green', alpha=0.7, width=0.4)
    ax2.tick_params(axis='y', labelcolor='tab:green')
    for bar in bars2:
        height = bar.get_height()
        ax2.annotate(f'{height:.1f} FPS',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3),
                     textcoords="offset points",
                     ha='center', va='bottom', fontweight='bold')
    ax2.set_title('Tốc độ khung hình xử lý (FPS)', fontsize=13, fontweight='bold')
    ax2.grid(axis='y', linestyle='--', alpha=0.5)

    plt.suptitle('ĐÁNH GIÁ HIỆU NĂNG XỬ LÝ VIDEO SONG SONG VỚI OPENMP', fontsize=14, fontweight='bold')
    plt.tight_layout()
    chart_file = "benchmark_result.png"
    plt.savefig(chart_file, dpi=300)
    print(f"\n[+] Đã lưu biểu đồ kết quả tại: {chart_file}")

if __name__ == "__main__":
    run_benchmark()
