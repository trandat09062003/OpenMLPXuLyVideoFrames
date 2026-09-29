#!/bin/bash
# ==============================================================================
# SCRIPT TỰ ĐỘNG THIẾT LẬP VÀ CHẠY THỬ NGHIỆM XỬ LÝ VIDEO SONG SONG BẰNG OPENMP
# ==============================================================================

set -e

echo "=== 1. Cài đặt các gói thư viện cần thiết ==="
if command -v apt-get &> /dev/null; then
    apt-get update -y
    apt-get install -y build-essential libomp-dev libopencv-dev pkg-config python3-opencv python3-matplotlib python3-numpy
fi

echo "=== 2. Tạo video kiểm thử mẫu (sample_video.mp4) ==="
python3 generate_sample_video.py

echo "=== 3. Biên dịch chương trình C++ OpenMP ==="
g++ -O3 -fopenmp -std=c++17 video_openmp.cpp -o video_omp $(pkg-config --cflags --libs opencv4 2>/dev/null || pkg-config --cflags --libs opencv 2>/dev/null || echo -lopencv_core -lopencv_imgproc -lopencv_videoio -lopencv_highgui)

echo "=== 4. Chạy Benchmark đo lường hiệu năng (1, 2, 4, 8 threads) ==="
python3 run_benchmark.py

echo "=== HOÀN TẤT! ==="
echo "Kiểm tra file kết quả:"
echo " - Video sau xử lý: test_out.mp4 hoặc output_processed.mp4"
echo " - Biểu đồ hiệu năng: benchmark_result.png"
