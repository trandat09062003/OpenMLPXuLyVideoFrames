# Xử Lý Video Song Song Theo Từng Frame Sử Dụng OpenMP và OpenCV

Dự án nghiên cứu và triển khai bài toán **xử lý video song song trên đa nhân CPU** (Parallel Video Frame Processing) bằng **C++ kết hợp OpenMP và thư viện OpenCV**. Dự án hỗ trợ cả việc chạy bằng mã nguồn C++ qua dòng lệnh (CLI/Makefile) và chạy trực quan qua **Jupyter Notebook (`.ipynb`)**.

---

## 1. Giới thiệu đề tài & Nguyên lý hoạt động

### 1.1. Bản chất bài toán
- Quá trình đọc (`cv::VideoCapture`) và ghi (`cv::VideoWriter`) tệp video về bản chất có tính phụ thuộc thời gian (tuần tự theo cấu trúc nén I/P/B-frame).
- Ngược lại, các phép biến đổi ảnh (như lọc đạo hàm không gian Sobel, khử nhiễu Gauss, phân ngưỡng, hiệu ứng màu) áp dụng trên từng khung hình độc lập có tính chất song song tự nhiên (**Embarrassingly Parallel**).

### 1.2. Kiến trúc xử lý theo Lô (Batch / Chunk Processing)
Để tối ưu hóa việc phân chia công việc cho các lõi CPU mà không làm tràn bộ nhớ RAM:
1. **Đọc theo mẻ (Batch)**: Đọc tuần tự một nhóm $B$ frames (mặc định $B = 64$) vào bộ đệm RAM.
2. **Xử lý song song bằng OpenMP**:
   ```cpp
   #pragma omp parallel for schedule(dynamic)
   for (int i = 0; i < current_batch_size; ++i) {
       batch_outputs[i] = process_single_frame(batch_inputs[i], filter_type);
   }
   ```
3. **Ghi đầu ra có thứ tự**: Ghi các khung hình đã qua xử lý theo đúng số thứ tự ban đầu $0 \to B-1$ vào `VideoWriter`.

---

## 2. Cấu trúc thư mục

```text
├── video_openmp.cpp          # Mã nguồn C++ chính (OpenMP parallel for + OpenCV)
├── Xu_Ly_Video_OpenMP.ipynb  # Jupyter Notebook tích hợp chạy & vẽ biểu đồ thực nghiệm
├── generate_sample_video.py  # Tạo video mẫu thử nghiệm chuẩn HD 720p 30 FPS
├── run_benchmark.py          # Script tự động đo lường và vẽ biểu đồ hiệu năng
├── setup_and_run.sh          # Script bash tự động cài đặt và chạy trên Linux
├── Makefile                  # Cấu hình biên dịch nhanh g++
└── README.md                 # Tài liệu hướng dẫn
```

---

## 3. Hướng dẫn cài đặt & Thực thi

### 3.1. Cài đặt các gói phụ thuộc (Ubuntu / Debian / Linux)
```bash
sudo apt-get update
sudo apt-get install -y build-essential libomp-dev libopencv-dev pkg-config python3-opencv python3-matplotlib python3-numpy
```

### 3.2. Biên dịch mã nguồn C++
Sử dụng `make`:
```bash
make
```
Hoặc biên dịch thủ công với `g++`:
```bash
g++ -O3 -fopenmp -std=c++17 video_openmp.cpp -o video_omp $(pkg-config --cflags --libs opencv4 2>/dev/null || pkg-config --cflags --libs opencv)
```

### 3.3. Chạy chương trình
1. **Tạo video kiểm thử mẫu**:
   ```bash
   python3 generate_sample_video.py
   ```
2. **Chạy xử lý video**:
   ```bash
   # Cú pháp: ./video_omp [video_input] [video_output] [loại_filter]
   ./video_omp sample_video.mp4 output_processed.mp4 1
   ```
3. **Đo đạc Benchmark tự động (1, 2, 4, 8 threads)**:
   ```bash
   python3 run_benchmark.py
   ```

---

## 4. Các bộ lọc hỗ trợ
- `1` - **FILTER_SOBEL_EDGES**: Trích xuất đường biên Sobel (mặc định)
- `2` - **FILTER_GAUSSIAN_BLUR**: Khử nhiễu làm mờ Gauss (tải tính toán CPU cao)
- `3` - **FILTER_CARTOON**: Hiệu ứng hoạt hình hóa kết hợp Bilateral Filter và Canny Edges
- `4` - **FILTER_GRAYSCALE**: Cân bằng lược đồ xám (Histogram Equalization)

---

## 5. Kết quả thực nghiệm

Chương trình được đo lường thực tế trên video HD 720p (180 frames):

| Cấu hình | Thời gian thực thi (s) | Tốc độ xử lý (FPS) | Hệ số tăng tốc (Speedup) |
| :--- | :---: | :---: | :---: |
| **1 Luồng (Tuần tự)** | **0.49 s** | **370.9 FPS** | **1.00x** |
| **2 Luồng (OpenMP)** | **0.35 s** | **508.4 FPS** | **1.37x** |
| **4 Luồng (OpenMP)** | **0.29 s** | **624.5 FPS** | **1.68x** |

**Nhận xét**: 
- Tốc độ tăng tốc đạt đỉnh ở mức 4 luồng với FPS tăng từ 370.9 lên 624.5 FPS, đáp ứng vượt trội yêu cầu xử lý video thời gian thực.
- Khi số luồng vượt quá số nhân vật lý hoặc chịu chi phí ghi đĩa tuần tự, hiệu năng bị giới hạn phù hợp với **Định luật Amdahl**.
