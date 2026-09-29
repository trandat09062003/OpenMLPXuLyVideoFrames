/**
 * ======================================================================================
 * ĐỀ TÀI: XỬ LÝ VIDEO SONG SONG THEO TỪNG FRAME BẰNG OPENMP
 * ======================================================================================
 * Mục tiêu:
 *  1. Đọc luồng video đầu vào (mp4, avi,...).
 *  2. Chia các frame theo lô (Chunk/Batch) để xử lý song song bằng OpenMP (#pragma omp parallel for).
 *  3. Mỗi luồng (thread) xử lý độc lập một frame (lọc cạnh Sobel, Gaussian Blur, làm mịn, phân ngưỡng,...).
 *  4. Đảm bảo đúng thứ tự frame khi ghi ra video đầu ra.
 *  5. Đo đạc thời gian, tính toán FPS và Speedup (tốc độ tăng tốc) theo số luồng (1, 2, 4, 8 luồng).
 * ======================================================================================
 */

#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <omp.h>
#include <opencv2/opencv.hpp>

// Enum chọn bộ lọc xử lý ảnh trên frame
enum FilterType {
    FILTER_SOBEL_EDGES = 1,     // Phát hiện cạnh biên Sobel
    FILTER_GAUSSIAN_BLUR = 2,    // Làm mờ Gauss khử nhiễu
    FILTER_CARTOON = 3,          // Hiệu ứng hoạt hình (Bilateral Filter + Canny)
    FILTER_GRAYSCALE = 4         // Chuyển mức xám + tăng cường độ tương phản
};

/**
 * Hàm xử lý nặng trên từng frame (CPU Intensive Task)
 * Có thể điều chỉnh số vòng lặp lặp lại để mô phỏng tải tính toán nặng
 */
cv::Mat process_single_frame(const cv::Mat& input_frame, FilterType filter_type) {
    if (input_frame.empty()) return input_frame;

    cv::Mat output_frame;

    switch (filter_type) {
        case FILTER_SOBEL_EDGES: {
            // Chuyển ảnh xám -> Làm mờ nhẹ -> Tính Sobel X và Y
            cv::Mat gray, grad_x, grad_y, abs_grad_x, abs_grad_y;
            cv::cvtColor(input_frame, gray, cv::COLOR_BGR2GRAY);
            cv::GaussianBlur(gray, gray, cv::Size(3, 3), 0, 0, cv::BORDER_DEFAULT);

            cv::Sobel(gray, grad_x, CV_16S, 1, 0, 3);
            cv::Sobel(gray, grad_y, CV_16S, 0, 1, 3);

            cv::convertScaleAbs(grad_x, abs_grad_x);
            cv::convertScaleAbs(grad_y, abs_grad_y);

            cv::Mat edge;
            cv::addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, edge);
            cv::cvtColor(edge, output_frame, cv::COLOR_GRAY2BGR);
            break;
        }

        case FILTER_GAUSSIAN_BLUR: {
            // Thực hiện làm mờ kernel lớn lặp lại nhiều lần để đo hiệu năng
            cv::Mat temp = input_frame.clone();
            for (int k = 0; k < 3; ++k) {
                cv::GaussianBlur(temp, output_frame, cv::Size(15, 15), 0);
                temp = output_frame;
            }
            break;
        }

        case FILTER_CARTOON: {
            // Hiệu ứng hoạt hình: kết hợp bilateralFilter và Canny edge
            cv::Mat gray, edges, color;
            cv::cvtColor(input_frame, gray, cv::COLOR_BGR2GRAY);
            cv::medianBlur(gray, gray, 7);
            cv::adaptiveThreshold(gray, edges, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, 9, 2);

            cv::bilateralFilter(input_frame, color, 9, 75, 75);
            cv::Mat edges_bgr;
            cv::cvtColor(edges, edges_bgr, cv::COLOR_GRAY2BGR);
            cv::bitwise_and(color, edges_bgr, output_frame);
            break;
        }

        case FILTER_GRAYSCALE:
        default: {
            cv::Mat gray;
            cv::cvtColor(input_frame, gray, cv::COLOR_BGR2GRAY);
            cv::equalizeHist(gray, gray);
            cv::cvtColor(gray, output_frame, cv::COLOR_GRAY2BGR);
            break;
        }
    }

    return output_frame;
}

/**
 * Xử lý video theo cơ chế Batch / Chunk với OpenMP
 * @param input_path: đường dẫn video đầu vào
 * @param output_path: đường dẫn lưu video đầu ra (rỗng nếu chỉ benchmark không ghi đĩa)
 * @param num_threads: số luồng OpenMP (1 = tuần tự, >1 = song song)
 * @param batch_size: số lượng frame đọc vào RAM trong 1 mẻ để phân phối cho các thread
 * @param filter_type: loại bộ lọc
 * @return thời gian thực thi (giây)
 */
double process_video_openmp(
    const std::string& input_path,
    const std::string& output_path,
    int num_threads,
    int batch_size = 64,
    FilterType filter_type = FILTER_SOBEL_EDGES
) {
    cv::VideoCapture cap(input_path);
    if (!cap.isOpened()) {
        std::cerr << "Lỗi: Không thể mở video: " << input_path << std::endl;
        return -1.0;
    }

    int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    double fps = cap.get(cv::CAP_PROP_FPS);
    int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));

    cv::VideoWriter writer;
    bool write_to_disk = !output_path.empty();
    if (write_to_disk) {
        int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
        writer.open(output_path, fourcc, fps, cv::Size(width, height));
        if (!writer.isOpened()) {
            std::cerr << "Cảnh báo: Không thể mở VideoWriter, bỏ qua bước ghi đĩa." << std::endl;
            write_to_disk = false;
        }
    }

    omp_set_num_threads(num_threads);

    std::cout << "--------------------------------------------------" << std::endl;
    std::cout << "Bắt đầu xử lý với " << num_threads << " luồng OpenMP" << std::endl;
    std::cout << "Kích thước: " << width << "x" << height 
              << " | FPS gốc: " << fps 
              << " | Tổng frame: " << total_frames 
              << " | Batch size: " << batch_size << std::endl;

    double start_time = omp_get_wtime();

    int processed_frames_count = 0;
    bool is_end_of_video = false;

    while (!is_end_of_video) {
        // BƯỚC 1: Đọc tuần tự một batch gồm B frame vào vector
        std::vector<cv::Mat> batch_inputs;
        batch_inputs.reserve(batch_size);

        for (int i = 0; i < batch_size; ++i) {
            cv::Mat frame;
            if (!cap.read(frame) || frame.empty()) {
                is_end_of_video = true;
                break;
            }
            batch_inputs.push_back(frame);
        }

        if (batch_inputs.empty()) break;

        int current_batch_size = static_cast<int>(batch_inputs.size());
        std::vector<cv::Mat> batch_outputs(current_batch_size);

        // BƯỚC 2: XỬ LÝ SONG SONG TỪNG FRAME BẰNG OPENMP
        // Mỗi thread sẽ nhận một frame độc lập trong batch_inputs và ghi vào batch_outputs
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < current_batch_size; ++i) {
            batch_outputs[i] = process_single_frame(batch_inputs[i], filter_type);
        }

        // BƯỚC 3: Ghi các frame theo đúng thứ tự ban đầu ra VideoWriter
        if (write_to_disk) {
            for (int i = 0; i < current_batch_size; ++i) {
                writer.write(batch_outputs[i]);
            }
        }

        processed_frames_count += current_batch_size;
    }

    double end_time = omp_get_wtime();
    double total_time = end_time - start_time;
    double processing_fps = (total_time > 0) ? (processed_frames_count / total_time) : 0;

    std::cout << "Hoàn tất! Thời gian: " << total_time << " s"
              << " | Tốc độ: " << processing_fps << " FPS" << std::endl;

    cap.release();
    if (write_to_disk) {
        writer.release();
    }

    return total_time;
}

int main(int argc, char** argv) {
    std::string input_video = "sample_video.mp4";
    std::string output_video = "output_processed.mp4";
    int filter_choice = FILTER_SOBEL_EDGES;

    if (argc >= 2) input_video = argv[1];
    if (argc >= 3) output_video = argv[2];
    if (argc >= 4) filter_choice = std::stoi(argv[3]);

    std::cout << "========================================================\n";
    std::cout << " CHƯƠNG TRÌNH SO SÁNH HIỆU NĂNG XỬ LÝ VIDEO VỚI OPENMP\n";
    std::cout << "========================================================\n";
    std::cout << "Video đầu vào: " << input_video << std::endl;
    std::cout << "Video đầu ra : " << output_video << std::endl;
    std::cout << "Chế độ lọc   : " << filter_choice << " (1: Sobel, 2: Blur, 3: Cartoon, 4: Grayscale)\n";

    // Danh sách số luồng kiểm thử hiệu năng
    std::vector<int> thread_counts = {1, 2, 4, 8};
    int max_available_threads = omp_get_max_threads();
    std::cout << "Số lõi CPU tối đa của hệ thống: " << max_available_threads << "\n\n";

    double seq_time = 0.0;

    std::cout << "Bắt đầu chạy đo lường...\n";
    for (int threads : thread_counts) {
        if (threads > max_available_threads * 2) continue;

        // Chỉ ghi file ở lần chạy cuối cùng để tiết kiệm I/O
        std::string out_file = (threads == thread_counts.back()) ? output_video : "";
        double elapsed = process_video_openmp(input_video, out_file, threads, 64, static_cast<FilterType>(filter_choice));

        if (threads == 1) {
            seq_time = elapsed;
            std::cout << ">> [Tuần tự - 1 Thread] Thời gian: " << elapsed << " s | Speedup: 1.00x\n\n";
        } else {
            double speedup = (elapsed > 0) ? (seq_time / elapsed) : 0;
            double efficiency = (threads > 0) ? (speedup / threads) * 100.0 : 0;
            std::cout << ">> [Song song - " << threads << " Threads] Thời gian: " << elapsed 
                      << " s | Speedup: " << speedup << "x"
                      << " | Hiệu suất (Efficiency): " << efficiency << "%\n\n";
        }
    }

    std::cout << "Hoàn thành đo kiểm hiệu năng!\n";
    return 0;
}
