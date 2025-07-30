# **Kiểm định theo Thời gian (Backtesting)**

Backtesting là một phương pháp kiểm tra hiệu suất của mô hình dự báo trong điều kiện thực tế, bằng cách giả lập cách mô hình sẽ được sử dụng qua các giai đoạn thời gian. Mục tiêu chính là đảm bảo rằng mô hình có thể duy trì hiệu suất ổn định và thích ứng với sự thay đổi của dữ liệu theo thời gian. Đây là một bước quan trọng trong việc đánh giá độ tin cậy và khả năng ứng dụng của mô hình trong môi trường thực tế.

## **Cảnh báo Quan trọng**

Cảnh báo: Chức năng backtesting trong module `backtesting.py` yêu cầu một cột `APP_DAT`E có tính chất chuỗi thời gian để có thể thực hiện việc chia dữ liệu theo các cửa sổ thời gian.

Lý do & Giải pháp:

Bộ dữ liệu application gốc không chứa trường thông tin về ngày nộp đơn thực tế. Để giải quyết vấn đề này và tái tạo lại một trục thời gian hợp lý cho việc kiểm định, chúng tôi đã quyết định sử dụng cột `DAYS_DECISION` (số ngày ra quyết định so với một ngày mốc) từ file `previous_application.csv` làm biến đại diện (proxy) cho ngày nộp đơn. Giả định ở đây là ngày ra quyết định cho khoản vay trước đó có tương quan chặt chẽ với thời điểm khách hàng nộp đơn cho khoản vay hiện tại.

Tuy nhiên, mặc dù cột `DAYS_DECISION` đã được chọn và thực hiện nghiệp vụ backtesting, kết quả cho thấy sai lệch khá lớn với mô hình, cho nên chức năng backtesting tạm thời không được thực hiện cho tới khi có trục thời gian phù hợp.

## **Mục đích và Triết lý**
Backtesting là bài "stress test" tiêu chuẩn vàng cho các mô hình sẽ được sử dụng theo thời gian. Thay vì một lần chia train-test tĩnh, backtesting giả lập lại chính xác cách mô hình sẽ được sử dụng trong thực tế: huấn luyện trên dữ liệu quá khứ để dự đoán cho tương lai gần.

Quy trình này giúp trả lời các câu hỏi cốt lõi:

- Mô hình có duy trì được hiệu suất ổn định qua các giai đoạn biến động của thị trường không?
- Quy trình huấn luyện và lựa chọn biến có đủ mạnh mẽ để thích ứng với dữ liệu mới không?

## **Quy trình Backtesting**
Quy trình được điều phối bởi hàm `run_backtesting`.

## **Phương pháp Cửa sổ Trượt (Rolling Window)**
Đây là trái tim của backtesting, được triển khai trong hàm `_generate_rolling_splits`.

Logic: Hàm này trượt một "cửa sổ" dọc theo trục thời gian (`APP_DATE`). Tại mỗi vị trí, nó sẽ định nghĩa một tập huấn luyện và một tập kiểm tra.

- `window_train_months`: Độ dài của khoảng thời gian dùng để huấn luyện (ví dụ: 18 tháng).

- `test_horizon_months`: Độ dài của khoảng thời gian ngay sau đó, dùng để kiểm tra (ví dụ: 3 tháng).

Ví dụ:

Lần chạy 1: Huấn luyện trên dữ liệu từ tháng 1/2015 - 6/2016 (window=18), kiểm tra trên tháng 7/2016 - 9/2016 (horizon=3).

Lần chạy 2: Trượt cửa sổ đi 3 tháng. Huấn luyện trên tháng 4/2015 - 9/2016, kiểm tra trên tháng 10/2016 - 12/2016.

... Quá trình tiếp tục cho đến khi hết dữ liệu.

## **Vòng lặp Kiểm định**
Với mỗi cặp (tập train, tập test) được tạo ra từ cửa sổ trượt, hàm `run_backtesting` sẽ thực hiện lại toàn bộ quy trình mô hình hóa:

Chạy lại Pipeline Dữ liệu: Tải, hợp nhất, tạo biến, và mã hóa lại toàn bộ dữ liệu chỉ trên tập train của cửa sổ hiện tại.

Huấn luyện lại Mô hình Stacking: Huấn luyện lại từ đầu toàn bộ kiến trúc Stacking 3 lớp trên dữ liệu train của cửa sổ đó.

Dự đoán và Đánh giá: Mô hình vừa được huấn luyện sẽ được dùng để dự đoán trên tập test của cửa sổ. Các chỉ số hiệu suất (AUC, KS...) và độ ổn định (Drift Score) sẽ được tính toán và ghi lại cho giai đoạn này.

Lưu trữ: Tất cả các "hiện vật" (artifacts) như mô hình, dự đoán, và báo cáo của từng giai đoạn đều được lưu lại.

## **Kết quả Đầu ra và Trực quan hóa**
Báo cáo Chi tiết: Kết quả cuối cùng là một file `backtest_metrics.csv`, chứa một dòng cho mỗi giai đoạn thời gian, ghi lại các chỉ số hiệu suất và độ ổn định tương ứng.

Biểu đồ Xu hướng: Hàm `_generate_performance_plots` sẽ tự động vẽ một biểu đồ tổng hợp, thể hiện xu hướng của AUC, KS, và Drift Score qua các giai đoạn thời gian.