# **Kiểm định mô hình**

## **Mục đích và Triết lý**
Trong lĩnh vực rủi ro tín dụng, một dự đoán sai có thể dẫn đến những tổn thất tài chính đáng kể. Do đó, việc xây dựng một mô hình có chỉ số kỹ thuật cao là chưa đủ. Mô hình đó phải được thử thách qua một quy trình kiểm định toàn diện để đảm bảo nó sẵn sàng cho thế giới thực: hoạt động chính xác, ổn định trước những thay đổi của môi trường, đáng tin cậy trong mọi quyết định, và bền vững theo thời gian.

Triết lý kiểm định của dự án này là xây dựng niềm tin. Mỗi bước phân tích được thiết kế để trả lời một câu hỏi cốt lõi, qua đó cung cấp bằng chứng vững chắc về chất lượng và độ tin cậy của mô hình Project Cerberus.

## **Các Trụ cột Kiểm định**
Quy trình validation của chúng tôi được xây dựng dựa trên 5 trụ cột chính, mỗi trụ cột là một bài kiểm tra chuyên sâu cho một khía cạnh của mô hình.

### **Hiệu suất (Performance)**
Câu hỏi cốt lõi: Liệu mô hình có đủ sức mạnh để phân biệt giữa khách hàng tốt và khách hàng rủi ro không?

Phân tích chi tiết: Đây là bước kiểm tra nền tảng nhất, đánh giá hiệu suất dự báo của mô hình qua ba lăng kính:

Sức mạnh xếp hạng (Ranking Power): Chúng tôi sử dụng các chỉ số AUC, Gini, và KS để đo lường khả năng của mô hình trong việc xếp hạng khách hàng rủi ro cao hơn những khách hàng ít rủi ro. Một mô hình có AUC cao sẽ đảm bảo việc ưu tiên xử lý hồ sơ hoặc áp dụng chính sách được hiệu quả.

Độ chính xác của xác suất (Probability Accuracy): Chỉ số Brier Score được dùng để đo lường sai số giữa xác suất dự báo và kết quả thực tế. Một Brier Score thấp cho thấy xác suất mà mô hình đưa ra là đáng tin cậy, rất quan trọng cho các ứng dụng như tính toán chi phí vốn hoặc định giá rủi ro.

Giá trị tại ngưỡng quyết định (Cutoff-based Value): Các chỉ số Precision, Recall, và F1-Score giúp đánh giá hiệu quả của mô hình khi áp dụng một ngưỡng phê duyệt/từ chối cụ thể trong kinh doanh.

Công cụ: performance_metrics.py.

Xem chi tiết tại: [Đánh giá Hiệu suất](./performance.md)

### **Độ ổn định (Stability)**
Câu hỏi cốt lõi: Liệu các đặc tính của dữ liệu đầu vào và điểm số đầu ra có giữ được sự ổn định giữa môi trường phát triển và môi trường thực tế không?

Phân tích chi tiết: Bước này kiểm tra "hiện tượng dịch chuyển dữ liệu" (dataset shift), một trong những nguyên nhân hàng đầu gây suy giảm hiệu suất mô hình sau khi triển khai.

Score Drift (PSI): Chúng tôi sử dụng Population Stability Index để so sánh phân phối điểm số của mô hình trên tập huấn luyện và tập kiểm tra. Một chỉ số PSI thấp cho thấy mô hình phản ứng một cách nhất quán trên các tập dữ liệu khác nhau.

Feature Drift: Chúng tôi phân tích sự dịch chuyển trong phân phối của từng biến đầu vào quan trọng. Việc phát hiện sớm các biến bị drift giúp cảnh báo về những thay đổi trong hành vi của khách hàng hoặc quy trình thu thập dữ liệu.

Công cụ: stability.py.

Xem chi tiết tại: [Phân tích Độ ổn định](./stability.md)

### **Độ tin cậy (Calibration)**
Câu hỏi cốt lõi: Khi mô hình nói rủi ro là 10%, liệu trong thực tế có thực sự 10% khách hàng trong nhóm đó vỡ nợ không?

Phân tích chi tiết: Hiệu chuẩn là thước đo sự trung thực của mô hình. Một mô hình được hiệu chuẩn tốt sẽ tạo ra các dự báo xác suất mà các bên liên quan có thể tin tưởng và sử dụng trực tiếp để phân loại rủi ro. Chúng tôi vẽ biểu đồ hiệu chuẩn và tính toán các chỉ số như ECE (Expected Calibration Error) để lượng hóa mức độ tin cậy này. Phân tích cũng được thực hiện trên các phân khúc khách hàng khác nhau để đảm bảo mô hình công bằng và đáng tin cậy trên mọi nhóm đối tượng.

Công cụ: calibration_analysis.py.

Xem chi tiết tại: [Phân tích Hiệu chuẩn](./calibration.md)

### **Hiệu suất Nghiệp vụ (Vintage)**
Câu hỏi cốt lõi: Hiệu suất của danh mục cho vay và hành vi của khách hàng thay đổi như thế nào qua từng tháng, từng quý?

Phân tích chi tiết: Phân tích Vintage nhóm các khoản vay theo tháng khởi tạo, cho phép chúng ta theo dõi các xu hướng nghiệp vụ quan trọng. Nó giúp trả lời các câu hỏi như: "Tỷ lệ vỡ nợ của các khoản vay mới đang tăng hay giảm?", "Hành vi trả nợ của khách hàng có xấu đi không?". Đây là công cụ giám sát sức khỏe danh mục không thể thiếu.

Công cụ: vintage_analysis.py.

Xem chi tiết tại: [Vintage](./vintage.md)

### **Độ Bền vững (Backtesting)**
Câu hỏi cốt lõi: Nếu mô hình này được triển khai 2 năm trước, nó sẽ hoạt động như thế nào qua từng giai đoạn biến động của thị trường?

Phân tích chi tiết: Backtesting là bài "stress test" tiêu chuẩn vàng. Bằng cách giả lập việc huấn luyện và dự đoán trên các cửa sổ thời gian trượt trong quá khứ, chúng tôi có thể đánh giá được sự bền bỉ và khả năng thích ứng của mô hình. Một kết quả backtesting tốt cho thấy mô hình không chỉ mạnh mẽ tại một thời điểm mà còn có khả năng duy trì hiệu suất ổn định trong dài hạn.

Công cụ: backtesting.py.

Xem chi tiết tại: [Backtesting](./backtesting.md)