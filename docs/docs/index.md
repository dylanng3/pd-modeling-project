# **Giới thiệu**

## **Bối cảnh**

Trong bối cảnh ngành tài chính số đang phát triển với tốc độ vũ bão, khả năng thẩm định và ra quyết định tín dụng một cách nhanh chóng, chính xác và nhất quán đã trở thành yếu tố sống còn, quyết định lợi thế cạnh tranh của mỗi tổ chức. Các phương pháp đánh giá truyền thống, vốn phụ thuộc nhiều vào quy trình thủ công và các quy tắc kinh doanh đơn giản, ngày càng bộc lộ những hạn chế: vừa tốn kém về thời gian, vừa tiềm ẩn rủi ro từ sự chủ quan, đồng thời không có khả năng nắm bắt các mẫu hành vi phức tạp ẩn sâu trong dữ liệu.

Trước thách thức đó, dự án này được thực hiện với mục đích rõ ràng: xây dựng một hệ thống đánh giá rủi ro tín dụng thông minh, tự động và đáng tin cậy, dựa trên nền tảng của khoa học dữ liệu và các kỹ thuật học máy tiên tiến.

## **Sứ mệnh và Mục tiêu**

Sứ mệnh: Xây dựng một "bộ não" phân tích rủi ro, có khả năng học hỏi từ dữ liệu lịch sử để đưa ra những dự báo chính xác về khả năng vỡ nợ (Probability of Default) của khách hàng, qua đó trở thành một công cụ hỗ trợ đắc lực cho các chuyên viên tín dụng.

Mục tiêu:

- Kinh doanh: Giảm thiểu tỷ lệ nợ xấu, tối ưu hóa danh mục cho vay và nâng cao trải nghiệm của khách hàng thông qua việc rút ngắn thời gian phê duyệt hồ sơ.
- Kỹ thuật: Phát triển một mô hình dự báo với hiệu suất vượt trội (chỉ số AUC > 0.77), đồng thời đảm bảo tính ổn định và khả năng diễn giải, sẵn sàng cho việc tích hợp vào quy trình nghiệp vụ thực tế.

## **Kiến trúc và Phương pháp luận**

Trái tim của hệ thống là một quy trình xử lý dữ liệu và mô hình hóa (pipeline) được thiết kế một cách khoa học và có cấu trúc, bao gồm các hợp phần chính:

**Nền tảng dữ liệu toàn diện**

Dự án được xây dựng trên nền tảng bộ dữ liệu Home Credit Default Risk danh tiếng từ cuộc thi trên nền tảng Kaggle. Đây là một bộ dữ liệu cực kỳ phong phú và phức tạp, mô phỏng chân thực bài toán rủi ro tín dụng trong thực tế.
Để có được cái nhìn 360 độ về khách hàng, quy trình xử lý dữ liệu đã tổng hợp thông tin từ nhiều bảng khác nhau, bao gồm:

- Dữ liệu đơn vay chính (application_train/test)

- Lịch sử tín dụng tại các tổ chức khác (bureau)

- Hồ sơ các khoản vay đã có trong quá khứ (previous_application)

- Lịch sử trả góp (installments_payments)

**Lựa chọn Biến Thông minh với SHAP**

Thay vì các phương pháp lựa chọn biến truyền thống, dự án sử dụng SHAP (SHapley Additive exPlanations) – một kỹ thuật tân tiến dựa trên lý thuyết trò chơi để đánh giá chính xác mức độ ảnh hưởng của từng biến số. Điều này không chỉ giúp mô hình trở nên tinh gọn và hiệu quả hơn mà còn tăng cường khả năng diễn giải, giúp chúng ta hiểu "tại sao" mô hình lại đưa ra một dự đoán nhất định.

**Kiến trúc Stacking Đa lớp**

Để đạt đến giới hạn cao nhất của độ chính xác, dự án triển khai một kiến trúc Stacking Ensemble 3 lớp tinh vi. Kiến trúc này hoạt động như một "hội đồng chuyên gia", nơi mỗi mô hình là một chuyên gia với thế mạnh riêng:

- Lớp 1 (Base Models): Một đội ngũ các mô hình cơ sở mạnh mẽ nhất hiện nay, bao gồm XGBoost, LightGBM, và CatBoost, được huấn luyện song song. Quá trình huấn luyện được tối ưu hóa bằng thư viện Optuna để tự động tìm ra bộ tham số tốt nhất cho mỗi mô hình.

- Lớp 2 (Meta Models): Học từ dự đoán của các chuyên gia ở Lớp 1, các meta-model như ExtraTreesClassifier và LogisticRegression tiếp tục phân tích và đưa ra một lớp dự đoán mới, tinh chỉnh hơn.

- Lớp 3 (Final Blender): Mô hình cuối cùng tổng hợp kết quả từ Lớp 2 để đưa ra quyết định sau cùng, đảm bảo sự chính xác và ổn định tối đa.

**Quy trình Kiểm định Toàn diện**

Một mô hình mạnh mẽ không chỉ cần có độ chính xác cao mà còn phải hoạt động một cách đáng tin cậy trong mọi điều kiện. Hiểu rõ điều này, dự đã trải qua một quy trình kiểm định đa chiều và nghiêm ngặt nhất:

- Hiệu suất (Performance): Mô hình được đánh giá qua một bộ chỉ số toàn diện từ khả năng xếp hạng (AUC, Gini) đến độ chính xác của xác suất dự báo (Brier Score).

- Độ ổn định (Stability): Chúng tôi thực hiện phân tích Data Drift và PSI để đảm bảo mô hình vững vàng trước những thay đổi tự nhiên của dữ liệu theo thời gian, khẳng định rằng hiệu suất sẽ không bị suy giảm khi đối mặt với dữ liệu mới.

- Độ tin cậy (Calibration): Phân tích hiệu chuẩn được áp dụng để xác nhận rằng xác suất mà mô hình đưa ra phản ánh đúng rủi ro trong thực tế, giúp các quyết định kinh doanh trở nên an toàn hơn.

- Phân tích theo Thời điểm (Vintage Analysis): Mô hình được kiểm tra trên từng "thế hệ" khoản vay để đảm bảo hiệu suất luôn nhất quán qua các thời kỳ, bất kể những biến động của thị trường.

## **Kết quả**

Dự án đã vượt qua các mục tiêu kỹ thuật đề ra, với mô hình Stacking cuối cùng đạt được hiệu suất ấn tượng và ổn định. Đây không chỉ là một dự án phân tích dữ liệu đơn thuần, mà còn là bước đi nền tảng, minh chứng cho tiềm năng to lớn của việc ứng dụng Trí tuệ Nhân tạo vào việc hiện đại hóa quy trình quản trị rủi ro, mở ra triển vọng cho một thế hệ công cụ hỗ trợ quyết định thông minh và hiệu quả hơn.