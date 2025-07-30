# **Phân tích Hiệu chuẩn**

## **Mục đích**
Module `calibration_analysis.py` cung cấp các công cụ nâng cao để phân tích và trực quan hóa độ hiệu chuẩn của mô hình phân loại nhị phân. Mục tiêu chính là trả lời câu hỏi cốt lõi: "Khi mô hình dự đoán xác suất vỡ nợ là X%, liệu trong thực tế có thực sự X% khách hàng trong nhóm đó vỡ nợ hay không?".

Một mô hình được hiệu chuẩn tốt sẽ tạo ra các dự báo xác suất đáng tin cậy, giúp các quyết định kinh doanh (như định giá rủi ro, phân bổ vốn) trở nên chính xác hơn.

### **Các Chức năng Chính**
Module này được cấu trúc xoay quanh ba chức năng chính, từ việc vẽ một biểu đồ đơn lẻ đến việc thực hiện các phân tích phức tạp.

#### **plot_calibration_curve: Trực quan hóa Độ tin cậy**
Đây là hàm cốt lõi, tạo ra một Biểu đồ Hiệu chuẩn (Calibration Plot) hay Biểu đồ Tin cậy (Reliability Diagram) rất chi tiết.

Chi tiết Triển khai:

- Hàm sử dụng `sklearn.calibration.calibration_curve` để tính toán tỷ lệ thực tế (`prob_true`) và xác suất dự báo trung bình (`prob_pred`) cho từng bin.

- Hiển thị các chỉ số chính: Tiêu đề của biểu đồ tự động bao gồm các chỉ số quan trọng như Brier score, ROC AUC, ECE (Expected Calibration Error), và MCE (Maximum Calibration Error), cung cấp một cái nhìn tổng quan nhanh chóng.

- Khoảng tin cậy (Confidence Intervals): Một tính năng nâng cao là khả năng vẽ các khoảng tin cậy 95% cho tỷ lệ thực tế trong mỗi bin. Các khoảng tin cậy này được tính bằng `scipy.stats.binomtest` và giúp đánh giá mức độ chắc chắn của kết quả, đặc biệt hữu ích cho các bin có ít mẫu.

- Biểu đồ Phân phối: Hàm có thể vẽ kèm một biểu đồ histogram phân phối của các xác suất dự báo, giúp người đọc có thêm ngữ cảnh về cách các dự đoán của mô hình được phân bổ.

#### **run_calibration_analysis: Phân tích theo Phân khúc**
Đây là hàm điều phối chính, cho phép thực hiện các kịch bản phân tích khác nhau.

Chi tiết Triển khai:

- Hàm nhận đầu vào là đường dẫn đến file dự đoán OOF và file nhãn thực tế.

- Phân tích Tổng thể: Nếu không có group_col được chỉ định, hàm sẽ chạy phân tích hiệu chuẩn trên toàn bộ tập dữ liệu.

- Phân tích theo Nhóm (Group-wise Analysis): Đây là tính năng mạnh mẽ nhất. Khi một `group_col` (ví dụ: `NAME_EDUCATION_TYPE`) được cung cấp, hàm sẽ tự động lặp qua từng giá trị duy nhất trong cột đó (ví dụ: 'Higher education', 'Secondary', ...), sau đó tạo ra một biểu đồ hiệu chuẩn và bộ chỉ số riêng cho từng nhóm khách hàng.

- Tại sao quan trọng? Phân tích này giúp kiểm tra xem mô hình có hoạt động đáng tin cậy và công bằng trên mọi phân khúc khách hàng hay không.

#### **plot_multiple_calibration_curves: So sánh các Mô hình**
Hàm này được thiết kế để so sánh trực quan độ hiệu chuẩn của nhiều mô hình khác nhau trên cùng một biểu đồ.

Chi tiết Triển khai:

- Hàm nhận đầu vào là một dictionary results, trong đó mỗi key là tên mô hình và value là một tuple chứa (`y_true`, `y_pred_prob`).

- Nó sẽ vẽ đường cong hiệu chuẩn cho mỗi mô hình với một màu khác nhau, giúp dễ dàng so sánh xem mô hình nào gần với đường "lý tưởng" nhất.

### **Ví dụ sử dụng và Diễn giải**

```python
# Ví dụ chạy phân tích hiệu chuẩn cho toàn bộ tập dữ liệu và theo nhóm
from src.validation.calibration_analysis import run_calibration_analysis

# Phân tích tổng thể
metrics_df = run_calibration_analysis(
    oof_path="models/l3_stacking/l3_extratree_oof_predictions.csv",
    target_path="data/raw/application_train.csv",
    model_name="L3_ExtraTree",
    save_dir="validation_results/calibration_results",
    show_plot=False
)

# Phân tích theo nhóm học vấn
group_metrics = run_calibration_analysis(
    oof_path="models/l3_stacking/l3_extratree_oof_predictions.csv",
    target_path="data/raw/application_train.csv",
    model_name="L3_ExtraTree",
    group_col="NAME_EDUCATION_TYPE",
    save_dir="validation_results/calibration_results",
    show_plot=False
)
```

## Diễn giải Kết quả

#### Biểu đồ Hiệu chuẩn Tổng thể
- **Đường cong hiệu chuẩn**: Đường cong màu xanh của mô hình `L3_ExtraTree` nằm rất gần với đường chéo "lý tưởng", cho thấy mô hình dự đoán xác suất rất chính xác trên toàn bộ tập dữ liệu.
- **Brier Score**: Giá trị thấp (0.0677) cho thấy sai số bình phương trung bình giữa xác suất dự đoán và kết quả thực tế là rất nhỏ, khẳng định độ tin cậy của mô hình.
- **ROC AUC**: Điểm số 0.7749 cho thấy mô hình có khả năng phân biệt tốt giữa các lớp (vỡ nợ và không vỡ nợ).
- **ECE và MCE**: Các giá trị thấp (0.0282 và 0.0684) cho thấy mô hình có độ hiệu chuẩn tốt, với sai lệch tối đa giữa xác suất dự đoán và tỷ lệ thực tế là rất nhỏ.
- **Biểu đồ histogram**: Phân phối xác suất dự đoán tập trung chủ yếu ở mức thấp, phù hợp với bản chất của bài toán rủi ro tín dụng (tỷ lệ vỡ nợ thấp).

Dưới đây là biểu đồ hiệu chuẩn:

![Biểu đồ Hiệu chuẩn Tổng thể](../../images/calibration_curve_overall.png)

#### Biểu đồ Hiệu chuẩn Theo Nhóm
- **Đường cong hiệu chuẩn**: Đường cong của nhóm `NAME_EDUCATION_TYPE = Incomplete higher` cũng gần với đường chéo "lý tưởng", nhưng có một số sai lệch nhỏ hơn so với tổng thể.
- **Brier Score**: Giá trị 0.0721 cao hơn một chút so với tổng thể, cho thấy sai số bình phương trung bình lớn hơn trong nhóm này.
- **ROC AUC**: Điểm số 0.7587 vẫn cho thấy khả năng phân biệt tốt, nhưng thấp hơn so với tổng thể.
- **ECE và MCE**: Giá trị 0.0286 và 0.0718 cho thấy sai lệch giữa xác suất dự đoán và tỷ lệ thực tế trong nhóm này lớn hơn một chút, nhưng vẫn nằm trong mức chấp nhận được.
- **Biểu đồ histogram**: Phân phối xác suất dự đoán trong nhóm này cũng tập trung ở mức thấp, nhưng có sự phân tán lớn hơn so với tổng thể.

Cần xem xét thêm các biểu đồ hiệu chuẩn khác trong cùng 1 nhóm để đánh giá toàn diện.

Dưới đây là biểu đồ hiệu chuẩn cho `NAME_EDUCATION_TYPE = Incomplete higher`:

![Biểu đồ Hiệu chuẩn Theo Nhóm](../images/calibration_curve_group.png)

#### Tổng kết
- **Hiệu chuẩn tổng thể**: Mô hình `L3_ExtraTree` được hiệu chuẩn rất tốt trên toàn bộ tập dữ liệu, với các chỉ số cho thấy độ tin cậy cao và khả năng phân biệt tốt.
- **Hiệu chuẩn theo nhóm**: Phân tích theo nhóm cho thấy sự khác biệt nhỏ trong hiệu suất hiệu chuẩn, nhấn mạnh tầm quan trọng của việc kiểm tra hiệu chuẩn trên các phân khúc khác nhau để đảm bảo tính công bằng và độ tin cậy của mô hình.
- **Hành động tiếp theo**: Để cải thiện hiệu chuẩn trong các nhóm cụ thể, có thể xem xét điều chỉnh mô hình hoặc áp dụng các kỹ thuật hiệu chuẩn bổ sung như isotonic regression hoặc Platt scaling.