# **Phân tích Độ ổn định**

## **Mục đích**
Hiệu suất của một mô hình được huấn luyện trên dữ liệu lịch sử có thể suy giảm nghiêm trọng khi được áp dụng trên dữ liệu mới trong thực tế. Nguyên nhân chính là do "sự dịch chuyển dữ liệu" (Data Drift), tức là khi các đặc tính thống kê của dữ liệu đầu vào hoặc điểm số đầu ra thay đổi.

Module `stability.py` được xây dựng để chẩn đoán và lượng hóa sự dịch chuyển này, trả lời câu hỏi: "Mô hình và dữ liệu có còn ổn định hay không?".

## **Các Chỉ số Chính và Cách Triển khai**
Module này tính toán hai loại chỉ số chính: PSI cho điểm số mô hình và các chỉ số Drift cho từng biến đầu vào.

### **Population Stability Index (PSI)**
PSI là một chỉ số đo lường mức độ thay đổi trong phân phối của một biến số giữa hai tập dữ liệu (thường là expected - kỳ vọng và actual - thực tế). Trong dự án này, nó được dùng để đo lường sự thay đổi trong phân phối điểm số (score) của mô hình.

Triển khai trong calculate_psi:

- Hàm sử dụng phương pháp chia bin dựa trên điểm phân vị (quantile-based binning). Các điểm chia (breakpoints) được xác định chỉ trên tập expected.

- Tỷ lệ phần trăm mẫu trong mỗi bin được tính cho cả hai tập dữ liệu.

- Giá trị PSI được tính bằng tổng của (expected_percents - actual_percents) * np.log(expected_percents / actual_percents).

- Một giá trị epsilon nhỏ được sử dụng để tránh lỗi chia cho zero.

Diễn giải giá trị PSI:

- PSI < 0.1: Không có sự thay đổi đáng kể (ổn định).

- 0.1 < PSI < 0.25: Có sự thay đổi nhỏ, cần theo dõi thêm.

- PSI > 0.25: Có sự thay đổi lớn, cần xem xét huấn luyện lại mô hình.

### **Feature Drift**
Chỉ số này đo lường sự thay đổi trong phân phối của từng biến đầu vào (feature). Hàm `calculate_feature_drift` thực hiện các kiểm định khác nhau cho biến số và biến phân loại.

Đối với Biến số (Numerical):

- KS-test (Kolmogorov-Smirnov): Sử dụng `scipy.stats.ks_2samp` để kiểm tra xem hai mẫu có đến từ cùng một phân phối không. Trả về giá trị thống kê KS_stat và KS_pvalue.

- Wasserstein Distance: Đo lường "khoảng cách" giữa hai phân phối.

- Phần trăm Chênh lệch Trung bình: Tính toán phần trăm chênh lệch tuyệt đối giữa giá trị trung bình của biến trên hai tập dữ liệu.

Đối với Biến Phân loại (Categorical):

- Chi-Squared Test: Xây dựng một bảng tần số chéo (contingency table) và sử dụng `scipy.stats.chi2_contingency` để kiểm tra sự độc lập giữa phân phối của biến trên hai tập dữ liệu. Trả về giá trị thống kê `Chi2_stat` và `Chi2_pvalue`.

- Jensen-Shannon Divergence: Đo lường sự khác biệt giữa hai phân phối xác suất.

Lưu ý Triển khai:

- Hàm có cơ chế lấy mẫu (sample_size) để tăng tốc độ xử lý trên các tập dữ liệu lớn.

- Các biến phân loại có quá nhiều giá trị duy nhất (max_unique_cats) sẽ được bỏ qua để tránh nhiễu và tăng hiệu suất.

### **Báo cáo Tóm tắt (stability_summary)**
Hàm stability_summary tổng hợp tất cả các kết quả phân tích vào một báo cáo cấp cao, dễ diễn giải.

Đầu ra: Một dictionary chứa:

- `Score_PSI`: Chỉ số PSI của điểm số mô hình.

- `Total_Drift_Features`: Tổng số lượng biến đầu vào được xác định là có drift (dựa trên p-value < 0.05).

- `Drift_Score_Percentage`: Tỷ lệ phần trăm các biến bị drift.

- `PSI_Severity`: Một nhãn định tính ("Low", "Medium", "High") về mức độ ổn định chung, dựa trên cả Score_PSI và tỷ lệ biến bị drift.

## **Trực quan hóa**
Module cung cấp các hàm để vẽ và lưu biểu đồ:

- `plot_psi_histogram`: Trực quan hóa sự khác biệt trong phân phối điểm số giữa hai tập dữ liệu.

- `plot_numerical_drift` & `plot_categorical_drift`: Vẽ biểu đồ so sánh phân phối cho từng biến riêng lẻ.

- `plot_top_feature_drifts`: Tự động tìm và vẽ biểu đồ cho top N biến bị drift nhiều nhất. Thay vì yêu cầu người dùng tự rà soát hàng trăm biến, chức năng này tự động làm nổi bật các biến bị dịch chuyển nhiều nhất, giúp tập trung vào việc chẩn đoán và giải quyết vấn đề.

## **Kết quả & Diễn giải thực tế**

Dưới đây là kết quả thực tế khi chạy `stability.py` trên dữ liệu dự án:

```
=== 1. Calculate PSI for model score ===
PSI (score): 0.0125
→ Phân phối điểm số mô hình giữa train và test gần như không thay đổi. Mô hình rất ổn định, không có dấu hiệu drift nghiêm trọng.

=== 2. Calculate feature drift ===
Checked features: 121, Skipped: 0
→ Tất cả các biến đều được kiểm tra, không có biến nào bị bỏ qua do thiếu dữ liệu hoặc quá nhiều giá trị unique.

=== 3. Stability summary ===
→ Tóm tắt sẽ được lưu trong file l3_stacking_stability_summary.csv và l3_stacking_feature_drift.csv.
→ Cảnh báo RuntimeWarning chỉ là thông báo kỹ thuật, không ảnh hưởng đến kết quả drift.

=== 4. Plot drift for top N features ===
Numerical drift: AMT_REQ_CREDIT_BUREAU_QRT (KS=0.264)
Numerical drift: AMT_REQ_CREDIT_BUREAU_MON (KS=0.159)
Numerical drift: AMT_CREDIT (KS=0.119)
Categorical drift: NAME_CONTRACT_TYPE (JS=0.182)
Categorical drift: WEEKDAY_APPR_PROCESS_START (JS=0.065)
Categorical drift: ORGANIZATION_TYPE (JS=0.054)
→ Các biến này có drift lớn nhất, nên kiểm tra kỹ xem có lý do business, data process, hoặc thay đổi thực tế nào không.

=== Done! All results saved in validation_results/stability_results ===
```

**Diễn giải tổng thể:**

- PSI tổng thể thấp (<0.1) ⇒ mô hình ổn định, không cần can thiệp.
- Một số biến đầu vào có drift vừa phải (KS, JS > 0.1), nhưng chưa đến mức báo động. Nếu business thấy các biến này quan trọng, nên kiểm tra lại quy trình data hoặc cập nhật mô hình.
- Nếu tỷ lệ biến drift < 10% và PSI < 0.1 ⇒ không cần can thiệp. Nếu PSI hoặc drift cao: cần kiểm tra lại nguồn dữ liệu, quy trình ETL, hoặc cân nhắc huấn luyện lại mô hình.
- Xem chi tiết các biến drift trong file `l3_stacking_feature_drift.csv` để xác định nguyên nhân (thay đổi business, lỗi nhập liệu, hoặc thay đổi quy trình).