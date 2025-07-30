# **Đánh giá Hiệu suất**

## **Mục đích**
Module `performance_metrics.py` cung cấp một bộ các hàm để tính toán các chỉ số hiệu suất định lượng và tạo các biểu đồ trực quan hóa cho các mô hình phân loại nhị phân. Các hàm được thiết kế để nhận đầu vào là giá trị thực tế (`y_true`) và điểm số xác suất (`y_score`) do mô hình dự đoán.

## **Hàm Tiện ích (_prepare_series)**
Đây là một hàm nội bộ, đóng vai trò chuẩn bị và làm sạch dữ liệu đầu vào cho tất cả các hàm tính toán khác.

Chi tiết Triển khai:

- Hàm nhận vào hai đối tượng pd.Series.
- Sử dụng một mặt nạ (mask) boolean để xác định và chỉ giữ lại các cặp quan sát mà cả `y_true` và `y_score` đều không phải là NaN.
- Chuyển đổi hai Series đã được làm sạch thành mảng numpy, một định dạng tối ưu cho các phép tính toán của sklearn và scipy.

## **Các Hàm Tính toán Chỉ số**
### **auc_gini**
Hàm `auc_gini` tính toán chỉ số Area Under ROC Curve (AUC) và chỉ số Gini.

Chi tiết Triển khai:

- Gọi hàm `roc_auc_score` của sklearn để tính AUC.
- Tính Gini theo công thức 2 * AUC - 1.
- Xử lý trường hợp đặc biệt: Nếu `y_true` chỉ chứa một lớp duy nhất (ví dụ: tất cả đều là 0), hàm sẽ trả về `AUC = 0.5` và `Gini = 0.0` để tránh lỗi.

### **ks_stat**
Hàm `ks_stat` tính toán chỉ số Kolmogorov–Smirnov (KS).

Chi tiết Triển khai:

- Các điểm chia (cuts) được xác định bằng cách sử dụng các điểm phân vị (quantile) của `y_score` (`np.percentile`).
- Dữ liệu được chia vào các bucket dựa trên các điểm chia này bằng hàm np.digitize của numpy cho hiệu suất cao.
- Dữ liệu sau đó được nhóm theo bucket và thống kê số lượng "tốt" (good) và "xấu" (bad) trong mỗi bucket.
- KS được xác định là chênh lệch tuyệt đối lớn nhất giữa phân phối tích lũy của lớp "tốt" (`cum_good_pct`) và lớp "xấu" (`cum_bad_pct`).

### **brier_score**
Hàm `brier_score` tính toán Brier Score Loss, là sai số toàn phương trung bình cho các dự báo xác suất. Nó là một wrapper đơn giản quanh hàm `brier_score_loss` của sklearn.

### **hosmer_lemeshow**
Hàm `hosmer_lemeshow` thực hiện kiểm định Hosmer–Lemeshow.

Chi tiết Triển khai:

- Tương tự ks_stat, dữ liệu được chia thành `n_groups` dựa trên điểm phân vị của `y_score`.
- Trong mỗi nhóm, hàm tính toán số lượng quan sát thực tế (obs) và số lượng kỳ vọng (exp).
- Giá trị chi-squared được tính bằng tổng của (obs - exp)^2 / denominator, trong đó denominator được tính toán cẩn thận để tránh lỗi chia cho zero.
- p-value được tính từ phân phối chi2 với bậc tự do là n_groups - 2.

### **find_optimal_cutoff**
Hàm find_optimal_cutoff tìm ra ngưỡng xác suất tối ưu để tối đa hóa chỉ số F1-Score.

Chi tiết Triển khai: Hàm sử dụng một phương pháp vector hóa để tăng hiệu suất. Thay vì lặp qua từng ngưỡng, nó tạo ra một mảng 2D preds chứa kết quả dự đoán (0/1) cho tất cả các ngưỡng cùng một lúc. Sau đó, một vòng lặp nhanh tính toán f1_score cho mỗi cột của mảng này để tìm ra chỉ số tốt nhất.

## **Các Hàm Trực quan hóa**
### **plot_roc_curve**

Hàm này có chức năng vẽ biểu đồ đường cong ROC (Receiver Operating Characteristic).

Triển khai: Nó sử dụng hàm `roc_curve` từ `sklearn.metrics` để tính toán các giá trị Tỷ lệ Dương tính Thật (True Positive Rate - tpr) và Tỷ lệ Dương tính Giả (False Positive Rate - fpr) tại nhiều ngưỡng quyết định khác nhau.

Đầu ra: Một biểu đồ matplotlib được tạo ra, vẽ tpr theo fpr. Biểu đồ này cũng bao gồm một đường chéo tham chiếu (y=x), đại diện cho hiệu suất của một mô hình ngẫu nhiên. Biểu đồ hoàn chỉnh sau đó được lưu vào file được chỉ định trong tham số filename.

![Biểu đồ đường cong ROC](../../validation_results/metrics_results/l3_extratree_oof_roc_curve.png)

### **plot_pr_curve**
Hàm này chịu trách nhiệm vẽ biểu đồ Precision-Recall.

Triển khai: Nó gọi hàm `precision_recall_curve` của `sklearn.metrics` để tính toán các cặp giá trị precision và recall tương ứng với các ngưỡng xác suất khác nhau.

Đầu ra: Một biểu đồ được tạo ra để thể hiện sự đánh đổi (trade-off) giữa Precision và Recall. Biểu đồ này đặc biệt hữu ích cho các bài toán có dữ liệu mất cân bằng. Kết quả cuối cùng được lưu vào file được chỉ định trong tham số filename.

### **plot_probability_histogram**
Hàm này tạo ra một biểu đồ histogram để trực quan hóa phân phối của các xác suất dự báo (`y_score`).

Triển khai: Nó sử dụng hàm `plt.hist` của matplotlib để chia dải xác suất (từ 0 đến 1) thành một số lượng bins có thể tùy chỉnh và đếm số lượng dự đoán rơi vào mỗi bin.

Đầu ra: Biểu đồ histogram này giúp đánh giá xem các điểm số của mô hình có xu hướng tập trung ở các giá trị cực đoan (gần 0 hoặc 1) hay trải đều. Biểu đồ được lưu vào file được chỉ định trong tham số filename.

### **plot_confusion_matrix**

Điểm nhấn: Nếu người dùng không cung cấp một threshold cụ thể, hàm sẽ tự động gọi `find_optimal_cutoff` để tìm ngưỡng tốt nhất (tối ưu hóa F1-score) và sử dụng ngưỡng đó để vẽ ma trận nhầm lẫn.

Thông tin gỡ lỗi: Hàm in ra các thông tin hữu ích như dải giá trị của `y_score`, ngưỡng được sử dụng, và phân phối của `y_pred` để giúp chẩn đoán các vấn đề (ví dụ: mô hình dự đoán tất cả là cùng một lớp).

### **Hàm Báo cáo Tổng hợp (get_perf_report)**
Đây là hàm chính, được thiết kế để người dùng gọi. Nó tuần tự gọi các hàm tính toán chỉ số riêng lẻ đã mô tả ở trên và tập hợp tất cả kết quả vào một dictionary duy nhất, được làm tròn đến các chữ số thập phân phù hợp.

## **Kết quả**
Sau khi chạy, kết quả từ `get_perf_report` sẽ được lưu dưới dạng file CSV.
Kết quả xuất hiện trong terminal:

```
AUC: 0.7749, Gini: 0.5498, KS: 0.4072, Brier: 0.0677
Main metrics: {'AUC': 0.77491, 'Gini': 0.54982, 'KS': 0.40723, 'Brier': 0.06773, 'HL_chi2': 3278.49, 'HL_p_value': 0.0, 'F1_Score': 0.0, 'Precision': 0.0, 'Recall': 0.0, 'PR_AUC': 0.26262, 'Optimal_Cutoff': np.float64(0.121), 'Max_F1': np.float64(0.3223)}

=== Confusion Matrix with Optimal Threshold ===
[INFO] Using optimal threshold: 0.121
[DEBUG] y_score range: [0.035197, 0.431060]
[DEBUG] Threshold: 0.12121212121212122
[DEBUG] Number of predictions >= threshold: 41220
[DEBUG] Number of predictions < threshold: 266291
[DEBUG] y_true distribution: [282686  24825]
[DEBUG] y_pred distribution: [266291  41220]
All plots saved.
```

**Lưu ý:** Để phân tích Population Stability Index (PSI) và drift analysis, vui lòng sử dụng module `stability.py`.