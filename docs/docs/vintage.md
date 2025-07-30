# **Phân tích theo Thời điểm (Vintage Analysis)**
## **Mục đích**
Phân tích Vintage là một kỹ thuật giám sát hiệu suất, trong đó các khoản vay được nhóm lại với nhau dựa trên thời điểm khởi tạo (thường là tháng), tạo thành các "thế hệ" hay "cohort". Module `vintage_analysis.py` được xây dựng để thực hiện phân tích này, giúp trả lời các câu hỏi nghiệp vụ quan trọng:

- Chất lượng danh mục có thay đổi không? Tỷ lệ vỡ nợ của các khoản vay mới cấp đang tăng hay giảm so với các khoản vay cũ?

- Hành vi khách hàng có thay đổi không? Khách hàng mới có xu hướng trả nợ chậm hơn không?

- Hiệu suất mô hình có ổn định theo thời gian không? Mô hình có dự báo tốt cho tất cả các "thế hệ" khoản vay không?

## **Quy trình Xử lý và Tính toán**
Quy trình cốt lõi được điều phối bởi hàm compute_vintage_metrics, bao gồm các bước chính sau:

### **Xác định "Tháng Khởi tạo" (Vintage Month)**
Thực thi: Hàm `prepare_vintage` chịu trách nhiệm xác định tháng khởi tạo cho mỗi khoản vay.

Logic: Vì không có ngày cấp khoản vay trực tiếp trong bộ dữ liệu application, module sử dụng một phương pháp ước tính thông minh: nó lấy ngày trả góp theo lịch sớm nhất (`DAYS_INSTALMENT.min()`) trong file `installments_payments.csv` làm ngày đại diện cho việc khởi tạo khoản vay. Ngày này sau đó được chuyển đổi thành định dạng tháng (YYYY-MM).

### **Tính toán các Chỉ số Hành vi**
Thực thi: Hàm `compute_payment_behaviour` tính toán các chỉ số phản ánh hành vi trả nợ của khách hàng cho mỗi khoản vay.

Các chỉ số chính:

- `mean_delay_days`: Số ngày trả nợ trễ trung bình, được tính bằng `DAYS_ENTRY_PAYMENT` - `DAYS_INSTALMENT`.

- m`ean_paid_ratio`: Tỷ lệ số tiền đã trả trên số tiền phải trả trung bình, được tính bằng `AMT_PAYMENT` / `AMT_INSTALMENT`.

### **Tổng hợp theo Vintage**
Thực thi: Sau khi có thông tin về tháng khởi tạo và các chỉ số hành vi cho từng khoản vay, hàm `compute_vintage_metrics` sẽ nhóm tất cả dữ liệu theo `vintage_month` và tính toán các chỉ số tổng hợp.

Các chỉ số tổng hợp:

- `num_loans`: Tổng số khoản vay được cấp trong tháng.

- `num_customers`: Tổng số khách hàng duy nhất.

- `default_rate`: Tỷ lệ vỡ nợ trung bình (tính bằng `TARGET.mean()`).

- `mean_delay_days`: Số ngày trả trễ trung bình của tất cả các khoản vay trong vintage đó.

- `mean_paid_ratio`: Tỷ lệ trả nợ trung bình.

### **(Tùy chọn) Đánh giá Hiệu suất Mô hình theo Vintage**
Thực thi: Nếu một `pd.Series` chứa điểm số của mô hình (`pd_scores`) được cung cấp, hàm `compute_vintage_metrics` sẽ tính toán thêm chỉ số AUC cho từng tháng vintage.

Logic: Nó lặp qua từng tháng, lấy ra các khoản vay thuộc tháng đó và tính AUC giữa điểm số mô hình và nhãn TARGET thực tế.

Ý nghĩa: Tính năng này cực kỳ mạnh mẽ, giúp theo dõi xem hiệu suất của mô hình có bị suy giảm ở những "thế hệ" khoản vay gần đây hay không.

## **Các Hàm Trực quan hóa**
Module cung cấp một bộ hàm vẽ biểu đồ đa dạng để trực quan hóa kết quả phân tích:

- `plot_vint_default_rate`: Vẽ biểu đồ đường cho một chỉ số duy nhất theo thời gian (ví dụ: tỷ lệ vỡ nợ).

- `plot_vint_bar`: Vẽ biểu đồ cột, phù hợp cho các chỉ số về số lượng như `num_loans` hay `num_customers`.

- `plot_vint_behaviour`: Vẽ biểu đồ hai trục (dual-axis), lý tưởng để so sánh hai chỉ số có thang đo khác nhau như `mean_delay_days` và `mean_paid_ratio`.

## **Kết quả và Diễn giải**

Dưới đây là code mẫu để thực hiện phân tích theo thời điểm:

```python
from vintage_analysis import prepare_vintage, compute_vintage_metrics, plot_vint_default_rate

# Load data
installments = pd.read_csv("data/interim/installments_payments.csv")
application = pd.read_csv("data/raw/application_train.csv")

# Compute vintage metrics
vintage_metrics = compute_vintage_metrics(installments, application)

# Save results
vintage_metrics.to_csv("output/vintage_analysis_results/vintage_metrics.csv", index=False)

# Plot default rate
plot_vint_default_rate(vintage_metrics, save_path="output/vintage_analysis_results/vintage_default_rate.png")
```

### **Tỷ lệ Vỡ nợ theo Thời điểm**

Biểu đồ dưới đây minh họa tỷ lệ vỡ nợ trung bình của các khoản vay theo từng tháng khởi tạo (`vintage_month`):

![Tỷ lệ Vỡ nợ theo Thời điểm](../../output/vintage_analysis_results/vintage_default_rate.png)

- **Xu hướng giảm dần**: Tỷ lệ vỡ nợ trung bình giảm dần theo thời gian, phản ánh sự cải thiện trong chất lượng danh mục. Điều này có thể là kết quả của việc áp dụng các chính sách cấp khoản vay chặt chẽ hơn hoặc cải tiến trong mô hình dự báo rủi ro.
- **Tháng gần đây**: Các tháng gần đây có tỷ lệ vỡ nợ thấp hơn đáng kể, cho thấy sự thành công của các chiến lược quản lý rủi ro.
- **Ý nghĩa nghiệp vụ**: Xu hướng này giúp củng cố niềm tin vào khả năng kiểm soát rủi ro của tổ chức, đồng thời tạo cơ sở để mở rộng danh mục trong tương lai.

### **Hành vi Trả nợ của Khách hàng**

Biểu đồ dưới đây so sánh số ngày trả nợ trễ trung bình (`mean_delay_days`) và tỷ lệ trả nợ trung bình (`mean_paid_ratio`) theo thời điểm khởi tạo:

![Hành vi Trả nợ của Khách hàng](../../output/vintage_analysis_results/vintage_behaviour.png)

- **Số ngày trả nợ trễ**: Số ngày trả nợ trễ trung bình giảm dần theo thời gian, cho thấy khách hàng mới có xu hướng trả nợ đúng hạn hơn. Điều này có thể phản ánh sự cải thiện trong việc lựa chọn khách hàng hoặc các biện pháp hỗ trợ trả nợ hiệu quả hơn.
- **Tỷ lệ trả nợ**: Tỷ lệ trả nợ trung bình tăng lên, cho thấy khách hàng đang trả nợ đầy đủ hơn. Đây là một tín hiệu tích cực về hành vi tài chính của khách hàng.
- **Ý nghĩa nghiệp vụ**: Sự cải thiện trong hành vi trả nợ giúp giảm thiểu rủi ro tín dụng và tăng cường hiệu quả hoạt động của tổ chức.

### **Số lượng Khoản vay và Khách hàng theo Thời điểm**

Biểu đồ dưới đây minh họa số lượng khoản vay (num_loans) và số lượng khách hàng (num_customers) theo từng tháng khởi tạo:

![Số lượng Khoản vay](../../output/vintage_analysis_results/vintage_num_loans.png)
![Số lượng Khách hàng](../../output/vintage_analysis_results/vintage_num_customers.png)

- **Số lượng khoản vay**: Số lượng khoản vay tăng lên theo thời gian, phản ánh sự mở rộng danh mục và tăng trưởng trong hoạt động kinh doanh.
- **Số lượng khách hàng**: Số lượng khách hàng cũng tăng lên, cho thấy tổ chức đang thu hút được nhiều khách hàng hơn.
- **Ý nghĩa nghiệp vụ**: Sự tăng trưởng này là một tín hiệu tích cực, nhưng cần tiếp tục giám sát để đảm bảo chất lượng danh mục không bị suy giảm.

### **Phân tích Đa Chỉ số**

Biểu đồ dưới đây tổng hợp nhiều chỉ số (`default_rate`, `mean_delay_days`, `mean_paid_ratio`) để cung cấp cái nhìn toàn diện về hiệu suất theo thời điểm:

![Phân tích Đa Chỉ số](../../output/vintage_analysis_results/vintage_multi_metric.png)

- **Tổng hợp xu hướng**: Biểu đồ đa chỉ số cung cấp cái nhìn toàn diện về hiệu suất theo thời gian, với sự cải thiện đồng thời của tỷ lệ vỡ nợ, số ngày trả nợ trễ, và tỷ lệ trả nợ.
- **Ý nghĩa nghiệp vụ**: Sự cải thiện đồng thời của các chỉ số này cho thấy danh mục đang phát triển theo hướng tích cực, đồng thời củng cố niềm tin vào hiệu quả của các chiến lược quản lý rủi ro.

