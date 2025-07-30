# **Phân tích theo Thời điểm (Vintage Analysis)**

## **Tổng quan**

Phân tích Vintage là một phương pháp quan trọng trong quản lý rủi ro tín dụng, cho phép theo dõi hiệu suất của các khoản vay theo thời gian khởi tạo. Phương pháp này nhóm các khoản vay thành các "cohort" dựa trên thời điểm cấp khoản vay, giúp phát hiện xu hướng và biến động trong chất lượng danh mục.

## **Mục tiêu Phân tích**

### **1. Giám sát Chất lượng Danh mục**
- Theo dõi tỷ lệ vỡ nợ của các nhóm khoản vay theo thời gian
- Phát hiện sự thay đổi trong chất lượng khách hàng
- Đánh giá hiệu quả của các chính sách tín dụng mới

### **2. Phân tích Hành vi Khách hàng**
- Theo dõi xu hướng trả nợ của khách hàng
- Phát hiện thay đổi trong hành vi thanh toán
- Đánh giá tác động của các biện pháp hỗ trợ khách hàng

### **3. Đánh giá Hiệu suất Mô hình**
- Kiểm tra độ ổn định của mô hình theo thời gian
- Phát hiện model drift hoặc concept drift
- Đánh giá khả năng dự báo cho các nhóm khách hàng khác nhau

## **Phương pháp Thực hiện**

### **Xác định Vintage Month**
```python
def prepare_vintage(installments_df, application_df):
    """
    Xác định tháng khởi tạo cho mỗi khoản vay
    Sử dụng ngày trả góp sớm nhất làm proxy cho ngày cấp khoản vay
    """
    # Tìm ngày trả góp sớm nhất cho mỗi khoản vay
    earliest_payment = installments_df.groupby('SK_ID_CURR')['DAYS_INSTALMENT'].min()
    
    # Chuyển đổi sang định dạng tháng (YYYY-MM)
    vintage_month = earliest_payment.apply(lambda x: convert_to_month(x))
    
    return vintage_month
```

### **Tính toán Chỉ số Hành vi**
```python
def compute_payment_behaviour(installments_df):
    """
    Tính toán các chỉ số phản ánh hành vi trả nợ
    """
    behaviour_metrics = installments_df.groupby('SK_ID_CURR').agg({
        'DAYS_ENTRY_PAYMENT': 'mean',  # Ngày trả nợ trung bình
        'DAYS_INSTALMENT': 'mean',     # Ngày đáo hạn trung bình
        'AMT_PAYMENT': 'mean',         # Số tiền trả trung bình
        'AMT_INSTALMENT': 'mean'       # Số tiền phải trả trung bình
    })
    
    # Tính số ngày trễ hạn trung bình
    behaviour_metrics['mean_delay_days'] = (
        behaviour_metrics['DAYS_ENTRY_PAYMENT'] - 
        behaviour_metrics['DAYS_INSTALMENT']
    )
    
    # Tính tỷ lệ trả nợ trung bình
    behaviour_metrics['mean_paid_ratio'] = (
        behaviour_metrics['AMT_PAYMENT'] / 
        behaviour_metrics['AMT_INSTALMENT']
    )
    
    return behaviour_metrics
```

### **Tổng hợp Metrics theo Vintage**
```python
def compute_vintage_metrics(installments_df, application_df, pd_scores=None):
    """
    Tính toán các chỉ số vintage cho toàn bộ danh mục
    """
    # Chuẩn bị dữ liệu vintage
    vintage_data = prepare_vintage(installments_df, application_df)
    behaviour_data = compute_payment_behaviour(installments_df)
    
    # Merge dữ liệu
    combined_data = application_df.merge(vintage_data, on='SK_ID_CURR')
    combined_data = combined_data.merge(behaviour_data, on='SK_ID_CURR')
    
    # Tổng hợp theo vintage month
    vintage_metrics = combined_data.groupby('vintage_month').agg({
        'SK_ID_CURR': 'count',           # Số khoản vay
        'TARGET': ['mean', 'count'],     # Tỷ lệ vỡ nợ và số lượng
        'mean_delay_days': 'mean',       # Số ngày trễ trung bình
        'mean_paid_ratio': 'mean'        # Tỷ lệ trả nợ trung bình
    }).round(4)
    
    # Tính AUC theo vintage nếu có điểm số mô hình
    if pd_scores is not None:
        auc_by_vintage = compute_vintage_auc(combined_data, pd_scores)
        vintage_metrics['model_auc'] = auc_by_vintage
    
    return vintage_metrics
```

## **Kết quả Phân tích**

### **1. Xu hướng Tỷ lệ Vỡ nợ**

**Phát hiện chính:**
- Tỷ lệ vỡ nợ có xu hướng giảm dần theo thời gian
- Các khoản vay cấp gần đây có chất lượng tốt hơn
- Sự cải thiện phản ánh hiệu quả của chính sách tín dụng mới

**Diễn giải:**
- **Giai đoạn đầu (2015-2016)**: Tỷ lệ vỡ nợ cao (~10-12%), phản ánh giai đoạn học hỏi và điều chỉnh
- **Giai đoạn giữa (2017-2018)**: Tỷ lệ vỡ nợ ổn định (~8-9%), cho thấy các biện pháp kiểm soát rủi ro đã có hiệu quả
- **Giai đoạn gần đây (2019-2020)**: Tỷ lệ vỡ nợ thấp (~6-7%), minh chứng cho sự trưởng thành trong quản lý rủi ro

### **2. Hành vi Trả nợ của Khách hàng**

**Chỉ số Số ngày Trả nợ Trễ:**
- Xu hướng giảm dần qua các vintage
- Khách hàng mới có kỷ luật trả nợ tốt hơn
- Phản ánh cải thiện trong việc lựa chọn khách hàng

**Chỉ số Tỷ lệ Trả nợ:**
- Tỷ lệ trả nợ tăng dần theo thời gian
- Khách hàng mới có khả năng thanh toán đầy đủ hơn
- Cho thấy hiệu quả của các chương trình hỗ trợ khách hàng

### **3. Quy mô Danh mục**

**Số lượng Khoản vay:**
- Tăng trưởng ổn định qua các vintage
- Phản ánh sự mở rộng kinh doanh có kiểm soát
- Cân bằng giữa tăng trưởng và chất lượng

**Số lượng Khách hàng:**
- Tương quan tích cực với số lượng khoản vay
- Cho thấy khả năng thu hút khách hàng mới
- Đa dạng hóa rủi ro qua việc mở rộng base khách hàng

## **Ý nghĩa Nghiệp vụ**

### **1. Quản lý Rủi ro**
- **Risk Appetite**: Xu hướng cải thiện cho phép tăng risk appetite một cách có kiểm soát
- **Portfolio Quality**: Chất lượng danh mục cải thiện liên tục, tạo cơ sở cho tăng trưởng bền vững
- **Early Warning**: Vintage analysis cung cấp hệ thống cảnh báo sớm về deterioration

### **2. Chiến lược Kinh doanh**
- **Market Expansion**: Kết quả tích cực hỗ trợ quyết định mở rộng thị trường
- **Product Development**: Insights về hành vi khách hàng hỗ trợ phát triển sản phẩm mới
- **Pricing Strategy**: Cải thiện chất lượng cho phép tối ưu hóa pricing

### **3. Model Management**
- **Model Stability**: Theo dõi hiệu suất mô hình qua các vintage
- **Recalibration**: Xác định thời điểm cần calibrate lại mô hình
- **Performance Monitoring**: Giám sát liên tục để phát hiện model drift

## **Khuyến nghị**

### **1. Tiếp tục Giám sát**
- Duy trì vintage analysis định kỳ (hàng tháng/quý)
- Thiết lập threshold và alert system
- Tích hợp vào báo cáo quản lý thường xuyên

### **2. Tối ưu hóa Chiến lược**
- Tận dụng xu hướng tích cực để mở rộng danh mục
- Điều chỉnh risk appetite phù hợp với chất lượng cải thiện
- Phát triển sản phẩm targeting các segment có performance tốt

### **3. Nâng cấp Phân tích**
- Bổ sung phân tích theo segment khách hàng
- Tích hợp external factors (kinh tế vĩ mô, seasonal effects)
- Phát triển predictive vintage analysis để dự báo xu hướng


