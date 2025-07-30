# **Chuyển đổi dữ liệu**

Sau khi dữ liệu được hợp nhất và các biến mới được tạo ra, giai đoạn tiếp theo là chuyển đổi nó thành một định dạng ma trận số hoàn chỉnh, sạch sẽ và sẵn sàng cho việc huấn luyện mô hình. Quy trình này bao gồm hai bước chính: xử lý các giá trị bị thiếu một cách nhất quán bằng phương pháp điền mean/mode và mã hóa các biến phân loại thành dạng số bằng kỹ thuật **Target Encoding** để tận dụng tối đa thông tin từ biến mục tiêu. Cuối cùng, một bước kiểm tra chất lượng dữ liệu được thực hiện để đảm bảo không còn giá trị rỗng, vô cực hay các biến không có giá trị thông tin, khẳng định dữ liệu đã hoàn toàn hợp lệ cho mô hình hóa.

## **Xử lý Dữ liệu thiếu (Imputation)**

### **Mục đích**

Trong các bộ dữ liệu thực tế, việc thiếu dữ liệu là không thể tránh khỏi. Các giá trị bị thiếu (NaN) có thể gây ra lỗi hoặc làm giảm hiệu suất của các mô hình học máy. Module `imputation.py` cung cấp một lớp `SimpleImputer` được thiết kế để xử lý vấn đề này một cách có hệ thống, đảm bảo dữ liệu đầu vào cho mô hình luôn đầy đủ và nhất quán.

### **Vấn đề: Tại sao phải xử lý dữ liệu thiếu ?**

Hầu hết các thuật toán học máy (như **LightGBM**, **XGBoost**) đều không thể hoạt động trên tập dữ liệu có chứa giá trị NaN. Do đó, việc điền (impute) các giá trị bị thiếu là một bước tiền xử lý bắt buộc. Tuy nhiên, việc điền giá trị phải được thực hiện một cách cẩn thận để tránh "rò rỉ dữ liệu" (data leakage) – một lỗi nghiêm trọng có thể dẫn đến việc đánh giá mô hình quá lạc quan.

### **Giải pháp: Lớp SimpleImputer**

Để giải quyết vấn đề trên một cách chuyên nghiệp, dự án xây dựng một lớp `SimpleImputer` tùy chỉnh, tuân theo quy ước thiết kế của thư viện `scikit-learn` với hai phương thức chính là `fit()` và `transform()`.

#### **Phương thức fit()**

Thực hiện: Phương thức `fit(df)` nhận vào một DataFrame (thường là tập huấn luyện) và chỉ tính toán các giá trị thống kê cần thiết để điền vào chỗ trống. Nó không thay đổi dữ liệu gốc.

Cụ thể, nó sẽ:

Tính và lưu trữ giá trị trung vị (median) cho tất cả các cột số.

Tính và lưu trữ giá trị yếu vị (mode) cho tất cả các cột phân loại.

Tại sao quan trọng? Bằng cách chỉ "học" từ dữ liệu huấn luyện, chúng ta đảm bảo rằng không có bất kỳ thông tin nào từ tập kiểm tra (validation/test) bị rò rỉ vào quá trình chuẩn bị dữ liệu.


```python
# Trích đoạn từ phương thức fit() trong imputation.py
class SimpleImputer:
    def fit(self, df):
        # Lưu median cho cột số
        numeric_cols = df.select_dtypes(include=np.number).columns
        for col in numeric_cols:
            self.median_values[col] = df[col].median()
        
        # Lưu mode cho cột phân loại
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            self.mode_values[col] = df[col].mode()[0]
```

#### **Phương thức transform()**

Thực hiện: Phương thức `transform(df)` sử dụng các giá trị (median và mode) đã được lưu từ bước `fit()` để điền vào các giá trị NaN trong một DataFrame mới.

Tính nhất quán: Cùng một đối tượng imputer có thể được dùng để transform cả tập huấn luyện, tập validation và tập kiểm tra, đảm bảo rằng tất cả các tập dữ liệu đều được xử lý theo cùng một quy tắc.

### **Chiến lược Điền giá trị** 

Đối với cột số: Sử dụng giá trị trung vị (median). Median được ưu tiên hơn giá trị trung bình (mean) vì nó ít bị ảnh hưởng bởi các giá trị ngoại lệ (outliers), giúp cho việc điền giá trị trở nên ổn định và đáng tin cậy hơn.

Đối với cột phân loại: Sử dụng giá trị yếu vị (mode), tức là giá trị xuất hiện nhiều nhất trong cột. Đây là phương pháp tiêu chuẩn và phù hợp nhất cho dữ liệu không có thứ tự.

Ví dụ sử dụng:

```python
# Giả sử X_train và X_test là các DataFrame
from src.processing.imputation import SimpleImputer

# 1. Khởi tạo imputer
imputer = SimpleImputer()

# 2. "Học" các giá trị median/mode từ tập train
imputer.fit(X_train)

# 3. Áp dụng để điền giá trị cho cả tập train và test
X_train_filled = imputer.transform(X_train)
X_test_filled = imputer.transform(X_test)
```

## **Mã hóa Biến (Feature Encoding)**

### **Mục đích**

Hầu hết các mô hình học máy đều yêu cầu đầu vào là dữ liệu dạng số. Tuy nhiên, dữ liệu thực tế thường chứa rất nhiều biến phân loại (categorical variables), ví dụ như `NAME_EDUCATION_TYPE` (loại hình học vấn) hay `OCCUPATION_TYPE` (ngành nghề). Module `encoding.py` cung cấp một lớp `TargetEncoder` để chuyển đổi các biến này thành dạng số một cách thông minh, giúp mô hình có thể học được từ chúng.

### **Vấn đề: Tại sao cần Mã hóa?**

Các phương pháp mã hóa phổ biến như **One-Hot Encoding** có thể tạo ra quá nhiều cột mới khi gặp các biến có số lượng giá trị duy nhất lớn (high cardinality), dẫn đến curse of dimensionality. Để giải quyết vấn đề này, dự án sử dụng một kỹ thuật mạnh mẽ hơn là Mã hóa Mục tiêu (**Target Encoding**).

### **Giải pháp: Lớp TargetEncoder**

`TargetEncoder` là một lớp tùy chỉnh giúp thay thế mỗi giá trị của biến phân loại bằng một con số có ý nghĩa, dựa trên mối quan hệ của nó với biến mục tiêu (`TARGET`).

### **Quy trình thực hiện**

#### **Ý tưởng cốt lõi**

Ý tưởng chính là thay thế mỗi danh mục (category) bằng giá trị trung bình của biến mục tiêu (`TARGET`) tương ứng với danh mục đó.

Ví dụ: Nếu trong tập huấn luyện, tỷ lệ vỡ nợ (`TARGET=1`) của nhóm khách hàng có `NAME_EDUCATION_TYPE` là 'Higher education' là 5%, và của nhóm 'Secondary' là 10%, thì:

- 'Higher education' sẽ được mã hóa thành 0.05.

- 'Secondary' sẽ được mã hóa thành 0.10.

Bằng cách này, chúng ta đã "nén" thông tin về rủi ro của từng nhóm khách hàng vào chính biến số đó.

#### **Kỹ thuật Làm mượt (Smoothing)**

Để tránh việc mô hình bị quá khớp (overfitting), đặc biệt với các danh mục có ít mẫu, `TargetEncoder` sử dụng một kỹ thuật làm mượt (smoothing). Giá trị mã hóa cuối cùng sẽ là sự kết hợp có trọng số giữa giá trị trung bình của danh mục đó và giá trị trung bình của toàn bộ tập dữ liệu.

Tham số smoothing kiểm soát mức độ ảnh hưởng của giá trị trung bình toàn cục. smoothing càng lớn, giá trị mã hóa càng có xu hướng tiến về giá trị trung bình chung, giúp mô hình ổn định hơn.

```python
# Trích đoạn công thức smoothing trong TargetEncoder
class TargetEncoder:
    def fit_transform(self, df, target, categorical_cols):
        # ...
        averages = encoded_df.groupby(col)[target].mean()
        counts = encoded_df.groupby(col)[target].count()
        
        # Áp dụng smoothing
        global_mean = averages.mean()
        self.maps[col] = (averages * counts + self.smoothing * global_mean) / (counts + self.smoothing)
        # ...
```

#### **Quy trình fit_transform và transform**

Lớp TargetEncoder tuân thủ quy ước của scikit-learn:

- `fit_transform(df, target, categorical_cols)`: Được sử dụng trên tập huấn luyện. Nó "học" các ánh xạ (maps) từ dữ liệu huấn luyện và đồng thời áp dụng mã hóa trên chính tập đó.

- `transform(df, categorical_cols)`: Được sử dụng trên tập validation hoặc test. Nó chỉ áp dụng các ánh xạ đã học từ bước trước đó để mã hóa dữ liệu mới, đảm bảo không có sự rò rỉ thông tin. Nó cũng xử lý các giá trị mới (chưa từng thấy trong tập train) bằng cách điền giá trị trung bình của ánh xạ đã học.


#### **Ví dụ sử dụng**

```python
from src.processing.encoding import TargetEncoder

# Giả sử train_df và test_df là các DataFrame
target_col = 'TARGET'
categorical_features = ['NAME_EDUCATION_TYPE', 'OCCUPATION_TYPE']

# 1. Khởi tạo encoder
encoder = TargetEncoder(smoothing=20)

# 2. Học và áp dụng trên tập train
train_encoded = encoder.fit_transform(train_df, target_col, categorical_features)

# 3. Chỉ áp dụng trên tập test
test_encoded = encoder.transform(test_df, categorical_features)
```