# 🏦 Dự báo Rủi ro Tín dụng & Xây dựng Credit Scorecard cho Ngân hàng

> **Mô hình Machine Learning dự báo khả năng vỡ nợ và Credit Scorecard chuẩn ngân hàng (WOE/IV) cho khách hàng vay tiêu dùng tại Việt Nam**

---

## 📌 Tóm tắt

Dự án xây dựng hệ thống đánh giá rủi ro tín dụng end-to-end gồm 2 đầu ra:

1. **Mô hình ML phân loại khách hàng** thành 2 nhóm — *trả nợ đúng hạn* và *có nguy cơ vỡ nợ* — sau khi so sánh **7 thuật toán** (Logistic Regression, Random Forest, **XGBoost**, LightGBM, SVM, Linear SVM, Mixture of Experts).
2. **Credit Scorecard chuẩn công nghiệp** dựa trên Weight of Evidence (WOE) và Information Value (IV), thang điểm **617–1151**, chia khách hàng thành **5 phân khúc rủi ro** kèm chiến lược *risk-based pricing*.

**Kết quả nổi bật:**
- XGBoost: AUC = 0.77, Gini = 0.54 (chỉ số chuẩn industry banking) — discriminatory power tốt nhất.
- Random Forest: Recall = 0.81 (cao nhất) — tối ưu phát hiện khách hàng vỡ nợ.
- Credit Scorecard với cut-off 978: tỷ lệ default trong phân khúc "Very Poor" là **33.5%**, trong khi phân khúc "Excellent" chỉ **3.32%** — cho thấy khả năng phân tách rất rõ.

---

## 🎯 Bối cảnh & Vì sao bài toán quan trọng

Đầu năm 2023, thị trường ngân hàng Việt Nam đối mặt với biến động vĩ mô và **tỷ lệ nợ xấu (NPL) tăng nhanh**, đặc biệt ở mảng tín dụng tiêu dùng. Các ngân hàng đứng trước bài toán chiến lược: **vừa mở rộng tệp khách hàng vừa kiểm soát rủi ro nợ quá hạn**.

Các quy trình chấm điểm tín dụng truyền thống chủ yếu dựa trên rule-based hoặc mô hình thống kê cổ điển — đơn giản, dễ giải thích, nhưng:
- Khó xử lý dữ liệu lớn, đa chiều, phi tuyến.
- Khó cân bằng hai loại rủi ro: *duyệt nhầm khách rủi ro cao* (thiệt hại tài chính trực tiếp) và *từ chối nhầm khách tốt* (mất doanh thu).

Dự án này áp dụng ML hiện đại để giải quyết chính bài toán đó, với **3 mục tiêu kinh doanh cụ thể**:

| Mục tiêu | Tác động |
|---|---|
| Giảm tỷ lệ NPL | Phát hiện sớm khách hàng có khả năng default → từ chối hoặc yêu cầu thêm tài sản bảo đảm |
| Tự động hóa quy trình phê duyệt | Thay thế thẩm định thủ công bằng metric định lượng → giảm thời gian xử lý hồ sơ, giảm bias chủ quan |
| Tối ưu lợi nhuận qua *Risk-based Pricing* | Khách "Excellent" → lãi suất ưu đãi, hạn mức cao; khách "Subprime" → lãi suất cao bù rủi ro |

---

## 📊 Dataset

- **124 biến đầu vào** (số lượng khoản vay, số dư còn lại, số lần enquiry, số quan hệ ngân hàng, hành vi trả nợ thẻ tín dụng, lịch sử số dư theo thời gian 3M/6M/9M/12M…).
- **Target**: nhị phân — `0` (trả đúng hạn) vs `1` (vỡ nợ/trả chậm).
- **Class imbalance**: nhãn `1` chiếm ~**20%** → cần xử lý bằng Stratified Sampling & metric phù hợp.
- **Tỷ lệ NULL**: ~10% trên hầu hết các biến — đáng kể, không thể bỏ qua.

Nhóm biến chính:
- **COUNT** — số lượng khoản vay theo loại (ngắn/trung/dài hạn, bank/non-bank).
- **OUTSTANDING_BAL** — số dư còn nợ, có cả phiên bản hiện tại và lịch sử 3M/6M/9M/12M, kèm biến chênh lệch giữa các kỳ.
- **NUM_NEW_LOAN_TAKEN** — số khoản vay mới trong cửa sổ thời gian gần.
- **ENQUIRIES** — số lần ngân hàng/tổ chức tín dụng tra cứu hồ sơ khách hàng.
- **NUMBER_OF_RELATIONSHIP** — số mối quan hệ tài chính của khách hàng.

---

## 🔧 Pipeline xử lý dữ liệu

```
RAW (124 cột, ~10% NULL, mất cân bằng 80/20)
    │
    ├─ Missing values  →  KNN Imputer (k=20)
    │                     [chọn k lớn để giảm bias, giữ cấu trúc dữ liệu]
    │
    ├─ Outliers        →  Winsorization (cap tại percentile 5 / 95)
    │
    ├─ Duplicates      →  loại bỏ
    │
    └─ Non-informative →  loại các cột chỉ có 1 giá trị duy nhất
    
DATASET SẠCH (117 cột, 0 NULL)
```

**Tại sao chọn KNN Imputer thay vì mean/zero?** Vì các biến trong bài có quan hệ phức tạp với nhau (số lượng khoản vay liên quan đến số quan hệ ngân hàng, đến số dư…). Điền bằng mean sẽ phá vỡ tương quan này. KNN với k=20 tận dụng "hàng xóm gần" để giữ cấu trúc dữ liệu.

**Tại sao Winsorization thay vì xóa outlier?** Outlier trong tài chính thường là *thông tin thật* (khách hàng VIP, khách hàng vấn đề), không phải lỗi nhập liệu. Cap thay vì xóa giúp giữ lại tín hiệu mà không để cực trị méo mô hình.

---

## 🎯 Feature Selection — kết hợp 3 phương pháp

Đây là phần mình tâm đắc nhất của pipeline — **không dùng 1 mà 3 kỹ thuật bổ sung cho nhau**:

| Bước | Kỹ thuật | Số biến | Mục đích |
|---|---|---|---|
| 1 | **PCA** | 117 → 35 | Gom nhóm các biến tương quan cao (OUTSTANDING_BAL, NUM_NEW_LOAN, ENQUIRIES theo các mốc thời gian) thành principal components |
| 2 | **Information Gain** (Filter) | top 30 | Chọn biến có entropy giảm nhiều nhất đối với target |
| 3 | **RFE + Logistic Regression** (Wrapper) | 35 → 20 | Loại dần biến yếu, giữ biến đóng góp thật vào performance |
| 4 | **StandardScaler** | — | Chuẩn hóa để các mô hình distance-based hoạt động đúng |
| 5 | **Stratified Split** 80/20 | — | Giữ tỷ lệ class giống nhau ở train/validation |

Logic 3 bước này phản ánh tư duy thực chiến: PCA giảm chiều cho dữ liệu tương quan thời gian, Information Gain lọc filter-based nhanh, RFE wrapper xác nhận từng biến thật sự đóng góp vào mô hình.

---

## 🔍 4 Insight quan trọng từ EDA

### 1. Nghịch lý "credit thin file" — khách ít khoản vay lại rủi ro nhất

Nhóm có **chỉ 1 khoản vay** (`NUMBER_OF_LOANS = 1`) có tỷ lệ default cao nhất. Khi số khoản vay >1, tỷ lệ này giảm đáng kể và ổn định. Lý giải: khách có "thick file" thường đã được vetting kỹ và có discipline tài chính tốt hơn.

### 2. Tín hiệu "Bẫy nợ" qua hành vi theo thời gian

- **Rủi ro cao (tỷ lệ default = 0.45):** khách vay mới đều đặn qua các mốc 12M/9M/6M/3M — dấu hiệu *vay mới để trả nợ cũ*.
- **An toàn (tỷ lệ default = 0.05):** khách có xu hướng *giảm* vay mới theo thời gian.

Chênh lệch **9 lần** giữa 2 nhóm — đây là một trong những tín hiệu mạnh nhất trong toàn bộ dataset.

### 3. Độ sâu quan hệ ngân hàng = chỉ báo trust

Khách chỉ có 1 quan hệ tài chính thường thiếu hỗ trợ tư vấn, dẫn đến quản lý nợ kém. IV của `NUMBER_OF_RELATIONSHIP_BANK` thuộc nhóm **Strong Predictor**.

### 4. Hành vi dài hạn quan trọng hơn biến động ngắn hạn

IV của `NUMBER_OF_LOANS` = **0.55** (rất cao), trong khi `INCREASING_BAL` và `ENQUIRIES` đóng góp thấp hơn nhiều. → Định hướng feature engineering tập trung vào **pattern dài hạn**, không phải biến động tức thời.

---

## 🤖 So sánh 7 mô hình

Tất cả mô hình đều được tune bằng **GridSearchCV** trên Stratified K-Fold. Kết quả trên validation set:

| Model | Accuracy | Precision | Recall | F1 | AUC | **Gini** |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.78 | 0.61 | 0.68 | 0.64 | 0.765 | 0.53 |
| Random Forest | 0.81 | 0.58 | **0.81** | **0.68** | 0.76 | 0.52 |
| **XGBoost** | **0.83** | **0.65** | 0.63 | 0.64 | **0.77** | **0.54** |
| LightGBM | 0.73 | 0.31 | 0.64 | 0.42 | 0.77 | 0.54 |
| Linear SVM | 0.68 | 0.28 | 0.71 | 0.40 | 0.77 | 0.54 |
| Kernel SVM | — | — | — | — | — | — |
| Mixture of Experts | 0.54 | 0.53 | 0.35 | 0.42 | 0.55 | 0.10 |

### Trade-off kinh doanh

| Mục tiêu chiến lược | Mô hình phù hợp | Lý do |
|---|---|---|
| **Tối ưu giảm rủi ro / phát hiện default** | Random Forest (Recall = 0.81) | Bắt được nhiều nhất khách vỡ nợ → giảm False Negative (loại sai lầm tốn kém nhất) |
| **Phân khúc tín dụng & risk-based pricing** | **XGBoost** (Gini = 0.54, AUC = 0.77) | Discriminatory power mạnh nhất → tách khách hàng thành nhiều bậc rủi ro rõ ràng |
| **Tuân thủ pháp lý & giải thích** | Logistic Regression | Hệ số trực tiếp → dễ trình bày với cơ quan quản lý |

→ **Mô hình chọn cuối cùng: XGBoost** — cân bằng tốt nhất giữa accuracy, AUC và khả năng giải thích thông qua feature importance.

### Vì sao MoE thất bại?

Mixture of Experts (mạng neural với gating mechanism) chỉ đạt AUC 0.55 — gần như random. Lý do: dataset chỉ ~20K mẫu — quá nhỏ cho deep learning, dễ rơi vào overfitting/underfitting. Đây là kết quả trung thực — **không phải mô hình phức tạp nào cũng tốt**, đôi khi XGBoost cổ điển vẫn vô địch trên dữ liệu tabular vừa và nhỏ.

---

## 💳 Credit Scorecard — phần "industry-standard" nhất của dự án

Đây là phần phân biệt một dự án ML thông thường với một dự án **deploy được trong ngân hàng thật**. Scorecard là chuẩn được Basel II/III chấp nhận, mọi ngân hàng đều dùng.

### Quy trình xây dựng

```
20 feature đã chọn  →  Binning (Equal Frequency hoặc Decision Tree tùy phân phối)
                  │
                  ▼
              Tính WOE cho mỗi bin:
              WOE = ln(Good_i/Good_total ÷ Bad_i/Bad_total)
                  │
                  ▼
              Tính IV cho mỗi biến (Strong > 0.3, Medium 0.1-0.3, Weak < 0.1)
                  │
                  ▼
              Logistic Regression trên các giá trị WOE
                  │
                  ▼
              Quy đổi sang điểm: Score = 600 + (50/ln2) × ln(Odds/20)
                                 [PDO=50, Base=600, Base Odds=20]
```

### Phân khúc 5 bậc + chiến lược tín dụng

| Phân khúc | Khoảng điểm | % khách | Default rate | Hành động ngân hàng |
|---|---|---|---|---|
| **Very Poor** | 617–978 | 30% | **33.5%** | **Từ chối cho vay** — rủi ro quá cao |
| Poor | 978–1030 | 30% | 11.88% | Subprime — duyệt nếu có tài sản bảo đảm, lãi suất cao |
| Fair | 1030–1062 | 20% | 7.0% | Duyệt với lãi suất trung bình + giám sát chặt |
| Good | 1062–1083 | 10% | 5.9% | Duyệt thuận lợi, lãi suất ưu đãi vừa phải |
| **Excellent** | 1083–1151 | 10% | **3.32%** | Hạn mức cao, lãi suất tốt nhất, ưu tiên xây quan hệ dài hạn |

**Cut-off được chọn = 978**, dưới mức này tự động từ chối. Ngưỡng này tối ưu hóa Recall (bắt được khách rủi ro) đồng thời giữ Precision đủ tốt để không từ chối oan khách tốt.

**Chênh lệch default rate giữa nhóm thấp nhất (3.32%) và cao nhất (33.5%)** ≈ **10 lần** — chứng tỏ scorecard tách rất sắc.

---

## 💡 Đề xuất kinh doanh cụ thể

### Cải tiến dữ liệu & mô hình
- Thu thập thêm dữ liệu **thu nhập & chi tiêu** — hiện scorecard chỉ dựa vào lịch sử vay nợ, thiếu chiều "khả năng trả".
- Retrain định kỳ (3–6 tháng/lần) để mô hình thích nghi với chu kỳ kinh tế.
- Tích hợp scorecard vào hệ thống phê duyệt real-time để rút ngắn thời gian xử lý hồ sơ.

### Chính sách phân khúc khách hàng
- **Very Poor / Poor**: hạn chế vay, chỉ duyệt khoản nhỏ ngắn hạn, lãi cao, yêu cầu collateral.
- **Fair**: lãi trung bình + giám sát repayment.
- **Good / Excellent**: ưu tiên duyệt, lãi thấp, hạn mức cao, kỳ hạn dài, tập trung **customer lifetime value**.

### Hỗ trợ nâng hạng tín dụng
- Reminder tự động (SMS/email/in-app) trước hạn thanh toán.
- Chương trình **financial literacy** cho nhóm rủi ro cao — vừa giảm default, vừa mở rộng tệp vay tương lai.
- Loyalty/incentive theo cải thiện điểm tín dụng.
- Credit insurance để chuyển một phần rủi ro sang công ty bảo hiểm.

---

## 🛠️ Tech Stack

| Hạng mục | Thư viện |
|---|---|
| Xử lý dữ liệu | `pandas`, `numpy` |
| Imputation & Outliers | `sklearn.impute.KNNImputer`, `scipy.stats.mstats.winsorize` |
| Feature Selection | `sklearn.decomposition.PCA`, `sklearn.feature_selection` (Information Gain, RFE) |
| Mô hình ML | `sklearn` (Logistic, RF, SVM, Linear SVM), `xgboost`, `lightgbm` |
| Mạng neural | `tensorflow` / `pytorch` (Mixture of Experts) |
| Tuning & Validation | `GridSearchCV`, `StratifiedKFold` |
| Scorecard | `optbinning` / `scorecardpy` cho WOE/IV (hoặc tự cài đặt) |
| Trực quan hóa | `matplotlib`, `seaborn` |

---

## 📁 Cấu trúc repository

```
.
├── README.md                         # File này
├── MCL__Credit_Risk_in_Bank.ipynb    # Notebook full pipeline
├── Analytics_report.docx             # Báo cáo chi tiết 46 trang (EN)
└── data/                             # Dữ liệu (theo NDA, không commit)
    ├── 01_dataset.csv
    └── 02_data_description.xlsx
```

---

## 🚀 Cách reproduce

```bash
# 1. Clone
git clone <repo-url>
cd credit-risk-scorecard

# 2. Môi trường
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install pandas numpy scikit-learn xgboost lightgbm tensorflow \
            matplotlib seaborn scipy optbinning jupyter

# 3. Chạy notebook
jupyter notebook MCL__Credit_Risk_in_Bank.ipynb
```

---

## ⚠️ Hạn chế & Hướng phát triển

Trung thực về phạm vi dự án — đây là phần nhà tuyển dụng đánh giá cao senior thinking:

- **Chưa tích hợp XGBoost vào scorecard.** Scorecard hiện tại dùng Logistic Regression (chuẩn industry vì interpretable), nhưng có thể thử *XGBoost + SHAP-based scorecard* để vừa mạnh vừa giải thích được.
- **Time-series feature engineering còn cơ bản.** Các biến `ENQUIRIES_FOR_3M / 6M / 9M / 12M` có thể được biến đổi thành features mô tả *trend* (slope, acceleration) thay vì chỉ giữ giá trị thô.
- **Chưa có rule-based fallback.** Một số case biên (ví dụ: số khoản vay non-bank quá nhiều) nên có rule cứng từ chối trước khi vào model, tăng độ tin cậy.
- **Chưa có giải thích cá nhân hóa (case-level).** Áp dụng SHAP để giải thích từng quyết định duyệt/từ chối cho từng khách hàng — quan trọng cho compliance.
- **MoE thất bại trên dataset này** không có nghĩa là deep learning vô dụng — có thể thử lại khi có dataset >100K mẫu.

---

## 📚 Tham khảo chính

Project có literature review của **10+ paper** về credit risk modeling, đáng chú ý:

- **Wang et al. (2022)** — XGBoost cho personal credit risk, AUC 0.943
- **Rajput et al. (2024)** — XGBoost + SHAP, accuracy 88.36%, interpretable cho regulation
- **Han et al. (2025)** — DeepCreditRisk với symmetry-aware deep learning, cải thiện AUC 7.2%
- **Zhao et al. (2024)** — Voting Ensemble với RFE, accuracy 83.93%

Xem mục **References** trong `Analytics_report.docx` để có danh sách đầy đủ và link DOI.

---

## 👥 Tác giả

[Điền tên + vai trò cụ thể bạn đảm nhiệm trong nhóm — ví dụ: "Phụ trách Feature Selection (PCA + IG + RFE), xây dựng XGBoost và LightGBM, thiết kế Credit Scorecard với WOE/IV"]

📧 Liên hệ: [email/linkedin]
