# Customer Behavior AI - Telco Churn Prediction

Dự án phân tích hành vi khách hàng và dự đoán tỷ lệ rời bỏ (churn) sử dụng Machine Learning.

## 📋 Mục lục

- [Giới thiệu](#giới-thiệu)
- [Cấu trúc dự án](#cấu-trúc-dự-án)
- [Cài đặt](#cài-đặt)
- [Hướng dẫn sử dụng](#hướng-dẫn-sử-dụng)
- [Mô hình](#mô-hình)
- [Kết quả](#kết-quả)
- [API Reference](#api-reference)

## 🎯 Giới thiệu

Dự án này xây dựng hệ thống dự đoán khách hàng có nguy cơ rời bỏ dịch vụ viễn thông, bao gồm:

- **Phân tích dữ liệu (EDA)**: Khám phá và trực quan hóa dữ liệu khách hàng
- **Feature Engineering**: Tạo các đặc trưng mới để cải thiện mô hình
- **Classification Models**: Dự đoán churn sử dụng Logistic Regression, Random Forest, XGBoost
- **Customer Segmentation**: Phân nhóm khách hàng sử dụng KMeans clustering
- **Model Evaluation**: Đánh giá hiệu quả mô hình và phân tích tác động kinh doanh

### Dataset

Sử dụng **Telco Customer Churn Dataset** với:
- **7,043** khách hàng
- **21** đặc trưng gốc → **26** features sau engineering
- **Train:** 4,929 samples (70%)
- **Validation:** 705 samples (10%)
- **Test:** 1,409 samples (20%)
- **Churn rate:** 26.54%

## 📁 Cấu trúc dự án

```
customer-behavior-ai/
│
├── data/
│   ├── raw/                    # Dữ liệu gốc (CSV từ Kaggle)
│   ├── processed/              # Dữ liệu sau làm sạch
│   ├── train/                  # Tập huấn luyện
│   ├── val/                    # Tập validation
│   └── test/                   # Tập kiểm thử
│
├── notebooks/
│   ├── eda.ipynb               # Phân tích & trực quan dữ liệu
│   ├── feature_engineering.ipynb
│   ├── training.ipynb          # Thử nghiệm model
│   └── evaluation.ipynb        # Đánh giá mô hình
│
├── src/
│   ├── preprocessing/
│   │   ├── data_cleaning.py    # Làm sạch dữ liệu
│   │   └── feature_engineering.py
│   │
│   ├── models/
│   │   ├── classification.py   # Logistic, RF, XGBoost
│   │   ├── clustering.py       # KMeans (phân nhóm KH)
│   │   └── regression.py       # CLV prediction
│   │
│   ├── training/
│   │   └── train.py            # Script huấn luyện
│   │
│   ├── evaluation/
│   │   └── metrics.py          # Accuracy, F1, AUC
│   │
│   ├── inference/
│   │   └── predict.py          # Dự đoán offline
│   │
│   └── config.py               # Cấu hình chung
│
├── models_saved/
│   ├── churn_model.pkl
│   └── clustering_model.pkl
│
├── reports/
│   └── figures/                # Biểu đồ, hình ảnh
│
├── requirements.txt
├── README.md
└── .gitignore
```

## 🛠 Cài đặt

### Yêu cầu

- Python 3.8+
- pip hoặc conda

### Cài đặt dependencies

```bash
# Clone repository
git clone <repository-url>
cd customer-behavior-ai

# Tạo virtual environment (khuyến nghị)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc
venv\Scripts\activate     # Windows

# Cài đặt packages
pip install -r requirements.txt
```

## 📖 Hướng dẫn sử dụng

### 1. Chuẩn bị dữ liệu

```bash
# Copy file CSV vào thư mục data/raw/
cp WA_Fn-UseC_-Telco-Customer-Churn.csv data/raw/

# Chạy data cleaning
python src/preprocessing/data_cleaning.py

# Chạy feature engineering
python src/preprocessing/feature_engineering.py
```

### 2. Training models

```bash
# Train tất cả models và so sánh
python src/training/train.py --compare

# Train một model cụ thể
python src/training/train.py --model xgboost

# Train với hyperparameter tuning
python src/training/train.py --model xgboost --tune
```

### 3. Dự đoán

```python
from src.inference.predict import ChurnPredictor

# Khởi tạo predictor
predictor = ChurnPredictor()

# Dự đoán cho một khách hàng
customer = {
    'gender': 'Male',
    'SeniorCitizen': 0,
    'Partner': 'Yes',
    'Dependents': 'No',
    'tenure': 12,
    'PhoneService': 'Yes',
    'MultipleLines': 'No',
    'InternetService': 'Fiber optic',
    'OnlineSecurity': 'No',
    'OnlineBackup': 'No',
    'DeviceProtection': 'No',
    'TechSupport': 'No',
    'StreamingTV': 'No',
    'StreamingMovies': 'No',
    'Contract': 'Month-to-month',
    'PaperlessBilling': 'Yes',
    'PaymentMethod': 'Electronic check',
    'MonthlyCharges': 70.35,
    'TotalCharges': 844.2
}

result = predictor.predict(customer)
print(result)
# {'churn_prediction': 'Yes', 'churn_probability': 0.72, 'risk_level': 'High'}
```

### 4. Chạy Notebooks

```bash
# Khởi động Jupyter
jupyter notebook notebooks/

# Chạy theo thứ tự:
# 1. eda.ipynb
# 2. feature_engineering.ipynb
# 3. training.ipynb
# 4. evaluation.ipynb
```

## 🤖 Mô hình

### Classification Models

| Model | Mô tả |
|-------|-------|
| **Logistic Regression** | Baseline model, interpretable |
| **Random Forest** | Ensemble method, handles non-linearity |
| **XGBoost** | Gradient boosting, best performance |
| **Gradient Boosting** | Alternative boosting method |

### Clustering Models

| Model | Mô tả |
|-------|-------|
| **KMeans** | Phân khách hàng thành các segment |

### Features được sử dụng

**Original Features:**
- Demographics: gender, SeniorCitizen, Partner, Dependents
- Services: PhoneService, InternetService, OnlineSecurity, etc.
- Account: Contract, PaperlessBilling, PaymentMethod
- Charges: MonthlyCharges, TotalCharges, tenure

**Engineered Features:**
- TenureGroup: Nhóm theo thời gian sử dụng
- AvgMonthlyCharge: Trung bình chi tiêu hàng tháng
- ChargeIncrease: Mức tăng chi phí
- TotalServices: Số lượng dịch vụ đăng ký
- HighValue: Khách hàng giá trị cao
- ShortTermContract: Hợp đồng ngắn hạn
- AutomaticPayment: Thanh toán tự động

## 📊 Kết quả

### Model Comparison (Validation Set)

| Model | Accuracy | F1 Score | ROC AUC |
|-------|----------|----------|--------|
| **Gradient Boosting** | 0.7956 | 0.5714 | 0.8328 |
| **XGBoost** | 0.7921 | 0.5607 | 0.8322 |
| **Random Forest** | 0.7679 | 0.6148 | 0.8348 |
| **Logistic Regression** | 0.7374 | 0.6138 | 0.8421 |

### XGBoost Final Model (Test Set)

| Metric | Score |
|--------|-------|
| Accuracy | 0.7921 |
| Precision | 0.6382 |
| Recall | 0.5000 |
| F1 Score | 0.5607 |
| Specificity | 0.8976 |
| ROC-AUC | 0.8322 |
| PR-AUC | 0.6363 |

**Confusion Matrix (Test Set):**
```
  TN:   929  FP:   106
  FN:   187  TP:   187
```

### Top 10 Feature Importances

| Feature | Importance |
|---------|------------|
| ShortTermContract | 0.6103 |
| OnlineSecurity | 0.0773 |
| InternetService | 0.0474 |
| Contract | 0.0267 |
| TechSupport | 0.0204 |
| StreamingMovies | 0.0186 |
| Tenure | 0.0168 |
| MultipleLines | 0.0142 |
| PaperlessBilling | 0.0134 |
| MonthlyCharges | 0.0132 |

### Key Findings

1. **ShortTermContract** là yếu tố quan trọng nhất (61%) - khách hàng hợp đồng ngắn hạn có nguy cơ churn rất cao
2. **OnlineSecurity** (7.7%) - khách hàng không có dịch vụ bảo mật online có xu hướng rời bỏ
3. **InternetService** (4.7%) - loại dịch vụ Internet ảnh hưởng đến quyết định của khách hàng
4. **TechSupport** - khách hàng không có hỗ trợ kỹ thuật dễ churn hơn
5. **Tenure** - khách hàng mới (<12 tháng) có nguy cơ cao hơn

### Customer Segmentation (KMeans Clustering)

**Clustering Features:** tenure, MonthlyCharges, TotalCharges (scaled)

**Evaluation Metrics:**
| Metric | Score |
|--------|-------|
| Silhouette Score | 0.4720 |
| Calinski-Harabasz Score | 6757.52 |
| Davies-Bouldin Score | 0.7071 |

**Customer Segments (4 clusters):**
| Segment | Đặc điểm | Chiến lược |
|---------|----------|------------|
| 0 | High tenure, high value | Loyalty programs, VIP benefits |
| 1 | Low tenure, high charges | Retention focus, satisfaction surveys |
| 2 | Medium tenure, medium charges | Upselling opportunities |
| 3 | New customers, low charges | Onboarding support, engagement |

### Generated Reports

Các biểu đồ đánh giá được lưu tại `reports/figures/`:
- `confusion_matrix.png` - Ma trận nhầm lẫn
- `roc_curve.png` - Đường cong ROC
- `precision_recall_curve.png` - Đường cong Precision-Recall
- `threshold_analysis.png` - Phân tích ngưỡng quyết định

## 📚 API Reference

### ChurnClassifier

```python
from src.models.classification import ChurnClassifier

# Khởi tạo
clf = ChurnClassifier(model_type='xgboost')

# Training
clf.fit(X_train, y_train)

# Prediction
predictions = clf.predict(X_test)
probabilities = clf.predict_proba(X_test)

# Cross-validation
cv_results = clf.cross_validate(X_train, y_train, scoring='roc_auc')

# Hyperparameter tuning
clf.tune_hyperparameters(X_train, y_train)

# Feature importance
importance_df = clf.get_feature_importance(feature_names)

# Save/Load
clf.save('model.pkl')
clf = ChurnClassifier.load('model.pkl')
```

### CustomerSegmentation

```python
from src.models.clustering import CustomerSegmentation

# Khởi tạo
segmenter = CustomerSegmentation(model_type='kmeans', n_clusters=4)

# Fit
segmenter.fit(X)

# Evaluate
metrics = segmenter.evaluate(X)

# Find optimal k
results = segmenter.find_optimal_k(X, k_range=range(2, 10))
```

### ChurnPredictor

```python
from src.inference.predict import ChurnPredictor

predictor = ChurnPredictor()

# Single prediction
result = predictor.predict(customer_dict)

# Batch prediction
results_df = predictor.predict_batch(customers_df)

# Get recommendations
recommendations = predictor.get_retention_recommendations(customer_dict)
```

## 🔧 Configuration

Xem file `src/config.py` để thay đổi các cấu hình:

```python
# Data paths
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

# Model parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.1
CV_FOLDS = 5
N_CLUSTERS = 4
```

## 📄 License

MIT License

## 👥 Tác giả

Customer Behavior AI Team

## 📧 Liên hệ

- Email: your-email@example.com
- GitHub: https://github.com/your-username
