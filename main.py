import pandas as pd
train = pd.read_csv('/kaggle/input/child-mind-institute-problematic-internet-use/train.csv')
test = pd.read_csv('/kaggle/input/child-mind-institute-problematic-internet-use/test.csv')
data_dictionary = pd.read_csv('/kaggle/input/child-mind-institute-problematic-internet-use/data_dictionary.csv')

display(train.head())
print(f"Train shape: {train.shape}")

display(test.head())
print(f"Test shape: {test.shape}")

display(data_dictionary.head())
print(f"Dictionary shape: {data_dictionary.shape}")

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import numpy as np

# Tạo dữ liệu giả lập và chia thành tập huấn luyện/đánh giá
np.random.seed(42)
torch.manual_seed(42)
n_samples = 1000
n_features = 10
input_data = np.random.rand(n_samples, n_features)
target_labels = np.random.randint(0, 2, size=n_samples)  # Nhãn nhị phân [0, 1]

# Chia dữ liệu thành tập training và validation
X_train, X_val, y_train, y_val = train_test_split(
    input_data, target_labels, test_size=0.2, random_state=42, stratify=target_labels
)
    
# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Chuyển dữ liệu sang tensor để sử dụng với PyTorch
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.long)

# Định nghĩa mô hình MLP (Multi-Layer Perceptron)
class NeuralNetworkModel(nn.Module):
    def __init__(self, input_size):
        super(NeuralNetworkModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)  # Output cho bài toán phân loại nhị phân (2 lớp)
        )
    
    def forward(self, inputs):
        return self.model(inputs)

# Khởi tạo mô hình, hàm mất mát và bộ tối ưu
mlp_model = NeuralNetworkModel(input_size=X_train_tensor.shape[1])
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(mlp_model.parameters(), lr=0.001)

# Vòng lặp huấn luyện
n_epochs = 20
for epoch in range(n_epochs):
    # Quá trình huấn luyện
    mlp_model.train()
    optimizer.zero_grad()
    train_outputs = mlp_model(X_train_tensor)
    train_loss = loss_function(train_outputs, y_train_tensor)
    train_loss.backward()
    optimizer.step()

# Quá trình đánh giá trên tập validation
    mlp_model.eval()
    with torch.no_grad():
        val_outputs = mlp_model(X_val_tensor)
        val_loss = loss_function(val_outputs, y_val_tensor)
        val_predictions = torch.argmax(val_outputs, dim=1)
        val_accuracy = accuracy_score(y_val_tensor.numpy(), val_predictions.numpy())

    print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {train_loss.item():.4f}, Val Loss: {val_loss.item():.4f}, Val Acc: {val_accuracy:.4f}")

# Đánh giá cuối cùng trên tập validation
mlp_model.eval()
with torch.no_grad():
    final_predictions = mlp_model(X_val_tensor)
    final_predictions = torch.argmax(final_predictions, dim=1).numpy()
print("Validation Accuracy:", accuracy_score(y_val_tensor.numpy(), final_predictions))

import os
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import VotingClassifier

from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler

import lightgbm as lgb
import catboost as cb
import xgboost as xgb

import optuna
from optuna.samplers import TPESampler

from imblearn.under_sampling import NearMiss
from imblearn.under_sampling import ClusterCentroids
from imblearn.under_sampling import TomekLinks
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.impute import SimpleImputer

import time
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif, VarianceThreshold
import numpy as np
import polars as pl
import pandas as pd
from sklearn.base import clone
from copy import deepcopy
import optuna
from scipy.optimize import minimize
import os
import matplotlib.pyplot as plt
import seaborn as sns

import re
from colorama import Fore, Style

from tqdm import tqdm
from IPython.display import clear_output
from concurrent.futures import ThreadPoolExecutor

import warnings
warnings.filterwarnings('ignore')
pd.options.display.max_columns = None

# Tải các thư viện học máy và xử lý dữ liệu cần thiết
import lightgbm as lgb
from catboost import CatBoostRegressor, CatBoostClassifier
from xgboost import XGBRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.model_selection import *
from sklearn.metrics import *
def process_file(file_name, directory_name):
    # Đọc file Parquet, loại bỏ cột 'step' và trả về mô tả thống kê của dữ liệu
    data = pd.read_parquet(os.path.join(directory_name, file_name, 'part-0.parquet'))
    data.drop('step', axis=1, inplace=True)
    return data.describe().values.reshape(-1), file_name.split('=')[1]

def load_series_data(directory_name) -> pd.DataFrame:
    # Tải và xử lý dữ liệu chuỗi thời gian từ thư mục
    identifiers = os.listdir(directory_name)
    
    with ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(lambda file: process_file(file, directory_name), identifiers), total=len(identifiers)))
    
    stats, indices = zip(*results)
    
    dataset = pd.DataFrame(stats, columns=[f"Metric_{i}" for i in range(len(stats[0]))])
    dataset['id'] = indices
    
    return dataset

# Đọc dữ liệu từ các file CSV
train_df = pd.read_csv('/kaggle/input/child-mind-institute-problematic-internet-use/train.csv')
test_df = pd.read_csv('/kaggle/input/child-mind-institute-problematic-internet-use/test.csv')
sample_df = pd.read_csv('/kaggle/input/child-mind-institute-problematic-internet-use/sample_submission.csv')

# Tải dữ liệu chuỗi thời gian cho tập huấn luyện và tập kiểm tra
train_series = load_series_data("/kaggle/input/child-mind-institute-problematic-internet-use/series_train.parquet")
test_series = load_series_data("/kaggle/input/child-mind-institute-problematic-internet-use/series_test.parquet")
series_cols = train_series.columns.tolist()
series_cols.remove("id")

# Kết hợp dữ liệu chuỗi thời gian với dữ liệu chính
train_df = pd.merge(train_df, train_series, how="left", on='id')
test_df = pd.merge(test_df, test_series, how="left", on='id')

# Loại bỏ cột 'id' khỏi dữ liệu
train_df = train_df.drop('id', axis=1)
test_df = test_df.drop('id', axis=1)
# Xác định các cột đặc trưng và xử lý các cột chuỗi thời gian
feature_columns = ['Basic_Demos-Enroll_Season', 'Basic_Demos-Age', 'Basic_Demos-Sex',
                   'CGAS-Season', 'CGAS-CGAS_Score', 'Physical-Season', 'Physical-BMI',
                   'Physical-Height', 'Physical-Weight', 'Physical-Waist_Circumference',
                   'Physical-Diastolic_BP', 'Physical-HeartRate', 'Physical-Systolic_BP',
                   'Fitness_Endurance-Season', 'Fitness_Endurance-Max_Stage',
                   'Fitness_Endurance-Time_Mins', 'Fitness_Endurance-Time_Sec',
                   'FGC-Season', 'FGC-FGC_CU', 'FGC-FGC_CU_Zone', 'FGC-FGC_GSND',
                   'FGC-FGC_GSND_Zone', 'FGC-FGC_GSD', 'FGC-FGC_GSD_Zone', 'FGC-FGC_PU',
                   'FGC-FGC_PU_Zone', 'FGC-FGC_SRL', 'FGC-FGC_SRL_Zone', 'FGC-FGC_SRR',
                   'FGC-FGC_SRR_Zone', 'FGC-FGC_TL', 'FGC-FGC_TL_Zone', 'BIA-Season',
                   'BIA-BIA_Activity_Level_num', 'BIA-BIA_BMC', 'BIA-BIA_BMI',
                   'BIA-BIA_BMR', 'BIA-BIA_DEE', 'BIA-BIA_ECW', 'BIA-BIA_FFM',
                   'BIA-BIA_FFMI', 'BIA-BIA_FMI', 'BIA-BIA_Fat', 'BIA-BIA_Frame_num',
                   'BIA-BIA_ICW', 'BIA-BIA_LDM', 'BIA-BIA_LST', 'BIA-BIA_SMM',
                   'BIA-BIA_TBW', 'PAQ_A-Season', 'PAQ_A-PAQ_A_Total', 'PAQ_C-Season',
                   'PAQ_C-PAQ_C_Total', 'SDS-Season', 'SDS-SDS_Total_Raw',
                   'SDS-SDS_Total_T', 'PreInt_EduHx-Season',
                   'PreInt_EduHx-computerinternet_hoursday']

feature_columns += series_cols
sii_target = train_df.sii
train_df = train_df.drop('sii', axis=1, inplace=False)
train_df = train_df[feature_columns]

# Xác định các cột phân loại
categorical_columns = ['Basic_Demos-Enroll_Season', 'CGAS-Season', 'Physical-Season', 'Fitness_Endurance-Season', 
                       'FGC-Season', 'BIA-Season', 'PAQ_A-Season', 'PAQ_C-Season', 'SDS-Season', 'PreInt_EduHx-Season']

def preprocess_categorical(data):
    # Xử lý giá trị bị thiếu và chuyển đổi kiểu dữ liệu sang 'category'
    for column in categorical_columns: 
        data[column] = data[column].fillna('Missing')
        data[column] = data[column].astype('category')
    return data
        
train_df = preprocess_categorical(train_df)
test_df = preprocess_categorical(test_df)

def create_value_mapping(column, dataset):
    # Tạo ánh xạ giá trị cho cột phân loại
    unique_values = dataset[column].unique()
    return {value: idx for idx, value in enumerate(unique_values)}

for column in categorical_columns:
    mapping_train = create_value_mapping(column, train_df)
    mapping_test = create_value_mapping(column, test_df)
    
    train_df[column] = train_df[column].replace(mapping_train).astype(int)
    test_df[column] = test_df[column].replace(mapping_test).astype(int)

train_df['sii'] = sii_target
unlabeled_data = train_df[train_df['sii'].isnull()]
train_df = train_df.dropna(subset='sii')

def calculate_weighted_kappa(estimator, X, y_actual):
    # Tính điểm kappa có trọng số giữa giá trị thực tế và giá trị dự đoán
    y_predicted = estimator.predict(X).astype(y_actual.dtype)
    return cohen_kappa_score(y_actual, y_predicted, weights='quadratic')
# Hiển thị thống kê mô tả dữ liệu chuỗi thời gian
print("Train Time Series Statistics:")
print(train_series.head())

print("\nTest Time Series Statistics:")
print(test_series.head())

# Vẽ biểu đồ phân phối của biến mục tiêu
plt.figure(figsize=(8, 6))
sns.countplot(x=sii_target)
plt.title("Distribution of Target Variable (sii)")
plt.xlabel("Sii")
plt.ylabel("Count")
plt.show()

# Đánh giá tầm quan trọng của đặc trưng sử dụng mô hình LightGBM
lgb_model = lgb.LGBMClassifier(random_state=42)
lgb_model.fit(train_df.drop('sii', axis=1), train_df['sii'])

# Vẽ biểu đồ tầm quan trọng của các đặc trưng
lgb.plot_importance(lgb_model, max_num_features=20, importance_type='gain', figsize=(12, 8))
plt.title("Top 20 Feature Importances")
plt.show()
import numpy as np
import pandas as pd
import os
import re
from sklearn.base import clone
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedKFold
from scipy.optimize import minimize
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from colorama import Fore, Style
from IPython.display import clear_output
import warnings
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import VotingRegressor
# Hàm tính hệ số Kappa có trọng số bậc hai
def quadratic_weighted_kappa(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')

# Hàm làm tròn dựa trên các ngưỡng
def threshold_Rounder(oof_non_rounded, thresholds):
    return np.where(oof_non_rounded < thresholds[0], 0,
                    np.where(oof_non_rounded < thresholds[1], 1,
                             np.where(oof_non_rounded < thresholds[2], 2, 3)))

# Hàm đánh giá các dự đoán so với giá trị thực tế
def evaluate_predictions(thresholds, y_true, oof_non_rounded):
    rounded_p = threshold_Rounder(oof_non_rounded, thresholds)
    return -quadratic_weighted_kappa(y_true, rounded_p)

# Dữ liệu mục tiêu (cột 'sii' từ `train_df`)
sii = train_df['sii']

# Hàm huấn luyện và tạo file dự đoán
def TrainML(model_class, test_data):
    X = train_df.drop(['sii'], axis=1)  # Dữ liệu huấn luyện (loại bỏ cột 'sii')
    y = train_df['sii']  # Nhãn (giá trị cần dự đoán)

    # Tạo đối tượng StratifiedKFold để chia dữ liệu
    SKF = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
    train_S = []  # Lưu điểm Kappa trên tập huấn luyện
    test_S = []   # Lưu điểm Kappa trên tập kiểm tra

    oof_non_rounded = np.zeros(len(y), dtype=float)  # Dự đoán chưa làm tròn
    oof_rounded = np.zeros(len(y), dtype=int)        # Dự đoán đã làm tròn
    test_preds = np.zeros((len(test_data), n_splits))  # Dự đoán trên tập test
        # Huấn luyện mô hình theo từng fold
        for fold, (train_idx, test_idx) in enumerate(tqdm(SKF.split(X, y), desc="Training Folds", total=n_splits)):
        X_train, X_val = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[test_idx]
       
        model = clone(model_class)  # Tạo bản sao của mô hình
        model.fit(X_train, y_train)  # Huấn luyện mô hình
         
        y_train_pred = model.predict(X_train)  # Dự đoán trên tập huấn luyện
        y_val_pred = model.predict(X_val)      # Dự đoán trên tập kiểm tra

        oof_non_rounded[test_idx] = y_val_pred
        y_val_pred_rounded = y_val_pred.round(0).astype(int)
        oof_rounded[test_idx] = y_val_pred_rounded
        # Tính điểm Kappa trên tập huấn luyện và kiểm tra
        train_kappa = quadratic_weighted_kappa(y_train, y_train_pred.round(0).astype(int))
        val_kappa = quadratic_weighted_kappa(y_val, y_val_pred_rounded)

        train_S.append(train_kappa)
        test_S.append(val_kappa)
        
        test_preds[:, fold] = model.predict(test_data)  # Lưu dự đoán cho tập test
        
        print(f"Fold {fold+1} - Train QWK: {train_kappa:.4f}, Validation QWK: {val_kappa:.4f}")
        clear_output(wait=True)

    print(f"Mean Train QWK --> {np.mean(train_S):.4f}")
    print(f"Mean Validation QWK ---> {np.mean(test_S):.4f}")
       # Tối ưu ngưỡng để cải thiện điểm Kappa
    KappaOptimizer = minimize(evaluate_predictions,
                              x0=[0.5, 1.5, 2.5], args=(y, oof_non_rounded), 
                              method='Nelder-Mead')
    assert KappaOptimizer.success, "Optimization did not converge."
    
    oof_tuned = threshold_Rounder(oof_non_rounded, KappaOptimizer.x)
    tKappa = quadratic_weighted_kappa(y, oof_tuned)

    print(f"----> || Optimized QWK SCORE :: {Fore.CYAN}{Style.BRIGHT} {tKappa:.3f}{Style.RESET_ALL}")

    tpm = test_preds.mean(axis=1)  # Trung bình các dự đoán trên tập test
    tpTuned = threshold_Rounder(tpm, KappaOptimizer.x)
    
    # Tạo file nộp (submission)
    submission = pd.DataFrame({
        'id': sample_df['id'],
        'sii': tpTuned
    })

    return submission
# Hiển thị phân phối biến mục tiêu
print(train_df['sii'].value_counts())

# Định nghĩa các tham số global
n_splits = 5  # Số lần gập (folds) trong StratifiedKFold
seed = 42     # Seed để tái lập kết quả

# Tham số mô hình LightGBM
Params = {
    'learning_rate': 0.046,
    'max_depth': 12,
    'num_leaves': 478,
    'min_data_in_leaf': 13,
    'feature_fraction': 0.893,
    'bagging_fraction': 0.784,
    'bagging_freq': 4,
    'lambda_l1': 10,
    'lambda_l2': 0.01
}

# Tham số mô hình XGBoost
XGB_Params = {
    'learning_rate': 0.05,
    'max_depth': 6,
    'n_estimators': 200,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 1,
    'reg_lambda': 5,
    'random_state': seed,
    'tree_method': 'exact'
}

# Tham số mô hình CatBoost
CatBoost_Params = {
    'learning_rate': 0.05,
    'depth': 6,
    'iterations': 200,
    'random_seed': seed,
    'verbose': 0,
    'l2_leaf_reg': 10
}
# Tạo mô hình
Light = LGBMRegressor(**Params, random_state=seed, verbose=-1, n_estimators=300)
XGB_Model = XGBRegressor(**XGB_Params)
CatBoost_Model = CatBoostRegressor(**CatBoost_Params)

# Kết hợp mô hình
voting_model = VotingRegressor(estimators=[
    ('lightgbm', Light),
    ('xgboost', XGB_Model),
    ('catboost', CatBoost_Model)
])

# Huấn luyện và tạo file dự đoán
Submission = TrainML(voting_model, test_df)
Submission.to_csv('submission.csv', index=False)
