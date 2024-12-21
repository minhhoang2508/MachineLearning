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
