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
