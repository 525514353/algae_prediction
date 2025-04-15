import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
from sklearn.preprocessing import RobustScaler,StandardScaler
from sklearn.model_selection import train_test_split


# ----------------------
# 数据预处理 (共享)
# ----------------------
def preprocess_data_regression(df, test_size=0.1, random_state=42):
    """数据预处理函数，包含特征工程和标准化"""
    print(f"原始数据形状: {df.shape}")

    # 处理密度标签
    y_density = df['Algae Density (x106cells/L)'].values.reshape(-1, 1).astype(np.float32)

    # 提取特征并进行特征工程
    features = df.drop(['Algae Density (x106cells/L)', 'Dominant Algae Species'], axis=1)


    # 使用更鲁棒的标准化方法
    x_scaler = RobustScaler()  # 对异常值不敏感
    X_scaled = x_scaler.fit_transform(features)

    y_scaler=RobustScaler()
    y_density=y_scaler.fit_transform(y_density)

    # 划分数据集
    X_train, X_test, yd_train, yd_test,  = train_test_split(
        X_scaled, y_density,  test_size=test_size,random_state=random_state)

    return X_train, X_test, yd_train, yd_test, x_scaler, y_scaler, features.columns


# ----------------------
# 模型定义: 回归模型
# ----------------------
class AlgaeDensityRegressor(nn.Module):
    """改进的藻类密度回归模型"""
    def __init__(self, input_size, dropout_rate=0.2):
        super(AlgaeDensityRegressor, self).__init__()

        # 使用ResNet风格的残差连接来改善梯度流
        self.fc1 = nn.Linear(input_size, 128)
        self.ln1 = nn.LayerNorm(128)

        self.fc2 = nn.Linear(128, 128)
        self.ln2 = nn.LayerNorm(128)

        self.fc3 = nn.Linear(128, 64)
        self.ln3 = nn.LayerNorm(64)

        self.fc4 = nn.Linear(64, 32)
        self.ln4 = nn.LayerNorm(32)

        self.fc5 = nn.Linear(32, 1)

        self.dropout = nn.Dropout(dropout_rate)
        self.act = nn.LeakyReLU(0.1)

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # 第一层
        out = self.fc1(x)
        out = self.ln1(out)
        out = self.act(out)
        out = self.dropout(out)

        # 残差块1
        residual = out
        out = self.fc2(out)
        out = self.ln2(out)
        out = self.act(out)
        out = self.dropout(out)
        out = out + residual  # 残差连接

        # 后续层
        out = self.fc3(out)
        out = self.ln3(out)
        out = self.act(out)
        out = self.dropout(out)

        out = self.fc4(out)
        out = self.ln4(out)
        out = self.act(out)

        out = self.fc5(out)
        return out

# ----------------------
class AlgaeDensityDataset(Dataset):
    """藻类密度回归任务的数据集"""
    def __init__(self, features, density):
        self.features = torch.FloatTensor(features)
        self.density = torch.FloatTensor(density)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return {
            'features': self.features[idx],
            'density': self.density[idx]
        }

