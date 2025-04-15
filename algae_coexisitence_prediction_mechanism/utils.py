import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer, RobustScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, f1_score, classification_report, r2_score
from sklearn.feature_selection import mutual_info_regression
import shap
import copy
from collections import Counter


def preprocess_data_class(df, test_size=0.1, random_state=42):
    """数据预处理函数，包含特征工程和标准化，并剔除出现频率小于5的藻类类别"""
    print(f"原始数据形状: {df.shape}")

    # 处理藻类标签
    algae_labels = df['Dominant Algae Species'].str.split('、|>|；').apply(lambda x: [i.strip() for i in x])

    # 展平标签列表以统计每个类别的出现频率
    all_labels = [label for sublist in algae_labels for label in sublist]
    label_counts = Counter(all_labels)

    # 确定高频标签（出现次数 >= 5）
    high_freq_labels = {label for label, count in label_counts.items() if count >= 7}

    # 过滤每个样本的标签，只保留高频标签
    filtered_algae_labels = [
        [label for label in labels if label in high_freq_labels]
        for labels in algae_labels
    ]

    # 多标签编码
    mlb = MultiLabelBinarizer()
    y_species = mlb.fit_transform(filtered_algae_labels)
    print(f"过滤后藻类种类标签数量: {len(mlb.classes_)}")

    # 提取特征并进行特征工程
    features = df.drop(['Algae Density (x106cells/L)', 'Dominant Algae Species'], axis=1)

    # # 可选的特征工程: 添加交互项（取消注释以启用）
    # features['Tempreature_pH'] = features['Tempreature'] * features['pH']
    # features['Tempreature_Conductivity'] = features['Tempreature'] * features['Conductivity']
    # features['Tempreature_Dissolved oxygen'] = features['Tempreature'] * features['Dissolved oxygen']
    # features['DO_pH'] = features['Dissolved oxygen'] * features['pH']

    # 使用更鲁棒的标准化方法
    x_scaler = RobustScaler()  # 对异常值不敏感
    X_scaled = x_scaler.fit_transform(features)

    # 划分数据集
    X_train, X_test, ys_train, ys_test = train_test_split(
        X_scaled, y_species, test_size=test_size, random_state=random_state)


    return X_train, X_test, ys_train, ys_test, mlb, x_scaler, features.columns



class AlgaeSpeciesDataset(Dataset):
    """藻类种类分类任务的数据集"""
    def __init__(self, features, species):
        self.features = torch.FloatTensor(features)
        self.species = torch.FloatTensor(species)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return {
            'features': self.features[idx],
            'species': self.species[idx]
        }

# ----------------------
# 模型定义: 多标签分类模型
# ----------------------
class AlgaeSpeciesClassifier(nn.Module):
    """改进的藻类种类多标签分类模型"""
    def __init__(self, input_size, num_species, dropout_rate=0.3):
        super(AlgaeSpeciesClassifier, self).__init__()

        # 更宽的网络结构，更好地处理多标签任务
        self.net = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate),

            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate),

            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate/2),

            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate/2),

            nn.Linear(64, num_species)
            # 注意: 此处不加sigmoid，使用BCEWithLogitsLoss
        )

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)

# ----------------------
# Focal Loss 实现 (提高多标签分类性能)
# ----------------------
class FocalLoss(nn.Module):
    """Focal Loss - 适用于类别不平衡的多标签分类问题"""
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, input, target):
        # 使用sigmoid函数获取预测概率
        BCE_loss = nn.functional.binary_cross_entropy_with_logits(
            input, target, reduction='none'
        )
        pt = torch.exp(-BCE_loss)  # 预测概率
        focal_loss = (1 - pt) ** self.gamma * BCE_loss

        if self.alpha is not None:
            # 添加类别权重
            focal_loss = self.alpha * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
