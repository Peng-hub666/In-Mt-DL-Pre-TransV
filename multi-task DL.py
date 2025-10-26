import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
from optuna.trial import TrialState
import joblib
import os
import random

# 设置工作目录
os.chdir()

# 设置随机种子以确保完全可重复性
SEED = 123
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 检查是否有可用的GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 读取数据
train_df = pd.read_csv('train_set.csv')
val_df = pd.read_csv('val_set.csv')
test_df = pd.read_csv('test_set.csv')

# 定义特征和目标列
feature_cols = ['Male', 'CPBtime', 'European scoce 1', 'DO2_average', 'D.V ratio_min', 'Hct_1',
                'Bladder_temperature_Min']
target_cols = ['RBC', 'transfusion']


# 数据预处理
def preprocess_data(train_df, val_df, test_df, feature_cols):
    # 初始化标准化器
    scaler = StandardScaler()

    # 拟合训练数据
    train_features = train_df[feature_cols]
    scaler.fit(train_features)

    # 转换所有数据集
    train_features_scaled = scaler.transform(train_features)
    val_features_scaled = scaler.transform(val_df[feature_cols])
    test_features_scaled = scaler.transform(test_df[feature_cols])

    # 获取目标值
    train_rbc = train_df['RBC'].values
    train_transfusion = train_df['transfusion'].values

    val_rbc = val_df['RBC'].values
    val_transfusion = val_df['transfusion'].values

    test_rbc = test_df['RBC'].values
    test_transfusion = test_df['transfusion'].values

    return (train_features_scaled, train_rbc, train_transfusion,
            val_features_scaled, val_rbc, val_transfusion,
            test_features_scaled, test_rbc, test_transfusion, scaler)


# 预处理数据
(train_features, train_rbc, train_transfusion,
 val_features, val_rbc, val_transfusion,
 test_features, test_rbc, test_transfusion, scaler) = preprocess_data(train_df, val_df, test_df, feature_cols)


# 创建自定义数据集
class MultiTaskDataset(Dataset):
    def __init__(self, features, rbc_targets, transfusion_targets):
        self.features = torch.FloatTensor(features)
        self.rbc_targets = torch.FloatTensor(rbc_targets).unsqueeze(1)
        self.transfusion_targets = torch.FloatTensor(transfusion_targets).unsqueeze(1)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.rbc_targets[idx], self.transfusion_targets[idx]


# 定义多任务学习模型，更侧重于RBC预测
class MultiTaskModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, dropout_rate=0.2, rbc_weight=0.7):
        super(MultiTaskModel, self).__init__()
        self.rbc_weight = rbc_weight

        # 共享层
        self.shared_layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate)
        )

        # RBC预测分支 (回归) - 更复杂的结构
        self.rbc_branch = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LeakyReLU(),
            nn.Linear(hidden_size // 2, 1)
        )

        # Transfusion预测分支 (分类) - 简化结构
        self.transfusion_branch = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LeakyReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        shared_features = self.shared_layers(x)
        rbc_output = self.rbc_branch(shared_features)
        transfusion_output = self.transfusion_branch(shared_features)
        return rbc_output, transfusion_output


# 训练和验证函数，更侧重于RBC损失
def train_and_validate(model, train_loader, val_loader, optimizer, rbc_criterion, transfusion_criterion,
                       num_epochs=100, rbc_weight=0.7):
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_state = None

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_rbc_loss = 0.0
        train_transfusion_loss = 0.0
        train_total_loss = 0.0

        for features, rbc_targets, transfusion_targets in train_loader:
            features = features.to(device)
            rbc_targets = rbc_targets.to(device)
            transfusion_targets = transfusion_targets.to(device)

            optimizer.zero_grad()
            rbc_output, transfusion_output = model(features)

            rbc_loss = rbc_criterion(rbc_output, rbc_targets)
            transfusion_loss = transfusion_criterion(transfusion_output, transfusion_targets)

            # 组合两个任务的损失，给RBC更高的权重
            total_loss = rbc_weight * rbc_loss + (1 - rbc_weight) * transfusion_loss
            total_loss.backward()
            optimizer.step()

            train_rbc_loss += rbc_loss.item()
            train_transfusion_loss += transfusion_loss.item()
            train_total_loss += total_loss.item()

        # 验证阶段
        model.eval()
        val_rbc_loss = 0.0
        val_transfusion_loss = 0.0
        val_total_loss = 0.0

        with torch.no_grad():
            for features, rbc_targets, transfusion_targets in val_loader:
                features = features.to(device)
                rbc_targets = rbc_targets.to(device)
                transfusion_targets = transfusion_targets.to(device)

                rbc_output, transfusion_output = model(features)

                rbc_loss = rbc_criterion(rbc_output, rbc_targets)
                transfusion_loss = transfusion_criterion(transfusion_output, transfusion_targets)
                total_loss = rbc_weight * rbc_loss + (1 - rbc_weight) * transfusion_loss

                val_rbc_loss += rbc_loss.item()
                val_transfusion_loss += transfusion_loss.item()
                val_total_loss += total_loss.item()

        # 计算平均损失
        train_rbc_loss /= len(train_loader)
        train_transfusion_loss /= len(train_loader)
        train_total_loss /= len(train_loader)

        val_rbc_loss /= len(val_loader)
        val_transfusion_loss /= len(val_loader)
        val_total_loss /= len(val_loader)

        train_losses.append(train_total_loss)
        val_losses.append(val_total_loss)

        # 保存最佳模型
        if val_total_loss < best_val_loss:
            best_val_loss = val_total_loss
            best_model_state = model.state_dict().copy()

        if (epoch + 1) % 50 == 0:
            print(
                f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_total_loss:.4f}, Val Loss: {val_total_loss:.4f}')
            print(f'  RBC Loss: {val_rbc_loss:.4f}, Transfusion Loss: {val_transfusion_loss:.4f}')

    # 加载最佳模型状态
    model.load_state_dict(best_model_state)
    return train_losses, val_losses, val_total_loss, val_rbc_loss


# 定义Optuna目标函数，更侧重于RBC损失
def objective(trial):
    # 设置试验特定的随机种子以确保可重复性
    trial_seed = SEED + trial.number
    torch.manual_seed(trial_seed)
    np.random.seed(trial_seed)
    random.seed(trial_seed)

    # 超参数建议
    hidden_size = trial.suggest_categorical('hidden_size', [4, 8, 16, 32])
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    num_epochs = trial.suggest_int('num_epochs', 100, 400, step=50)
    rbc_weight = trial.suggest_float('rbc_weight', 0.5, 0.8)  # RBC损失权重

    # 创建数据加载器
    train_dataset = MultiTaskDataset(train_features, train_rbc, train_transfusion)
    val_dataset = MultiTaskDataset(val_features, val_rbc, val_transfusion)

    # 使用固定种子生成器确保数据加载可重复
    generator = torch.Generator()
    generator.manual_seed(trial_seed)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 初始化模型
    input_size = len(feature_cols)
    model = MultiTaskModel(input_size, hidden_size, dropout_rate, rbc_weight).to(device)

    # 定义损失函数和优化器
    rbc_criterion = nn.MSELoss()
    transfusion_criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 训练和验证
    _, _, _, val_rbc_loss = train_and_validate(
        model, train_loader, val_loader, optimizer,
        rbc_criterion, transfusion_criterion, num_epochs, rbc_weight
    )

    # 返回RBC的验证损失作为优化目标
    return val_rbc_loss


# 运行超参数优化
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=1000)

# 输出最佳超参数
print("Number of finished trials: ", len(study.trials))
print("Best trial:")
trial = study.best_trial

print("  Value (RBC Loss): ", trial.value)
print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

# 使用最佳超参数训练最终模型
best_params = trial.params

# 设置最终训练的随机种子
final_seed = SEED + 1000  # 使用一个不同的种子以确保独立性
torch.manual_seed(final_seed)
np.random.seed(final_seed)
random.seed(final_seed)

# 创建数据加载器
train_dataset = MultiTaskDataset(train_features, train_rbc, train_transfusion)
val_dataset = MultiTaskDataset(val_features, val_rbc, val_transfusion)
test_dataset = MultiTaskDataset(test_features, test_rbc, test_transfusion)

# 使用固定种子生成器
generator = torch.Generator()
generator.manual_seed(final_seed)

train_loader = DataLoader(
    train_dataset,
    batch_size=best_params['batch_size'],
    shuffle=True,
    generator=generator
)
val_loader = DataLoader(val_dataset, batch_size=best_params['batch_size'], shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=best_params['batch_size'], shuffle=False)

# 初始化模型
input_size = len(feature_cols)
model = MultiTaskModel(
    input_size,
    hidden_size=best_params['hidden_size'],
    dropout_rate=best_params['dropout_rate'],
    rbc_weight=best_params['rbc_weight']
).to(device)

# 定义损失函数和优化器
rbc_criterion = nn.MSELoss()
transfusion_criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=best_params['learning_rate'])

# 训练最终模型
print("开始训练最终模型...")
num_epochs = best_params['num_epochs']
train_losses, val_losses, _, _ = train_and_validate(
    model, train_loader, val_loader, optimizer,
    rbc_criterion, transfusion_criterion, num_epochs, best_params['rbc_weight']
)

# 绘制训练和验证损失曲线
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.savefig('training_curve.png')
plt.show()


# 评估函数
def evaluate_model(model, data_loader):
    model.eval()
    rbc_predictions = []
    transfusion_predictions = []
    rbc_targets_list = []
    transfusion_targets_list = []

    with torch.no_grad():
        for features, rbc_targets, transfusion_targets in data_loader:
            features = features.to(device)
            rbc_targets = rbc_targets.to(device)
            transfusion_targets = transfusion_targets.to(device)

            rbc_output, transfusion_output = model(features)

            rbc_predictions.extend(rbc_output.cpu().numpy())
            transfusion_predictions.extend(transfusion_output.cpu().numpy())
            rbc_targets_list.extend(rbc_targets.cpu().numpy())
            transfusion_targets_list.extend(transfusion_targets.cpu().numpy())

    # 处理RBC预测 (回归)
    rbc_predictions = np.array(rbc_predictions).flatten()
    rbc_targets_list = np.array(rbc_targets_list).flatten()
    rbc_mse = mean_squared_error(rbc_targets_list, rbc_predictions)
    rbc_rmse = np.sqrt(rbc_mse)
    rbc_mae = np.mean(np.abs(rbc_targets_list - rbc_predictions))
    rbc_r2 = 1 - (rbc_mse / np.var(rbc_targets_list))

    # 处理Transfusion预测 (分类)
    transfusion_predictions = np.array(transfusion_predictions).flatten()
    transfusion_pred_classes = (transfusion_predictions > 0.5).astype(int)
    transfusion_targets_list = np.array(transfusion_targets_list).flatten().astype(int)
    transfusion_accuracy = accuracy_score(transfusion_targets_list, transfusion_pred_classes)

    return {
        'rbc': {
            'predictions': rbc_predictions,
            'targets': rbc_targets_list,
            'mse': rbc_mse,
            'rmse': rbc_rmse,
            'mae': rbc_mae,
            'r2': rbc_r2
        },
        'transfusion': {
            'predictions': transfusion_predictions,
            'pred_classes': transfusion_pred_classes,
            'targets': transfusion_targets_list,
            'accuracy': transfusion_accuracy,
            'confusion_matrix': confusion_matrix(transfusion_targets_list, transfusion_pred_classes),
            'classification_report': classification_report(transfusion_targets_list, transfusion_pred_classes,
                                                           output_dict=True)
        }
    }


# 在验证集上评估模型
print("Validation Set Evaluation:")
val_results = evaluate_model(model, val_loader)
print(
    f"RBC - MSE: {val_results['rbc']['mse']:.4f}, RMSE: {val_results['rbc']['rmse']:.4f}, "
    f"MAE: {val_results['rbc']['mae']:.4f}, R²: {val_results['rbc']['r2']:.4f}")
print(f"Transfusion - Accuracy: {val_results['transfusion']['accuracy']:.4f}")
print("\nTransfusion Classification Report:")
print(classification_report(
    val_results['transfusion']['targets'],
    val_results['transfusion']['pred_classes']
))

# 在测试集上评估模型
print("\nTest Set Evaluation:")
test_results = evaluate_model(model, test_loader)
print(
    f"RBC - MSE: {test_results['rbc']['mse']:.4f}, RMSE: {test_results['rbc']['rmse']:.4f}, "
    f"MAE: {test_results['rbc']['mae']:.4f}, R²: {test_results['rbc']['r2']:.4f}")
print(f"Transfusion - Accuracy: {test_results['transfusion']['accuracy']:.4f}")
print("\nTransfusion Classification Report:")
print(classification_report(
    test_results['transfusion']['targets'],
    test_results['transfusion']['pred_classes']
))

# 绘制RBC预测值与真实值的散点图
plt.figure(figsize=(10, 6))
plt.scatter(test_results['rbc']['targets'], test_results['rbc']['predictions'], alpha=0.5)
plt.plot([test_results['rbc']['targets'].min(), test_results['rbc']['targets'].max()],
         [test_results['rbc']['targets'].min(), test_results['rbc']['targets'].max()], 'r--')
plt.xlabel('True RBC Values')
plt.ylabel('Predicted RBC Values')
plt.title('RBC Prediction: True vs Predicted')
plt.savefig('rbc_prediction.png')
plt.show()

# 绘制Transfusion的混淆矩阵
plt.figure(figsize=(8, 6))
sns.heatmap(test_results['transfusion']['confusion_matrix'], annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Transfusion Confusion Matrix')
plt.savefig('transfusion_confusion_matrix.png')
plt.show()

# 保存模型
torch.save(model.state_dict(), 'multi_task_model.pth')
print("Model saved as multi_task_model.pth")

# 保存标准化器
joblib.dump(scaler, 'scaler.pkl')
print("Scaler saved as scaler.pkl")

# 保存最佳超参数
with open('best_hyperparameters.txt', 'w') as f:
    for key, value in best_params.items():
        f.write(f"{key}: {value}\n")
print("Best hyperparameters saved as best_hyperparameters.txt")

# 绘制超参数重要性图
optuna.visualization.plot_param_importances(study).show()

# 绘制优化历史图
optuna.visualization.plot_optimization_history(study).show()

# 保存研究对象以便后续分析
joblib.dump(study, 'optuna_study.pkl')
print("Optuna study saved as optuna_study.pkl")