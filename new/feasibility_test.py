import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from scipy.signal import butter, filtfilt

# ==========================================
# 1. 하이퍼파라미터 및 설정
# ==========================================
MAIN_USER_FILE = './data/user_002_extracted.csv'
OTHER_USER_FILE = './data/user_003_extracted.csv'

SEQ_LEN = 75 
NUM_FEATURES = 21 
BATCH_SIZE = 16
EPOCHS = 150 
LR = 0.0005 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==========================================
# 2. 특징 추출 및 정규화 함수
# ==========================================
def compute_features(df_group):
    dt = 0.02
    # 1. Magnitudes
    df_group['a_mag'] = np.sqrt(df_group['ax']**2 + df_group['ay']**2 + df_group['az']**2)
    df_group['g_mag'] = np.sqrt(df_group['gx']**2 + df_group['gy']**2 + df_group['gz']**2)
    # 2. Jerk
    df_group['jx'] = df_group['ax'].diff().fillna(0) / dt
    df_group['jy'] = df_group['ay'].diff().fillna(0) / dt
    df_group['jz'] = df_group['az'].diff().fillna(0) / dt
    df_group['j_mag'] = np.sqrt(df_group['jx']**2 + df_group['jy']**2 + df_group['jz']**2)
    # 3. Gravity & Linear (LPF)
    def lowpass_filter(data, cutoff=2.0, fs=50.0):
        if len(data) < 15: return data
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(3, normal_cutoff, btype='low', analog=False)
        return filtfilt(b, a, data)
    df_group['grav_x'] = lowpass_filter(df_group['ax'].values)
    df_group['grav_y'] = lowpass_filter(df_group['ay'].values)
    df_group['grav_z'] = lowpass_filter(df_group['az'].values)
    df_group['la_x'] = df_group['ax'] - df_group['grav_x']
    df_group['la_y'] = df_group['ay'] - df_group['grav_y']
    df_group['la_z'] = df_group['az'] - df_group['grav_z']
    # 4. Velocity
    df_group['vx'] = df_group['la_x'].cumsum() * dt
    df_group['vy'] = df_group['la_y'].cumsum() * dt
    df_group['vz'] = df_group['la_z'].cumsum() * dt
    return df_group

def load_and_preprocess(filepath):
    print(f"Loading and processing: {filepath}")
    df = pd.read_csv(filepath)
    feature_cols = [
        'ax', 'ay', 'az', 'gx', 'gy', 'gz', 'a_mag', 'g_mag', 'jx', 'jy', 'jz', 'j_mag',
        'grav_x', 'grav_y', 'grav_z', 'la_x', 'la_y', 'la_z', 'vx', 'vy', 'vz'
    ]
    processed_chunks = []
    for _, group in df.groupby('event_id'):
        processed_group = compute_features(group.copy())
        seq = processed_group[feature_cols].values
        if len(seq) > SEQ_LEN: seq = seq[:SEQ_LEN]
        elif len(seq) < SEQ_LEN:
            seq = np.pad(seq, ((0, SEQ_LEN - len(seq)), (0, 0)), 'edge')
        processed_chunks.append(seq)
    return np.array(processed_chunks)

def scale_per_window(X):
    """각 윈도우(샘플)별로 [0, 1] Min-Max Scaling 적용"""
    # X shape: (Samples, Seq, Features)
    X_scaled = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[2]): # 각 특징별로
            min_val = X[i, :, j].min()
            max_val = X[i, :, j].max()
            if max_val - min_val > 1e-8:
                X_scaled[i, :, j] = (X[i, :, j] - min_val) / (max_val - min_val)
            else:
                X_scaled[i, :, j] = 0
    return X_scaled

# 데이터 준비
main_raw = load_and_preprocess(MAIN_USER_FILE)
other_raw = load_and_preprocess(OTHER_USER_FILE)

# 윈도우별 정규화 적용 (가장 중요한 변경점)
main_scaled = scale_per_window(main_raw)
other_scaled = scale_per_window(other_raw)

# Train / Test 분리
X_train, X_test_main = train_test_split(main_scaled, test_size=0.2, random_state=42)
X_test_other = other_scaled

def to_tensor(X):
    # (Batch, Seq, Channel) -> (Batch, Channel, Seq)
    return torch.tensor(X, dtype=torch.float32).permute(0, 2, 1).to(device)

X_train_t = to_tensor(X_train)
X_test_main_t = to_tensor(X_test_main)
X_test_other_t = to_tensor(X_test_other)

train_loader = torch.utils.data.DataLoader(X_train_t, batch_size=BATCH_SIZE, shuffle=True)

# ==========================================
# 3. 강화된 Tight CNN Autoencoder 모델
# ==========================================
class TightConvAE(nn.Module):
    def __init__(self):
        super(TightConvAE, self).__init__()
        # Encoder: 병목(Bottleneck)을 8채널로 축소
        self.encoder = nn.Sequential(
            nn.Conv1d(NUM_FEATURES, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 8, kernel_size=3, padding=1), # 16 -> 8 채널로 강화
            nn.BatchNorm1d(8), nn.ReLU(),
            nn.MaxPool1d(2)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv1d(8, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.Upsample(size=SEQ_LEN),
            nn.Conv1d(32, NUM_FEATURES, kernel_size=5, padding=2)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

model = TightConvAE().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# ==========================================
# 4. 모델 학습 및 평가
# ==========================================
print("\n--- Training Started ---")
for epoch in range(EPOCHS):
    model.train()
    e_loss = 0
    for bx in train_loader:
        optimizer.zero_grad()
        out = model(bx)
        loss = criterion(out, bx)
        loss.backward()
        optimizer.step()
        e_loss += loss.item()
    if (epoch + 1) % 20 == 0:
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {e_loss/len(train_loader):.4f}")

model.eval()
with torch.no_grad():
    m_err = torch.mean((X_test_main_t - model(X_test_main_t))**2, dim=[1, 2]).cpu().numpy()
    o_err = torch.mean((X_test_other_t - model(X_test_other_t))**2, dim=[1, 2]).cpu().numpy()

# ROC-AUC
y_true = np.concatenate([np.zeros_like(m_err), np.ones_like(o_err)])
y_scores = np.concatenate([m_err, o_err])
fpr, tpr, _ = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)
print(f"\n🎯 최종 결과 AUC Score: {roc_auc:.4f}")

# 시각화 (Density=True 추가)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(m_err, bins=30, alpha=0.5, label='Main (Test)', color='blue', density=True)
plt.hist(o_err, bins=30, alpha=0.5, label='Other (Imposter)', color='red', density=True)
plt.title('Reconstruction Error (Normalized Density)'); plt.legend()

plt.subplot(1, 2, 2)
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}'); plt.plot([0,1],[0,1],'--')
plt.title('ROC Curve'); plt.legend()
plt.savefig('final_integrated_results.png')