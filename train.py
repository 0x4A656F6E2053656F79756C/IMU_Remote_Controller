import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split

# ---------------------------
# 1. 설정 및 데이터 로드 함수
# ---------------------------
FEATURE_COLS = ['gx', 'gy', 'gz', 'ax', 'ay', 'az']

def process_file_variable_length(file_path):
    df = pd.read_csv(file_path)
    data = df[FEATURE_COLS].values
    return data

def load_and_split_data(user_folder_dict):
    """사용자별로 여러 폴더의 데이터를 긁어와 Train/Val/Test로 분할"""
    data_dict = {'train': [], 'val': [], 'test': []}
    user_names = list(user_folder_dict.keys())
    
    for user_id, user_name in enumerate(user_names):
        folder_paths = user_folder_dict[user_name]
        user_files = []
        
        # 해당 사용자에 매핑된 모든 폴더 순회
        for folder_path in folder_paths:
            if not os.path.exists(folder_path):
                print(f"경고: {folder_path} 폴더가 존재하지 않습니다.")
                continue
            files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv')]
            user_files.extend(files)
            
        if len(user_files) == 0:
            print(f"경고: {user_name} 사용자의 데이터가 없습니다.")
            continue
            
        # 긁어모은 전체 파일에 대해 분할 수행
        train_files, temp_files = train_test_split(user_files, test_size=0.3, random_state=42)
        val_files, test_files = train_test_split(temp_files, test_size=0.5, random_state=42)
        
        for split, flist in zip(['train', 'val', 'test'], [train_files, val_files, test_files]):
            for f in flist:
                seq = process_file_variable_length(f)
                data_dict[split].append((seq, user_id)) # user_id는 enumerate에서 나온 정수
                
    return data_dict, user_names

# ---------------------------
# 2. Triplet Dataset 및 Collate 함수 (기존과 동일)
# ---------------------------
class VariableLengthTripletDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list
        self.labels = np.array([item[1] for item in data_list])
        
    def __len__(self):
        return len(self.data_list)
        
    def __getitem__(self, idx):
        anchor_seq, anchor_label = self.data_list[idx]
        pos_indices = np.where(self.labels == anchor_label)[0]
        pos_idx = np.random.choice(pos_indices)
        pos_seq = self.data_list[pos_idx][0]
        
        neg_indices = np.where(self.labels != anchor_label)[0]
        neg_idx = np.random.choice(neg_indices)
        neg_seq = self.data_list[neg_idx][0]
        
        return anchor_seq, pos_seq, neg_seq, anchor_label

def triplet_collate_fn(batch):
    anchors, positives, negatives, labels = zip(*batch)
    len_a = torch.tensor([len(a) for a in anchors], dtype=torch.int64)
    len_p = torch.tensor([len(p) for p in positives], dtype=torch.int64)
    len_n = torch.tensor([len(n) for n in negatives], dtype=torch.int64)
    
    pad_a = nn.utils.rnn.pad_sequence([torch.tensor(a, dtype=torch.float32) for a in anchors], batch_first=True)
    pad_p = nn.utils.rnn.pad_sequence([torch.tensor(p, dtype=torch.float32) for p in positives], batch_first=True)
    pad_n = nn.utils.rnn.pad_sequence([torch.tensor(n, dtype=torch.float32) for n in negatives], batch_first=True)
    
    labels = torch.tensor(labels, dtype=torch.int64)
    return (pad_a, len_a), (pad_p, len_p), (pad_n, len_n), labels

# ---------------------------
# 3. Feature Extractor (기존과 동일)
# ---------------------------
class Pure_LSTM_Extractor(nn.Module):
    def __init__(self, input_size=6, hidden_size=128, num_layers=2):
        super(Pure_LSTM_Extractor, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, 
                            num_layers=num_layers, batch_first=True, bidirectional=True)
        self.embed_dim = hidden_size * 2
        
    def forward(self, x, lengths):
        packed_x = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed_x)
        hidden_forward = h_n[-2, :, :]
        hidden_backward = h_n[-1, :, :]
        embedding = torch.cat((hidden_forward, hidden_backward), dim=1)
        embedding = nn.functional.normalize(embedding, p=2, dim=1)
        return embedding

# ---------------------------
# 4. 학습 루프 및 t-SNE (기존과 동일)
# ---------------------------
def train_lstm_model(train_data, epochs=50, device='cpu'):
    print(f"[{device.upper()}] 환경에서 학습을 시작합니다...")
    dataset = VariableLengthTripletDataset(train_data)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=triplet_collate_fn)
    
    model = Pure_LSTM_Extractor(input_size=6, hidden_size=64, num_layers=2).to(device)
    criterion = nn.TripletMarginLoss(margin=1.0, p=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for (a_seq, a_len), (p_seq, p_len), (n_seq, n_len), _ in dataloader:
            a_seq, a_len = a_seq.to(device), a_len.to(device)
            p_seq, p_len = p_seq.to(device), p_len.to(device)
            n_seq, n_len = n_seq.to(device), n_len.to(device)
            
            optimizer.zero_grad()
            emb_a = model(a_seq, a_len)
            emb_p = model(p_seq, p_len)
            emb_n = model(n_seq, n_len)
            
            loss = criterion(emb_a, emb_p, emb_n)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")
            
    return model

def visualize_tsne_lstm(model, eval_data, user_names, device='cpu'):
    model.eval()
    embeddings = []
    labels = []
    with torch.no_grad():
        for seq, label in eval_data:
            x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)
            length = torch.tensor([len(seq)], dtype=torch.int64).to(device)
            emb = model(x, length).squeeze(0).cpu().numpy()
            embeddings.append(emb)
            labels.append(label)
            
    embeddings = np.array(embeddings)
    labels = np.array(labels)
    
    perplexity_val = min(5, len(embeddings) - 1)
    if perplexity_val < 1: return # 데이터 부족시 시각화 생략
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity_val)
    tsne_results = tsne.fit_transform(embeddings)
    
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap='viridis', s=100)
    plt.legend(handles=scatter.legend_elements()[0], labels=[user_names[i] for i in np.unique(labels)])
    plt.title("t-SNE Visualization of Variable-Length IMU Embeddings")
    
    save_path = "tsne_result.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"t-SNE 결과가 '{save_path}' 로 저장되었습니다.")

# ---------------------------
# 5. 메인 실행 블록
# ---------------------------
if __name__ == "__main__":
    # 변경됨: 사용자별로 폴더를 여러 개 매핑할 수 있는 딕셔너리 구조
    USER_FOLDERS = {
        'Jun_Yong': ['./Jun_Yong', './Jun_Yong_2'], # 여러 폴더 허용
        'Seo_Yul': ['./Seo_Yul', './Seo_Yul_2']
    }
    
    print("데이터를 로드하고 Train/Val/Test로 분할합니다...")
    data_splits, user_names = load_and_split_data(USER_FOLDERS)
    
    if len(data_splits['train']) > 0:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        trained_model = train_lstm_model(data_splits['train'], epochs=30, device=device)
        
        model_save_path = "saved_lstm_model.pth"
        torch.save(trained_model.state_dict(), model_save_path)
        print(f"학습된 모델이 '{model_save_path}' 파일로 저장되었습니다.")
        
        visualize_tsne_lstm(trained_model, data_splits['test'], user_names, device=device)
    else:
        print("학습할 데이터가 없습니다.")