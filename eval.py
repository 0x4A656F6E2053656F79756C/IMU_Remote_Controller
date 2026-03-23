import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.manifold import TSNE  # t-SNE 라이브러리 추가

# ---------------------------
# 1. 평가 폴더 설정
# ---------------------------
REF_FOLDERS = {
    'Jun_Yong': ['./Jun_Yong'],
    'Seo_Yul': ['./Seo_Yul']
}

TEST_FOLDERS = {
    'Jun_Yong': ['./test_Jun_Yong'], 
    'Seo_Yul': ['./test_Seo_Yul']
}

FEATURE_COLS = ['gx', 'gy', 'gz', 'ax', 'ay', 'az']
USER_NAMES = list(REF_FOLDERS.keys())

# ---------------------------
# 2. 모델 구조 및 헬퍼 함수
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

def process_file_variable_length(file_path):
    df = pd.read_csv(file_path)
    return df[FEATURE_COLS].values

def load_data_from_dict(folder_dict, user_names):
    data_list = []
    for user_id, user_name in enumerate(user_names):
        if user_name not in folder_dict: continue
        
        folder_paths = folder_dict[user_name]
        for folder_path in folder_paths:
            if not os.path.exists(folder_path): 
                print(f"경고: 평가 폴더 {folder_path} 가 존재하지 않습니다.")
                continue
            files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv')]
            for f in files:
                seq = process_file_variable_length(f)
                data_list.append((seq, user_id))
    return data_list

def extract_embeddings(model, data_list, device='cpu'):
    model.eval()
    embeddings, labels = [], []
    with torch.no_grad():
        for seq, label in data_list:
            x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)
            length = torch.tensor([len(seq)], dtype=torch.int64).to(device)
            emb = model(x, length).squeeze(0).cpu().numpy()
            
            embeddings.append(emb)
            labels.append(label)
    return np.array(embeddings), np.array(labels)

# ---------------------------
# 추가: Evaluation 전용 t-SNE 시각화 함수
# ---------------------------
def visualize_evaluation_tsne(ref_emb, ref_labels, test_emb, test_labels, user_names):
    # 동일한 공간에 매핑하기 위해 Reference와 Test 임베딩을 하나로 합칩니다.
    all_emb = np.vstack((ref_emb, test_emb))
    
    perplexity_val = min(5, len(all_emb) - 1)
    if perplexity_val < 1: return
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity_val)
    tsne_results = tsne.fit_transform(all_emb)
    
    # 합쳤던 결과를 다시 분리합니다.
    ref_tsne = tsne_results[:len(ref_emb)]
    test_tsne = tsne_results[len(ref_emb):]
    
    plt.figure(figsize=(10, 8))
    
    # 색상 맵 설정
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # Reference 플롯 (투명한 동그라미)
    for i, name in enumerate(user_names):
        idx = (ref_labels == i)
        plt.scatter(ref_tsne[idx, 0], ref_tsne[idx, 1], marker='o', 
                    color=colors[i % len(colors)], alpha=0.3, s=100, label=f'{name} (Reference)')
        
    # Test 플롯 (진한 별표, 테두리 추가)
    for i, name in enumerate(user_names):
        idx = (test_labels == i)
        plt.scatter(test_tsne[idx, 0], test_tsne[idx, 1], marker='*', 
                    color=colors[i % len(colors)], edgecolors='black', s=250, label=f'{name} (Test)')
        
    plt.title("t-SNE: Reference vs Test Embeddings")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout() # 레전드가 잘리지 않도록 레이아웃 조정
    
    save_path = "tsne_evaluation.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"t-SNE 시각화 결과가 '{save_path}' 로 저장되었습니다.")

# ---------------------------
# 3. 메인 실행 블록
# ---------------------------
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. 모델 불러오기
    model_path = "lstm_model_1.pth"
    model = Pure_LSTM_Extractor(input_size=6, hidden_size=64, num_layers=2).to(device)
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"'{model_path}' 모델을 성공적으로 불러왔습니다.")
    else:
        print(f"'{model_path}' 파일이 없습니다. 먼저 train.py를 실행해주세요.")
        exit()

    # 2. Reference 및 Test 데이터 분리 로드
    print("Reference 및 Test 데이터를 불러옵니다...")
    ref_data = load_data_from_dict(REF_FOLDERS, USER_NAMES)
    test_data = load_data_from_dict(TEST_FOLDERS, USER_NAMES)
    
    if not ref_data or not test_data:
        print("데이터 로드에 실패했습니다. 폴더 설정을 다시 확인해주세요.")
        exit()

    print("임베딩을 추출합니다...")
    ref_embeddings, ref_labels = extract_embeddings(model, ref_data, device)
    test_embeddings, test_labels = extract_embeddings(model, test_data, device)

    # 3. K-NN 분류기 (임베딩 거리를 기반으로 가장 가까운 사람 찾기)
    knn = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
    knn.fit(ref_embeddings, ref_labels)
    
    # 4. 테스트 데이터 예측
    predictions = knn.predict(test_embeddings)
    
    # 5. 오차 행렬 (Confusion Matrix) 생성
    cm = confusion_matrix(test_labels, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=USER_NAMES)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    plt.title("Confusion Matrix of IMU Classification")
    
    cm_save_path = "confusion_matrix.png"
    plt.savefig(cm_save_path, dpi=300)
    plt.close()
    print(f"평가 완료! 오차 행렬 결과가 '{cm_save_path}' 로 저장되었습니다.")
    
    # 6. t-SNE 시각화 생성 및 저장
    print("t-SNE 시각화를 진행합니다...")
    visualize_evaluation_tsne(ref_embeddings, ref_labels, test_embeddings, test_labels, USER_NAMES)