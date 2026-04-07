import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def extract_movement_window(file_path):
    # 1. 데이터 불러오기 및 시간 변환
    df = pd.read_csv(file_path)
    if 'timestamp_us' in df.columns:
        df['ms'] = df['timestamp_us'] / 1000.0
    else:
        print("올바른 시간 컬럼(timestamp_us)이 없습니다.")
        return

    # 2. 가속도 크기(Magnitude) 계산
    df['a_mag'] = np.sqrt(df['ax']**2 + df['ay']**2 + df['az']**2)

    # 3. 움직임 활성도(Activity) 계산
    # Window size를 20(약 0.1~0.2초)으로 설정하여 가속도의 분산(표준편차)을 구함
    window_size = 20 
    df['activity'] = df['a_mag'].rolling(window=window_size, center=True).std().fillna(0)

    # 4. 임계값(Threshold) 설정 (최대 활성도의 5%를 노이즈 마진으로 설정)
    threshold = df['activity'].max() * 0.05

    # 5. 움직임 시작(Start)과 끝(End) 인덱스 찾기
    active_mask = df['activity'] > threshold
    if not active_mask.any():
        print(f"{file_path}: 유의미한 움직임이 감지되지 않았습니다.")
        return

    active_indices = df.index[active_mask].tolist()
    start_idx = active_indices[0]
    end_idx = active_indices[-1]

    start_time = df['ms'].iloc[start_idx]
    end_time = df['ms'].iloc[end_idx]
    duration = end_time - start_time

    # 6. 구간이 1초(1000ms) 미만일 경우 앞쪽으로 패딩(Padding)
    target_duration = 1000.0
    if duration < target_duration:
        padding_needed = target_duration - duration
        new_start_time = start_time - padding_needed

        # 0초(데이터 시작점) 이하로 내려가지 않도록 방어 로직
        if new_start_time < df['ms'].iloc[0]:
            new_start_time = df['ms'].iloc[0]

        # 새로운 시작 시간에 가장 가까운 인덱스 찾기
        start_idx = (df['ms'] - new_start_time).abs().idxmin()
        start_time = df['ms'].iloc[start_idx]
        
    final_duration = end_time - start_time

    # 7. 시각화 (2개의 서브플롯)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # (1) Activity 및 Threshold 그래프
    ax1.plot(df['ms'], df['activity'], color='orange', label='Movement Activity (Std Dev)')
    ax1.axhline(threshold, color='red', linestyle=':', label=f'Threshold ({threshold:.1f})')
    ax1.axvline(start_time, color='green', linestyle='--', linewidth=2, label='Start')
    ax1.axvline(end_time, color='purple', linestyle='--', linewidth=2, label='End')
    ax1.set_title('Movement Detection based on Activity Level')
    ax1.set_ylabel('Activity')
    ax1.legend()
    ax1.grid(True)

    # (2) 원본 가속도 크기 및 추출 구간 하이라이트
    ax2.plot(df['ms'], df['a_mag'], color='black', label='Acc Magnitude')
    # 추출된 전체 구간을 노란색으로 하이라이트
    ax2.axvspan(start_time, end_time, color='yellow', alpha=0.3, label='Extracted Window')
    ax2.axvline(start_time, color='green', linestyle='--', linewidth=2)
    ax2.axvline(end_time, color='purple', linestyle='--', linewidth=2)
    ax2.set_title(f'Extracted Action Window (Final Duration: {final_duration:.1f} ms)')
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Acc (Raw)')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    
    # 이미지 저장
    filename_only = os.path.splitext(os.path.basename(file_path))[0]
    save_path = f"{filename_only}_window_extraction.png"
    plt.savefig(save_path)
    plt.close()
    
    print(f"[{filename_only}]")
    print(f" - 추출 시작: {start_time:.1f}ms")
    print(f" - 추출 종료: {end_time:.1f}ms")
    print(f" - 최종 확보된 구간 길이: {final_duration:.1f}ms")
    print(f" - 그래프 저장 완료: {save_path}\n")

# 실행 예시 (로컬 환경에서 돌릴 때 실제 파일명으로 변경)
extract_movement_window('../2_1/imu_record_20260330_210700.csv')