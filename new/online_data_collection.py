import os
import glob
import pandas as pd
import numpy as np

def read_bin_imu_fast(filepath, cols):
    """
    구조화된 Dtype(Structured Dtype)을 사용하여 
    Endian 에러 없이 안전하고 빠르게 바이너리를 해독하는 함수
    """
    with open(filepath, 'rb') as f:
        data = f.read()
    
    # 1개의 데이터 행(Row)은 20 Bytes로 구성됨
    step = 20
    num_records = len(data) // step
    data = data[:num_records * step]
    
    # Kaggle 데이터의 Timestamp는 밀리초(ms) 단위
    sensor_dtype = np.dtype([
        ('timestamp_ms', '>i8'), 
        ('x', '<f4'), 
        ('y', '<f4'), 
        ('z', '<f4')
    ])
    
    arr = np.frombuffer(data, dtype=sensor_dtype)
    
    df = pd.DataFrame({
        cols[0]: arr['timestamp_ms'].astype(np.int64),
        cols[1]: arr['x'].astype(np.float32),
        cols[2]: arr['y'].astype(np.float32),
        cols[3]: arr['z'].astype(np.float32)
    })
    
    return df

# 1. 경로 설정
base_dir = '../IMU_specific_motion/train_val_test'
output_dir = './data'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"📁 '{output_dir}' 폴더가 생성되었습니다.")

# 2. 001번 ~ 090번 유저 폴더 리스트업
user_folders = sorted(glob.glob(os.path.join(base_dir, '*')))

for user_folder in user_folders:
    user_id = os.path.basename(user_folder) 
    
    target_path_pattern = os.path.join(user_folder, 's20', f'*')
    target_folders = glob.glob(target_path_pattern)
    
    if not target_folders:
        continue
        
    data_folder = target_folders[0]
    
    accel_file = os.path.join(data_folder, 'accel.txt')
    gyro_file = os.path.join(data_folder, 'gyro.txt')
    screen_file = os.path.join(data_folder, 'screen.txt')
    
    if not os.path.exists(accel_file) or not os.path.exists(gyro_file) or not os.path.exists(screen_file):
        print(f"⚠️ [{user_id}] 센서 파일 또는 screen.txt가 부족하여 건너뜁니다.")
        continue
        
    try:
        # 3. Screen 로그 파싱하여 USER_PRESENT 타임스탬프(ms) 추출
        # on_bad_lines='skip'을 추가해 엉뚱한 형식의 줄은 무시
        df_screen = pd.read_csv(screen_file, sep='\s+', header=None, names=['timestamp_ms', 'event'], on_bad_lines='skip')
        
        # 💡 [핵심 수정] 타임스탬프 컬럼을 숫자로 '강제 변환'
        # 문자가 섞여 있으면 NaN(결측치)으로 만들고, 결측치가 된 행을 깔끔하게 지워버림
        df_screen['timestamp_ms'] = pd.to_numeric(df_screen['timestamp_ms'], errors='coerce')
        df_screen = df_screen.dropna(subset=['timestamp_ms'])
        
        # 안전하게 정수형(int64)으로 타입 캐스팅
        df_screen['timestamp_ms'] = df_screen['timestamp_ms'].astype(np.int64)
        
        unlock_events = df_screen[df_screen['event'] == 'android.intent.action.USER_PRESENT']
        unlock_timestamps = unlock_events['timestamp_ms'].values
        
        # 4. 고속 바이너리 해독 함수로 IMU 데이터 불러오기
        cols_accel = ['timestamp_ms', 'ax', 'ay', 'az']
        cols_gyro  = ['timestamp_ms', 'gx', 'gy', 'gz']
        
        df_accel = read_bin_imu_fast(accel_file, cols_accel)
        df_gyro = read_bin_imu_fast(gyro_file, cols_gyro)
        
        # 5. 시간축(ms) 동기화 병합을 위한 정렬 및 결측치 제거
        df_accel = df_accel.sort_values('timestamp_ms').dropna()
        df_gyro = df_gyro.sort_values('timestamp_ms').dropna()
        
        df_6axis = pd.merge_asof(df_accel, df_gyro, on='timestamp_ms', direction='nearest')
        df_6axis.dropna(inplace=True)
        
        # 6. Screen 이벤트 기반으로 1.5초 구간(Window) 추출
        user_windows = []
        for event_id, trigger_time in enumerate(unlock_timestamps):
            # 논문 기준: 잠금해제(Unlock) 시점 기준 -1260ms ~ +240ms
            start_time = trigger_time - 1260
            end_time = trigger_time + 240
            
            mask = (df_6axis['timestamp_ms'] >= start_time) & (df_6axis['timestamp_ms'] <= end_time)
            window_df = df_6axis[mask].copy()
            
            if not window_df.empty:
                # 데이터 병합 후 동작 구분을 위해 event_id 컬럼 추가 (1, 2, 3...)
                window_df.insert(0, 'event_id', event_id + 1)
                user_windows.append(window_df)
                
        if not user_windows:
            print(f"⚠️ [{user_id}] 추출된 유효한 동작 구간이 없습니다.")
            continue
            
        # 7. 추출된 모든 윈도우를 하나의 데이터프레임으로 합치기
        df_final = pd.concat(user_windows, ignore_index=True)
        
        # 💡 8. 네 데이터 규격과 맞추기 위해 밀리초(ms)를 마이크로초(us)로 변환
        df_final.insert(1, 'timestamp_us', df_final['timestamp_ms'] * 1000)
        df_final.drop(columns=['timestamp_ms'], inplace=True) # 기존 ms 컬럼은 삭제
        
        # 9. 최종 csv 파일 저장
        output_filename = os.path.join(output_dir, f'user_{user_id}_extracted.csv')
        df_final.to_csv(output_filename, index=False)
        
        print(f"✅ [User {user_id}] 해독+병합+크롭 완벽 처리! -> {output_filename} (총 {len(user_windows)}개 동작 확보)")
        
    except Exception as e:
        print(f"❌ [User {user_id}] 파일 처리 중 에러 발생: {e}")

print("\n🎉 모든 유저의 데이터 전처리 및 1.5초 윈도우 추출 파이프라인이 완료되었습니다!")