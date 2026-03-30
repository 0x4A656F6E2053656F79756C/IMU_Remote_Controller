import serial
import serial.tools.list_ports
import struct
import time
import csv
import os
import threading

def speak(text):
    """macOS 내장 'say' 명령어를 사용하여 음성 출력 (백그라운드 실행)"""
    # '&'를 붙여서 음성 재생 중에도 파이썬 코드가 멈추지 않고 즉시 다음줄 실행
    os.system(f"say '{text}' &")

def select_port():
    ports = list(serial.tools.list_ports.comports())
    if not ports:
        print("연결된 시리얼 장치를 찾을 수 없습니다.")
        return None
    
    print("\n--- 사용 가능한 포트 목록 ---")
    for i, port in enumerate(ports):
        print(f"{i}: {port.device} ({port.description})")
    
    if len(ports) == 1:
        return ports[0].device
    else:
        idx = input("사용할 포트 번호를 입력하세요: ")
        try:
            return ports[int(idx)].device
        except:
            return None

def setup_save_directory():
    print("\n" + "-"*50)
    folder_name = input("데이터를 저장할 폴더 이름을 입력하세요 (엔터 시 현재 폴더): ").strip()
    
    if not folder_name:
        folder_name = "."
    elif not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"📂 폴더 생성 완료: {folder_name}")
        
    return folder_name

def main():
    save_dir = setup_save_directory()
    port_name = select_port()
    if not port_name: return

    BAUD = 500000
    try:
        ser = serial.Serial(port_name, BAUD, timeout=1)
        print(f"✅ 포트 연결 성공: {port_name}")
        time.sleep(2) 
    except Exception as e:
        print(f"❌ 연결 실패: {e}")
        return

    saved_files_count = 0

    try:
        while True:
            print("\n" + "="*60)
            cmd = input(">> [Enter] 수집 시작 / [q] 프로그램 종료: ")
            
            if cmd.strip().lower() == 'q':
                break
            
            # 수집 시작 알림
            speak("시작")
            print("🟢 수집 중... (중지하려면 [Enter]를 누르세요)")
            
            stop_flag = False
            def wait_for_stop():
                nonlocal stop_flag
                input() # 단순히 엔터 입력 대기
                stop_flag = True

            input_thread = threading.Thread(target=wait_for_stop, daemon=True)
            input_thread.start()

            ser.reset_input_buffer()
            ser.write(b's') # 아두이노 시작 신호
            
            collected_data = []
            count = 0

            while not stop_flag:
                if ser.in_waiting >= 18:
                    header = ser.read(1)
                    if header == b'\xAA':
                        if ser.read(1) == b'\x55':
                            raw_payload = ser.read(16)
                            if len(raw_payload) == 16:
                                data = struct.unpack('<Ihhhhhh', raw_payload)
                                collected_data.append(data)
                                count += 1
                                if count % 100 == 0:
                                    print(f"\r수집된 샘플 수: {count}", end="")
            
            ser.write(b's') # 아두이노 정지 신호
            print(f"\n\n[정지] 수집 완료. 총 {count}개.")

            # 수집 종료 알림
            speak(f"종료")

            # 저장 조건 확인
            if count < 1000:
                print("⚠️ 데이터가 1000개 미만이므로 저장하지 않습니다.")
                speak(f"데이터가 1000개 미만입니다.")
            else:
                FILE_NAME = f"imu_record_{time.strftime('%Y%m%d_%H%M%S')}.csv"
                FILE_PATH = os.path.join(save_dir, FILE_NAME)
                
                with open(FILE_PATH, mode='w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['timestamp_us', 'gx', 'gy', 'gz', 'ax', 'ay', 'az'])
                    writer.writerows(collected_data)
                
                saved_files_count += 1
                print(f"💾 저장 완료: {FILE_PATH}")
                print(f"📊 현재 세션 총 저장 파일: {saved_files_count}개")

    except KeyboardInterrupt:
        print("\n\n[강제 종료] 프로그램을 마칩니다.")
    finally:
        if 'ser' in locals() and ser.is_open:
            ser.write(b's')
            ser.close()

if __name__ == "__main__":
    main()