import serial
import serial.tools.list_ports
import struct
import time
import csv
import os

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
        return ports[int(idx)].device

def main():
    port_name = select_port()
    if not port_name:
        return

    BAUD = 500000
    # 파일명: 현재 폴더에 imu_data_시간.csv 형식으로 저장
    FILE_NAME = f"imu_record_{time.strftime('%Y%m%d_%H%M%S')}.csv"

    try:
        ser = serial.Serial(port_name, BAUD, timeout=1)
        print(f"\n포트 연결 성공: {port_name}")
        time.sleep(2) # 아두이노 리셋 대기
    except Exception as e:
        print(f"포트 연결 실패: {e}")
        return

    print("\n" + "="*50)
    input(">> 엔터(Enter)를 누르면 아두이노가 데이터 전송을 시작합니다!")
    print("="*50)
    
    # 아두이노에 시작 신호('s') 전송
    ser.write(b's')
    
    f = None
    writer = None
    count = 0

    try:
        print("데이터 수집 중... (중지하려면 Ctrl + C를 누르세요)")
        while True:
            # 최소 1패킷(18바이트)이 쌓였을 때 읽기
            if ser.in_waiting >= 18:
                header = ser.read(1)
                if header == b'\xAA':
                    if ser.read(1) == b'\x55':
                        raw_payload = ser.read(16)
                        if len(raw_payload) == 16:
                            # 파싱: t(unsigned int), gx,gy,gz,ax,ay,az(short)
                            data = struct.unpack('<Ihhhhhh', raw_payload)
                            
                            if f is None:
                                f = open(FILE_NAME, mode='w', newline='')
                                writer = csv.writer(f)
                                writer.writerow(['timestamp_us', 'gx', 'gy', 'gz', 'ax', 'ay', 'az'])
                            
                            writer.writerow(data)
                            count += 1
                            
                            # 100개 단위로 진행 상황 출력
                            if count % 100 == 0:
                                print(f"\r수집된 샘플 수: {count}", end="")

    except KeyboardInterrupt:
        print("\n\n[정지] 사용자가 수집을 중단했습니다.")
        # 아두이노에도 정지 신호 전송 (필요 시)
        ser.write(b's')
    finally:
        if f:
            f.close()
            print(f"✅ 파일 저장 완료: {os.path.abspath(FILE_NAME)}")
        if 'ser' in locals() and ser.is_open:
            ser.close()

if __name__ == "__main__":
    main()