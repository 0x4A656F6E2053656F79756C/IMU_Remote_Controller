import serial
import time
import struct
import csv
import os

SERIAL_PORT = "/dev/cu.usbmodem11301"
BAUD_RATE = 921600

FRAME_SIZE = 18
SYNC = b'\xAA\x55'

# 파일 이름 자동 생성
base_filename = "imu_data"
extension = ".csv"
counter = 1

while os.path.exists(f"{base_filename}_{counter}{extension}"):
    counter += 1

filename = f"{base_filename}_{counter}{extension}"

# 🔥 blocking read로 변경 (중요)
ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
time.sleep(2)

print(f"Start collecting... (Saving to {filename})")

data = []
start_time = time.time()

sync_buf = b''

while True:
    if time.time() - start_time > 10:
        break

    # 🔥 sync 맞출 때까지 1바이트씩 읽기
    while sync_buf != SYNC:
        sync_buf = (sync_buf + ser.read(1))[-2:]

    # 🔥 나머지 프레임 읽기
    payload = ser.read(FRAME_SIZE - 2)

    if len(payload) != FRAME_SIZE - 2:
        continue

    try:
        t, gx, gy, gz, ax, ay, az = struct.unpack('<Ihhhhhh', payload)
        data.append((t, ax, ay, az, gx, gy, gz))
    except:
        continue

    sync_buf = b''  # 다음 sync 탐색

ser.close()

# CSV 저장
with open(filename, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["time_us","ax","ay","az","gx","gy","gz"])
    writer.writerows(data)

print(f"Finished! Saved {len(data)} rows to {filename}")

# 🔥 sampling rate 계산 (두 가지 방식)
duration = 10
rate_simple = len(data) / duration

print("Sampling rate (simple) ≈", rate_simple, "Hz")

# 🔥 timestamp 기반 계산 (더 정확)
if len(data) > 1:
    t0 = data[0][0]
    t1 = data[-1][0]
    rate_timestamp = (len(data)-1) / ((t1 - t0) / 1e6)
    print("Sampling rate (timestamp) ≈", rate_timestamp, "Hz")