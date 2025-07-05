import serial
import time
from datetime import datetime
import csv
from pythonosc import dispatcher, osc_server
import threading

PYTHON_ZEPHYR = "C:/Users/cecil/AppData/Local/Programs/Python/Python39/python.exe"

# Impostazioni
PORT = "COM9"
BAUDRATE = 115200
ACQUISITION_DELAY = 4.0

recording = False
sync_epoch = 0
ser = None
stop_event = None
buffer = bytearray()

COMMANDS = {
    "lifesign": (0x23, []),
    "enable_ecg": (0x16, [1]),
    "enable_breathing": (0x15, [1]),
    "enable_rr": (0x19, [1]),
    "enable_accel": (0x1E, [1]),
    "summary_interval": (0xBD, [1, 0])
}

def create_frame(msg_id, payload):
    dlc = len(payload)
    crc = 0
    for b in payload:
        crc ^= b
        for _ in range(8):
            crc = (crc >> 1) ^ 0x8C if (crc & 1) else (crc >> 1)
    return bytes([0x02, msg_id, dlc] + payload + [crc, 0x03])

def send_command(ser, name):
    msg_id, payload = COMMANDS[name]
    frame = create_frame(msg_id, payload)
    ser.write(frame)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Sent: {name}")

def lifesign_loop(ser, stop_event):
    while not stop_event.is_set():
        send_command(ser, "lifesign")
        time.sleep(1)

def parse_ecg_waveform(payload):
    if len(payload) < 9:
        return []
    signal_bytes = payload[9:]
    bitstream = int.from_bytes(signal_bytes, byteorder='little')
    samples = []
    for i in range((len(signal_bytes) * 8) // 10):
        val = (bitstream >> (i * 10)) & 0x3FF
        val -= 512
        ecg_mv = val * 0.025
        samples.append(ecg_mv)
    return samples

def parse_breathing_samples(payload):
    if len(payload) < 9:
        return []
    signal_bytes = payload[9:]
    bitstream = int.from_bytes(signal_bytes, byteorder='little')
    samples = []
    for i in range((len(signal_bytes) * 8) // 10):
        val = (bitstream >> (i * 10)) & 0x3FF
        val -= 512
        samples.append(val)
    return samples

def parse_10bit_waveform(payload):
    if len(payload) < 9:
        return []
    signal_bytes = payload[9:]
    bitstream = int.from_bytes(signal_bytes, byteorder='little')
    samples = []
    for i in range((len(signal_bytes) * 8) // 10):
        val = (bitstream >> (i * 10)) & 0x3FF
        val -= 512
        samples.append(val)
    return samples

def handle_breathing_packet(payload, timestamp, writer):
    global sync_epoch
    samples = parse_breathing_samples(payload)
    sampling_rate = 25.0
    valid_samples = []
    for i, s in enumerate(samples):
        ts_adjusted = (timestamp - ACQUISITION_DELAY) - (len(samples) - i - 1) / sampling_rate
        if ts_adjusted >= sync_epoch:
          valid_samples.append([ts_adjusted, s])
        else:
            print(f"[DEBUG] scarto ts_adjusted {ts_adjusted:.3f} < sync_epoch {sync_epoch:.3f}")

    for ts, s in valid_samples:
        writer.writerow([ts, "breathing", s])


def handle_ecg_packet(payload, timestamp, writer):
    global sync_epoch
    samples = parse_ecg_waveform(payload)
    rate = 250.0
    for i, sample in enumerate(samples):
        ts = (timestamp - ACQUISITION_DELAY) - (len(samples) - i - 1) / rate
        if ts >= sync_epoch:
            writer.writerow([ts, "ecg", sample])
            print(f"ECG sample: {sample}")

def handle_accel_packet(payload, timestamp, writer):
    global sync_epoch
    ts = timestamp - ACQUISITION_DELAY
    if ts < sync_epoch:
        return
    try:
        x = int.from_bytes(payload[0:2], byteorder='little', signed=True)
        y = int.from_bytes(payload[2:4], byteorder='little', signed=True)
        z = int.from_bytes(payload[4:6], byteorder='little', signed=True)
        writer.writerow([ts, "accel", x, y, z])
        print(f"x: {x}, y: {y}, z: {z}")
    except Exception as e:
        print(f"[ERROR] Accel parsing: {e}")

def handle_summary_packet(payload, timestamp, writer):
    global sync_epoch
    ts = timestamp - ACQUISITION_DELAY
    if ts < sync_epoch:
        return
    try:
        hr = int.from_bytes(payload[10:12], 'little')
        rr_raw = int.from_bytes(payload[12:14], 'little')
        rr = rr_raw * 0.1
        if hr > 0:
            writer.writerow([ts, "heart_rate", hr])
            print(hr)
        if rr > 0:
            writer.writerow([ts, "respiration_rate", rr])
            print(rr)
    except Exception as e:
        print(f"[ERROR] Summary parsing: {e}")

def start_recording_handler(address, *args):
    global recording, ser, buffer, sync_epoch
    print(f">>> {address} — START recording")
    buffer.clear()
    ser.reset_input_buffer()
    sync_epoch = time.time()
    print(f"SYNC EPOCH = {sync_epoch}")
    recording = True

def stop_recording_handler(address, *args):
    global recording
    print(f">>> {address} — STOP recording")
    time.sleep(1.0)
    recording = False

def main():
    global ser, stop_event
    try:
        ser = serial.Serial(PORT, baudrate=BAUDRATE, timeout=2)
        ser.reset_input_buffer()
        for cmd in COMMANDS:
            send_command(ser, cmd)
            time.sleep(0.1)

        osc_dispatcher = dispatcher.Dispatcher()
        osc_dispatcher.map("/startRecording", start_recording_handler)
        osc_dispatcher.map("/stopRecording", stop_recording_handler)

        threading.Thread(
            target=lambda: osc_server.ThreadingOSCUDPServer(("0.0.0.0", 6576), osc_dispatcher).serve_forever(),
            daemon=True
        ).start()

        stop_event = threading.Event()
        threading.Thread(target=lifesign_loop, args=(ser, stop_event), daemon=True).start()

        with open("bio_data.csv", "w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            #writer.writerow(["timestamp", "signal", "value1", "value2", "value3"])
            writer.writerow(["timestamp", "signal", "value"])
            while True:
                byte = ser.read(1)
                if byte:
                    buffer.append(byte[0])
                    if buffer[0] != 0x02:
                        buffer.clear()
                        continue
                    if len(buffer) >= 3:
                        msg_id = buffer[1]
                        dlc = buffer[2]
                        if len(buffer) >= 3 + dlc + 2:
                            payload = buffer[3:3 + dlc]
                            crc = buffer[3 + dlc]
                            etx = buffer[3 + dlc + 1]
                            buffer.clear()
                            if etx != 0x03:
                                continue
                            timestamp = time.time()

                            if recording:
                                if msg_id == 0x21:
                                    handle_breathing_packet(payload, timestamp, writer)
                                elif msg_id == 0x22:
                                    handle_ecg_packet(payload, timestamp, writer)
                                elif msg_id == 0x2B:
                                    handle_summary_packet(payload, timestamp, writer)
                                #elif msg_id == 0x25:
                                #    handle_accel_packet(payload, timestamp, writer)
                            else:
                                print(f"[DEBUG] msg_id {hex(msg_id)} ignored (not recording)")
                else:
                    print("Timeout: no data received")

    except KeyboardInterrupt:
        print("Interrupted manually.")
    finally:
        if stop_event:
            stop_event.set()
        if ser and ser.is_open:
            ser.close()
        print("Serial port closed. CSV saved.")

if __name__ == "__main__":
    main()
