""" Receive The AKIA clock and sensor data, record only after startRecording """
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer
import socket
from datetime import datetime
import csv
from pythonosc.udp_client import SimpleUDPClient


# output file
#CSV_FILENAME = "custom_bio_data_pipeline.csv"
CSV_FILENAME = "custom_bio_data.csv"

# To record
recording = False

# Server OSC configuration
zephyr_client = SimpleUDPClient("127.0.0.1", 6576)  # IP localhost, port for Zephyr


# CSV
csv_file = open(CSV_FILENAME, "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["local_timestamp", "bio_time", "BF", "HR"])


# Handler for /bio
def bio_handler(address, *args):
    global recording
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    print(f"\n Received {address}: {args}")

    if not recording:
        print("Don't saved because recording = False")
        return

    # I know that I have: time, bf, hr
    if len(args) == 3:
        time_val, bf, hr = args
        print(f" Save: Time: {time_val} | BF: {bf} | HR: {hr}")
        csv_writer.writerow([now, time_val, bf, hr])
    else:
        print("Data /bio with unespected number of args:", args)


# Handler to start recording
def start_recording_handler(address, *args):
    global recording
    recording = True
    print(f"Recording {address}: {args}")
    zephyr_client.send_message("/startRecording", [])

def stop_recording_handler(address, *args):
    global recording
    recording = False
    print("Stop recording")
    zephyr_client.send_message("/stopRecording", [])

# Default handler
def default_handler(address, *args):
    print(f"DEFAULT {address}: {args}")

if __name__ == "__main__":
    try:
        dispatcher = Dispatcher()
        dispatcher.set_default_handler(default_handler)

        dispatcher.map("/bio", bio_handler)
        dispatcher.map("/startRecording", start_recording_handler)
        dispatcher.map("/stopRecording", stop_recording_handler)

        ip = socket.gethostbyname(socket.gethostname())
        port = 6575

        print(f"{ip}:{port} (path /bio and /startRecording)...")
        server = BlockingOSCUDPServer((ip, port), dispatcher)
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n keyboard interrupt")
    finally:
        csv_file.close()
        print(f"CSV saved: {CSV_FILENAME}")



