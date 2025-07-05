""" Receive The AKIA clock and sensor data, record only after startRecording """
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer
import socket
from datetime import datetime
import csv
from pythonosc.udp_client import SimpleUDPClient


# Nome file output
#CSV_FILENAME = "custom_bio_data_pipeline.csv"
CSV_FILENAME = "custom_bio_data.csv"

# Flag per decidere quando registrare
recording = False

# Configura il server OSC
zephyr_client = SimpleUDPClient("127.0.0.1", 6576)  # IP localhost, porta scelta per Zephyr


# Crea e apre il file CSV
csv_file = open(CSV_FILENAME, "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["local_timestamp", "bio_time", "BF", "HR"])


# Handler per messaggi /bio
def bio_handler(address, *args):
    global recording
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    print(f"\n Ricevuto {address}: {args}")

    if not recording:
        print("Ignorato perché recording = False")
        return

    # Se sai che sono time, bf, hr:
    if len(args) == 3:
        time_val, bf, hr = args
        print(f" SALVO: Time: {time_val} | BF: {bf} | HR: {hr}")
        csv_writer.writerow([now, time_val, bf, hr])
    else:
        print("Dati /bio con numero di argomenti inaspettato:", args)


# Handler per avvio registrazione
def start_recording_handler(address, *args):
    global recording
    recording = True
    print(f"Registrazione {address}: {args}")
    zephyr_client.send_message("/startRecording", [])

def stop_recording_handler(address, *args):
    global recording
    recording = False
    print("Registrazione FERMATA")
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

        print(f"In ascolto su {ip}:{port} (path /bio e /startRecording)...")
        server = BlockingOSCUDPServer((ip, port), dispatcher)
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n Interruzione manuale.")
    finally:
        csv_file.close()
        print(f"File CSV salvato come: {CSV_FILENAME}")



