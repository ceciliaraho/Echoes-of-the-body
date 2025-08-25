import subprocess
import time
from datetime import datetime

PYTHON_CUSTOM = "C:/Users/cecil/Desktop/akia-env/Scripts/python.exe"
PYTHON_ZEPHYR = "C:/Users/cecil/AppData/Local/Programs/Python/Python39/python.exe"

CUSTOM_SCRIPT = "retrieveClock.py"   # script to receive OSC 
ZEPHYR_SCRIPT = "zephyr_data.py"     # zephyr


# Output log file
#LOG_CUSTOM = open("log_custom_pipeline.txt", "w")
#LOG_ZEPHYR = open("log_zephyr_pipeline.txt", "w")
LOG_CUSTOM = open("log_custom_session.txt", "w")
LOG_ZEPHYR = open("log_zephyr_session.txt", "w")

# File synch timestamp
SYNC_FILE = "sync_start_timestamp.txt"

def start_script(python_path, script_name, log_file):
    return subprocess.Popen(
        [python_path, script_name],
        stdout=log_file,
        stderr=subprocess.STDOUT,
        bufsize=1,
        universal_newlines=True
    )

def main():
    print("Synchronization Zephyr + OSC Receiver")

    # Timestamp
    sync_time = time.time()
    human_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    print(f"Sync time: {human_time} (epoch: {sync_time})")

    # File saved
    with open(SYNC_FILE, "w") as f:
        f.write(f"Start Time: {human_time} (epoch: {sync_time})\n")

    zephyr_proc = start_script(PYTHON_ZEPHYR, ZEPHYR_SCRIPT, LOG_ZEPHYR)
    custom_proc = start_script(PYTHON_CUSTOM, CUSTOM_SCRIPT, LOG_CUSTOM)
    
    print("CTRL+C to interrupt.")

    try:
        zephyr_proc.wait()
        custom_proc.wait()
        
    except KeyboardInterrupt:
        print("\n Close the process...")
        
        custom_proc.terminate()
        zephyr_proc.terminate()
    finally:
        LOG_CUSTOM.close()
        LOG_ZEPHYR.close()
        print(" Log saved.")

if __name__ == "__main__":
    main()
