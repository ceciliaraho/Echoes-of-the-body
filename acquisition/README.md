
# Bio-Adaptive Soundscape for Breath-Based Meditation

This repository contains the code and data used for Cecilia Raho's master's thesis project, developed at IPEM - Ghent University.

The project explores how real-time physiological signals — primarily breathing and heart rate — collected during the Shambhavi Mahamudra practice, can be used to:
- classify the practitioner's psychophysiological state,
- and generate adaptive musical soundscapes driven by biofeedback.

The goal is to study whether these adaptive compositions can induce similar states in a secondary listener, bridging physiological awareness and musical interaction.


---

### 📂 Folder Contents

| File                   | Description |
|------------------------|-------------|
| `zephyr_data.py`       | Connects to the Zephyr BioHarness device via serial port (`COM9`), enables ECG, respiration, RR and HR signals, and records real-time data to `bio_data.csv`. |
| `retrieveClock.py`     | Receives OSC messages (`/bio`, `/startRecording`, `/stopRecording`) from a Max for Live patch in Ableton and records breathing frequency and heart rate to `custom_bio_data.csv`. |
| `sync_with_start.py`   | Launches both `zephyr_data.py` and `retrieveClock.py` in parallel, creates a shared synchronization timestamp (`sync_start_timestamp.txt`), and logs both processes. |

---

### ▶️ How to Run the Acquisition Pipeline

1. **Create the Python virtual environment for the custom belt (Python 3.8)**:
   ```bash
   py -3.8 -m venv akia-env
   .\akia-env\Scripts\activate
   pip install python-osc
   ```

2. **Ensure OSC messages are correctly routed:**
   - Set the **target IP address** in Max for Live to your computer's local IPv4 address (e.g., `192.168.1.24`).
   - Find your IP using:
     ```bash
     ipconfig
     ```
   - If using Norton or another antivirus, **disable the firewall** or allow traffic on ports `6575` (custom belt) and `6576` (Zephyr).

3. **Run the synchronization script:**
   ```bash
   python acquisition/sync_with_start.py
   ```

   This will:
   - Save the shared session start timestamp to `sync_start_timestamp.txt`
   - Automatically launch the two recording scripts
   - Write output files: `bio_data.csv` and `custom_bio_data.csv`

4. **Start the recording from Ableton Live:**
   - When you trigger the session in Ableton via Max for Live, it sends `/startRecording` via OSC.
   - At that moment, both Zephyr and the custom belt start saving data in sync.

---

### 📄 Output Files

| File                     | Description |
|--------------------------|-------------|
| `bio_data.csv`           | Recorded data from Zephyr (breathing waveform, HR, ECG, RR) |
| `custom_bio_data.csv`    | Recorded data from the custom belt (BF and HR via OSC) |
| `sync_start_timestamp.txt` | Shared timestamp of the session start |
| `log_zephyr_session.txt` | Log of the Zephyr data recording session |
| `log_custom_session.txt` | Log of the OSC receiver session |
