# Echoes of the body
Master Thesis in Music Engineering, Politecnico di Milano with collaboration IPEM - Ghent University

The project explores how real-time physiological signals, primarily breathing and heart rate, collected during the Shambhavi Mahamudra practice, can be used to:
- classify the practitioner's psychophysiological state,
- generate musical soundscapes driven by biofeedback.

The goal is to study whether these compositions can induce similar states in a secondary listener, bridging physiological awareness and musical interaction.

---

## 🎙️ Acquisition Module (`/acquisition`)

This folder contains the scripts necessary to **record synchronized physiological data** from two sources:

- **Zephyr BioHarness** (via serial port)
- **Custom belt** (via OSC messages from Ableton Live through Max for Live)

The system waits for a `/startRecording` message sent via OSC and then starts writing aligned data streams to CSV files. This ensures the beginning of the musical session matches exactly the start of physiological signal recording.



