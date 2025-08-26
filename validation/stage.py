#!/usr/bin/env python3
# pip install python-osc
import argparse, unicodedata, re
from pythonosc.udp_client import SimpleUDPClient

def norm(s: str) -> str:
    # minuscolo + rimozione diacritici
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode("ascii").lower()
    # spazi e trattini -> underscore (mantieni underscore!)
    s = re.sub(r"[\s\-]+", "_", s)
    # compattazione underscore ripetuti e trim
    s = re.sub(r"_+", "_", s).strip("_")
    return s

ALIASES = {
    "pranayama":0,
    "chanting":1,
    "viparita_swasa":2,
    "breath_retention":3,
    "meditation":4,
}

def to_idx(v):
    # accetta 0..4, 1..5 o stringhe
    try:
        i = int(v)
        return i+1 if i in (0,1,2,3,4) else i
    except:
        k = norm(str(v))
        if k.isdigit():
            i = int(k);  return i+1 if i in (0,1,2,3,4) else i
        if k in ALIASES: return ALIASES[k]
        raise SystemExit(f"Label sconosciuta: {v!r}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, help="label o indice (0..4 / 1..5)")
    args = ap.parse_args()

    client = SimpleUDPClient("127.0.0.1", 9000)
    idx = int(to_idx(args.stage))
    client.send_message("/stage", idx)
    print(f"OK → /stage {idx} ")
