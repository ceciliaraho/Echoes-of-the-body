#!/usr/bin/env python3
# pip install python-osc
import argparse, unicodedata, re
from pythonosc.udp_client import SimpleUDPClient

def norm(s):
    s = ''.join(c for c in unicodedata.normalize('NFKD', s) if not unicodedata.combining(c))
    return re.sub(r'[^a-z0-9]+','', s.lower())

ALIASES = {
    "pranayama":1,
    "chanting":0,
    "viparita_swasa":2,
    "retention":3,
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
