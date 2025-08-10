# fetch_midis.py
import os
from music21 import corpus

OUT = "data/midi"
os.makedirs(OUT, exist_ok=True)

# Bach corpus (chorales + other works)
files = []
try:
    files += corpus.getBachChorales()
except Exception:
    pass
try:
    files += corpus.getComposer('bach')
except Exception:
    pass

# unique + sort
seen = set()
clean = []
for f in files:
    s = str(f)
    if s not in seen:
        seen.add(s)
        clean.append(f)

print(f"Found {len(clean)} Bach pieces in music21 corpus. Exporting to MIDI…")

saved = 0
for f in clean:
    try:
        sc = corpus.parse(f)
        base = os.path.basename(str(f)).split('.')[0]
        out_path = os.path.join(OUT, f"{base}.mid")
        sc.write('midi', fp=out_path)
        saved += 1
    except Exception as e:
        # kuch files fail ho sakti hain; skip
        pass

print(f"Done. Saved {saved} MIDI files to {OUT}")
        