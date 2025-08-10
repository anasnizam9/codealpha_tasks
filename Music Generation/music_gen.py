import os, glob, random, json
import numpy as np
from tqdm import tqdm

# Deep learning
import tensorflow as tf
from tensorflow.keras import layers, callbacks

# MIDI
from music21 import converter, instrument, note, chord, stream

# -----------------------
# Config
# -----------------------
DATA_DIR = "data/midi"
SEQ_LEN = 50
BATCH_SIZE = 64
EPOCHS = 50
EMBED_DIM = 256
LSTM_UNITS = 512
CHECKPOINT_DIR = "checkpoints"
MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best.keras")
VOCAB_PATH = "vocab.json"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# -----------------------
# Utils: MIDI -> tokens
# -----------------------
def midi_to_tokens(midi_path):
    """Parse one MIDI to a list of tokens: '<pitches>_<dur>'"""
    try:
        score = converter.parse(midi_path)
    except Exception as e:
        print(f"Skip (parse fail): {midi_path} ({e})")
        return []

    parts = instrument.partitionByInstrument(score)
    if parts:
        elements = parts.parts[0].recurse().notes
    else:
        elements = score.flat.notes

    tokens = []
    for el in elements:
        try:
            dur = int(round(el.quarterLength * 4))  # duration in 1/4 beats
            if dur <= 0:
                dur = 1
            if isinstance(el, note.Note):
                tok = f"{el.pitch.midi}_{dur}"
            elif isinstance(el, chord.Chord):
                pitches = ".".join(str(n.pitch.midi) for n in el.notes)
                tok = f"{pitches}_{dur}"
            else:
                continue
            tokens.append(tok)
        except Exception:
            continue
    return tokens

def load_all_tokens(data_dir):
    files = sorted(glob.glob(os.path.join(data_dir, "**/*.mid*"), recursive=True))
    all_tokens = []
    for f in tqdm(files, desc="Parsing MIDI"):
        toks = midi_to_tokens(f)
        all_tokens.extend(toks)
    return all_tokens

# -----------------------
# Build sequences
# -----------------------
def build_vocab(tokens):
    vocab = sorted(set(tokens))
    tok2id = {t:i for i,t in enumerate(vocab)}
    id2tok = {i:t for t,i in tok2id.items()}
    return tok2id, id2tok

def make_sequences(tokens, tok2id, seq_len):
    ids = np.array([tok2id[t] for t in tokens if t in tok2id], dtype=np.int32)
    X, y = [], []
    for i in range(0, len(ids) - seq_len):
        X.append(ids[i:i+seq_len])
        y.append(ids[i+seq_len])
    X = np.array(X, dtype=np.int32)
    y = np.array(y, dtype=np.int32)
    return X, y

# -----------------------
# Model
# -----------------------
def build_model(vocab_size, seq_len):
    model = tf.keras.Sequential([
        layers.Input(shape=(seq_len,)),
        layers.Embedding(vocab_size, EMBED_DIM),
        layers.LSTM(LSTM_UNITS, return_sequences=True),
        layers.LSTM(LSTM_UNITS),
        layers.Dense(vocab_size, activation="softmax"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"]
    )
    return model

# -----------------------
# Sampling
# -----------------------
def sample_from_logits(probs, temperature=1.0):
    probs = np.asarray(probs).astype("float64")
    if temperature != 1.0:
        probs = np.log(probs + 1e-9) / max(temperature, 1e-6)
        probs = np.exp(probs)
    probs = probs / np.sum(probs)
    return np.random.choice(len(probs), p=probs)

def generate_tokens(model, seed_ids, length, temperature=0.9):
    out = list(seed_ids)
    for _ in range(length):
        x = np.array([out[-SEQ_LEN:]], dtype=np.int32)
        preds = model.predict(x, verbose=0)[0]
        next_id = sample_from_logits(preds, temperature)
        out.append(next_id)
    return out

# -----------------------
# Tokens -> MIDI
# -----------------------
def tokens_to_midi(tokens, out_path):
    s = stream.Stream()
    for tok in tokens:
        if "_" not in tok:
            continue
        pitch_part, dur_part = tok.rsplit("_", 1)
        try:
            dur = int(dur_part) / 4.0
        except:
            dur = 1.0
        if "." in pitch_part:
            pitches = [int(p) for p in pitch_part.split(".")]
            ch = chord.Chord(pitches)
            ch.quarterLength = dur
            s.append(ch)
        else:
            try:
                p = int(pitch_part)
            except:
                continue
            n = note.Note(p)
            n.quarterLength = dur
            s.append(n)
    s.write("midi", fp=out_path)
    return out_path

# -----------------------
# Train or Generate
# -----------------------
def main_train():
    print("Loading tokens…")
    tokens = load_all_tokens(DATA_DIR)
    if len(tokens) < 1000:
        raise RuntimeError("Data too small. Add more MIDI files in data/midi/")

    print(f"Total tokens: {len(tokens)}")
    tok2id, id2tok = build_vocab(tokens)
    with open(VOCAB_PATH, "w") as f:
        json.dump({"tok2id": tok2id, "id2tok": {int(k):v for k,v in id2tok.items()}}, f)

    X, y = make_sequences(tokens, tok2id, SEQ_LEN)
    print("X shape:", X.shape, "y shape:", y.shape)
    ds = tf.data.Dataset.from_tensor_slices((X, y)).shuffle(10000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    model = build_model(vocab_size=len(tok2id), seq_len=SEQ_LEN)
    ckpt = callbacks.ModelCheckpoint(MODEL_PATH, monitor="val_loss", save_best_only=True)
    es = callbacks.EarlyStopping(patience=5, restore_best_weights=True)

    history = model.fit(
        ds.take(int(len(X)*0.8/BATCH_SIZE)),  # train split
        validation_data=ds.skip(int(len(X)*0.8/BATCH_SIZE)),
        epochs=EPOCHS,
        callbacks=[ckpt, es],
        verbose=1
    )
    print("Training done. Best model saved at:", MODEL_PATH)

def main_generate(length=600, temperature=0.9):
    with open(VOCAB_PATH, "r") as f:
        vocab = json.load(f)
    tok2id = vocab["tok2id"]
    id2tok = {int(k):v for k,v in vocab["id2tok"].items()}

    model = build_model(vocab_size=len(tok2id), seq_len=SEQ_LEN)
    model.load_weights(MODEL_PATH)

    seed = [random.randrange(len(tok2id)) for _ in range(SEQ_LEN)]
    gen_ids = generate_tokens(model, seed, length=length, temperature=temperature)
    gen_tokens = [id2tok[i] for i in gen_ids]

    os.makedirs("outputs", exist_ok=True)
    out_mid = f"outputs/generated_t{temperature:.1f}_len{length}.mid"
    tokens_to_midi(gen_tokens, out_mid)
    print("Saved:", out_mid)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train","gen"], default="train")
    parser.add_argument("--length", type=int, default=600)
    parser.add_argument("--temperature", type=float, default=0.9)
    args = parser.parse_args()

    if args.mode == "train":
        main_train()
    else:
        main_generate(length=args.length, temperature=args.temperature)
