import json
from pathlib import Path
from typing import List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ------------------ Config ------------------
FAQ_PATH = Path("data/faqs.json")
CONFIDENCE_THRESHOLD = 0.25  # is se kam ho to "sorry I don't know" bolo
SHOW_DEBUG = False           # True karo to matched question + score bhi dikhega
# --------------------------------------------

# --- NLP (spaCy) for proper preprocessing ---
# Safe loader + fallback so script doesn't crash if model missing.
try:
    import spacy
    _NLP = None

    def _ensure_spacy():
        """Load spaCy model once; raise with a helpful message if missing."""
        global _NLP
        if _NLP is None:
            try:
                _NLP = spacy.load("en_core_web_sm", disable=["ner", "parser"])
            except Exception:
                print("⚠️ spaCy model not found. Run:\n"
                      "   python -m spacy download en_core_web_sm")
                raise
        return _NLP

    def spacy_lemmas(text: str):
        """Tokenize + lowercase + remove stopwords + keep alpha + lemmatize."""
        nlp = _ensure_spacy()
        doc = nlp(text.lower())
        return [t.lemma_ for t in doc if t.is_alpha and not t.is_stop]

    _HAS_SPACY = True
except Exception:
    # If spaCy import/load fails, we'll fall back to plain sklearn preprocessing
    _HAS_SPACY = False
    spacy_lemmas = None  # name defined for type hints

def load_faqs(path: Path) -> Tuple[List[str], List[str]]:
    if not path.exists():
        raise FileNotFoundError(f"FAQ file not found at: {path.resolve()}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError("faqs.json is empty or not a list of Q/A objects.")
    questions = [str(row["question"]).strip() for row in data if "question" in row and "answer" in row]
    answers   = [str(row["answer"]).strip()   for row in data if "question" in row and "answer" in row]
    if not questions:
        raise ValueError("faqs.json has no valid {question, answer} pairs.")
    return questions, answers

def build_index(questions: List[str]):
    # Prefer spaCy-based analyzer (better matching); fallback to default if unavailable.
    if _HAS_SPACY:
        # callable analyzer ke sath ngram_range pass nahi karte (warna warning aati hai)
        vectorizer = TfidfVectorizer(analyzer=spacy_lemmas)
    else:
        vectorizer = TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2))
    X = vectorizer.fit_transform(questions)
    return vectorizer, X

def best_match(user_query: str, vectorizer, X, questions: List[str]) -> Tuple[int, float]:
    q_vec = vectorizer.transform([user_query])
    sims = cosine_similarity(q_vec, X)[0]
    best_id = int(sims.argmax())
    return best_id, float(sims[best_id])

def main():
    try:
        questions, answers = load_faqs(FAQ_PATH)
        vectorizer, X = build_index(questions)

        print("🤖 FAQ Chatbot ready! Type 'exit' to quit.\n")
        while True:
            user = input("You: ").strip()
            if not user:
                continue
            if user.lower() in {"exit", "quit"}:
                print("Bot: Bye!")
                break

            idx, score = best_match(user, vectorizer, X, questions)
            if score < CONFIDENCE_THRESHOLD:
                print("Bot: Sorry, mujhe sure nahi. Kya aap apna sawal thoda aur clear kar sakte hain?")
            else:
                print(f"Bot: {answers[idx]}")
                if SHOW_DEBUG:
                    print(f'(matched: "{questions[idx]}" | score={score:.2f})')

    except KeyboardInterrupt:
        print("\nBot: Bye!")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
