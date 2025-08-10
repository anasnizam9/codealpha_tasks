import json
from pathlib import Path
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

data = json.loads(Path("faqs.json").read_text(encoding="utf-8"))
questions = [d["question"] for d in data]
answers = [d["answer"] for d in data]

vectorizer = TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1,2))
X = vectorizer.fit_transform(questions)

st.title("FAQ Chatbot")
q = st.text_input("Ask a question")
if q:
    sims = cosine_similarity(vectorizer.transform([q]), X)[0]
    i = int(sims.argmax())
    score = float(sims[i])
    if score < 0.25:
        st.write("**Bot:** Sorry, I’m not sure. Can you rephrase?")
    else:
        st.write(f"**Bot:** {answers[i]}")
        with st.expander("Why this answer?"):
            st.write(f"Matched FAQ: **{questions[i]}** (score {score:.2f})")
