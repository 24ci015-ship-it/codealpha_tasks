import streamlit as st
import string
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download stopwords
nltk.download("stopwords")

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="FAQ Chatbot",
    page_icon="🤖",
    layout="centered"
)

# -----------------------------
# FAQ DATASET
# -----------------------------
faqs = [
    {
        "question": "What is your return policy?",
        "answer": "You can return products within 30 days of purchase."
    },
    {
        "question": "How can I track my order?",
        "answer": "You can track your order using the tracking link sent to your email."
    },
    {
        "question": "Do you offer international shipping?",
        "answer": "Yes, we ship to most countries worldwide."
    },
    {
        "question": "How can I contact customer support?",
        "answer": "You can contact us via email at support@example.com."
    },
    {
        "question": "What payment methods do you accept?",
        "answer": "We accept Credit Cards, Debit Cards, UPI, Net Banking and PayPal."
    },
    {
        "question": "How long does delivery take?",
        "answer": "Delivery usually takes 3 to 7 business days."
    },
    {
        "question": "Can I cancel my order?",
        "answer": "Yes. Orders can be cancelled before shipment."
    },
    {
        "question": "Do you provide cash on delivery?",
        "answer": "Yes, Cash on Delivery is available in selected locations."
    }
]

# -----------------------------
# STOP WORDS
# -----------------------------
stop_words = set(stopwords.words("english"))

# -----------------------------
# PREPROCESSING
# -----------------------------
def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

# -----------------------------
# VECTORIZATION
# -----------------------------
questions = [preprocess(faq["question"]) for faq in faqs]

vectorizer = TfidfVectorizer()

question_vectors = vectorizer.fit_transform(questions)

# -----------------------------
# CHATBOT FUNCTION
# -----------------------------
def chatbot_response(user_question):

    processed_question = preprocess(user_question)

    user_vector = vectorizer.transform([processed_question])

    similarity = cosine_similarity(user_vector, question_vectors)

    best_match = similarity.argmax()

    confidence = similarity[0][best_match]

    if confidence < 0.20:
        return (
            "Sorry! I couldn't find a suitable answer.",
            confidence
        )

    return (
        faqs[best_match]["answer"],
        confidence
    )

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.title("🤖 FAQ Chatbot")

st.write("### Ask questions about our services")

st.write(
    "This chatbot uses **NLP (NLTK)**, **TF-IDF**, and **Cosine Similarity** "
    "to find the most relevant FAQ."
)

# Chat history
if "history" not in st.session_state:
    st.session_state.history = []

user_input = st.text_input(
    "Enter your question:"
)

col1, col2 = st.columns(2)

with col1:
    ask = st.button("Ask")

with col2:
    clear = st.button("Clear Chat")

if clear:
    st.session_state.history = []
    st.rerun()

if ask:

    if user_input.strip() == "":
        st.warning("Please enter a question.")

    else:

        answer, confidence = chatbot_response(user_input)

        st.session_state.history.append(
            {
                "question": user_input,
                "answer": answer,
                "confidence": confidence
            }
        )

# -----------------------------
# DISPLAY CHAT
# -----------------------------
for chat in reversed(st.session_state.history):

    st.markdown("### 👤 You")
    st.info(chat["question"])

    st.markdown("### 🤖 Bot")
    st.success(chat["answer"])

    st.caption(
        f"Similarity Score: {chat['confidence']:.2f}"
    )