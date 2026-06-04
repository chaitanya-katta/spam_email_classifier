import streamlit as st
import pickle
import nltk

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Email Spam Classifier",
    page_icon="📧",
    layout="wide"
)

# ==========================================
# LOAD MODEL & VECTORIZER
# ==========================================

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# ==========================================
# NLP PREPROCESSING
# ==========================================

ps = PorterStemmer()
stop_words = set(stopwords.words("english"))

def transform_text(text):

    text = str(text).lower()

    words = nltk.word_tokenize(text)

    words = [word for word in words if word.isalnum()]

    words = [word for word in words if word not in stop_words]

    words = [ps.stem(word) for word in words]

    return " ".join(words)

# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.title("📊 Model Information")

st.sidebar.markdown("### 🤖 Algorithm")
st.sidebar.write("Logistic Regression")

st.sidebar.markdown("### 🔤 Feature Extraction")
st.sidebar.write("TF-IDF Vectorization")

st.sidebar.markdown("### 📂 Dataset Information")
st.sidebar.write("Total Emails : 5,171")
st.sidebar.write("Ham Emails : 3,672")
st.sidebar.write("Spam Emails : 1,499")

st.sidebar.markdown("### 📈 Model Performance")
st.sidebar.write("Accuracy : 98.45%")
st.sidebar.write("Precision : 96.10%")
st.sidebar.write("Recall : 98.67%")
st.sidebar.write("F1 Score : 97.37%")

st.sidebar.markdown("### ⚙️ NLP Pipeline")
st.sidebar.write("• Lowercasing")
st.sidebar.write("• Tokenization")
st.sidebar.write("• Stopword Removal")
st.sidebar.write("• Stemming")

# ==========================================
# HEADER
# ==========================================

st.title("📧 Email Spam Classifier")

st.markdown("""
This project uses **Natural Language Processing (NLP)** and **Machine Learning**
to classify emails as **Spam** or **Ham (Not Spam)**.

### Technologies Used
- Python
- NLTK
- Scikit-Learn
- Streamlit

### Workflow
1. Text Preprocessing
2. Tokenization
3. Stopword Removal
4. Stemming
5. TF-IDF Vectorization
6. Logistic Regression Classification
""")

# ==========================================
# PERFORMANCE METRICS
# ==========================================

st.subheader("📊 Model Performance")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Accuracy", "98.45%")

with col2:
    st.metric("Precision", "96.10%")

with col3:
    st.metric("Recall", "98.67%")

with col4:
    st.metric("F1 Score", "97.37%")

st.markdown("---")

# ==========================================
# SAMPLE EMAILS
# ==========================================

st.subheader("📨 Sample Emails")

col1, col2 = st.columns(2)

with col1:

    if st.button("Load Ham Email"):

        st.session_state.email_text = """
Hi Team,

Please find today's project update attached.

The development work is progressing as planned and we will discuss the remaining tasks in tomorrow's meeting.

Regards,
Chaitanya
"""

with col2:

    if st.button("Load Spam Email"):

        st.session_state.email_text = """
Congratulations!

You have been selected to receive a free iPhone.

Click the link below immediately to claim your reward.
"""

# ==========================================
# INPUT AREA
# ==========================================

st.subheader("✍️ Enter Email Content")

input_text = st.text_area(
    "",
    value=st.session_state.get("email_text", ""),
    height=250,
    placeholder="Paste your email content here..."
)

# ==========================================
# PREDICTION
# ==========================================

if st.button("Predict"):

    if input_text.strip() == "":

        st.warning("Please enter an email message.")

    else:

        transformed_text = transform_text(input_text)

        if len(transformed_text.split()) < 3:

            st.warning(
                "Very short messages may not be classified accurately because the model was trained on full email content."
            )

        vector_input = vectorizer.transform([transformed_text])

        prediction = model.predict(vector_input)[0]

        probabilities = model.predict_proba(vector_input)[0]

        ham_probability = probabilities[0] * 100
        spam_probability = probabilities[1] * 100

        st.markdown("---")

        st.subheader("🎯 Prediction Result")

        # label mapping:
        # ham = 0
        # spam = 1

        if prediction == 1:

            st.error(
                f"🚨 Spam Email ({spam_probability:.2f}% confidence)"
            )

        else:

            st.success(
                f"✅ Ham / Not Spam ({ham_probability:.2f}% confidence)"
            )

        # ==========================================
        # CONFIDENCE SCORES
        # ==========================================

        st.subheader("📈 Confidence Scores")

        st.write(f"Ham Probability : {ham_probability:.2f}%")
        st.progress(int(ham_probability))

        st.write(f"Spam Probability : {spam_probability:.2f}%")
        st.progress(int(spam_probability))

        # ==========================================
        # DEBUG / TECHNICAL DETAILS
        # ==========================================

        with st.expander("🔍 View Technical Details"):

            st.write("Processed Text")
            st.code(transformed_text)

            st.write("Prediction Value")
            st.write(prediction)

            st.write("Active TF-IDF Features")
            st.write(vector_input.nnz)

# ==========================================
# FOOTER
# ==========================================

st.markdown("---")

st.markdown("""
### 👨‍💻 Developer

**Chaitanya**

Machine Learning & Data Science Enthusiast

**Technologies Used**
- Python
- NLTK
- Scikit-Learn
- Logistic Regression
- TF-IDF
- Streamlit
""")