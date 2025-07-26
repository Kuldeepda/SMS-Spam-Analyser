import streamlit as st
import pickle
import string
import nltk
import os
import ssl
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Configure Streamlit for faster loading
st.set_page_config(
    page_title="Spam Classifier",
    page_icon="📧",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Handle SSL certificate issues
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Set up NLTK data directory
nltk_data_dir = os.path.join(os.getcwd(), 'nltk_data')
if os.path.exists(nltk_data_dir):
    nltk.data.path.append(nltk_data_dir)

@st.cache_resource(show_spinner=False)
def initialize_nltk():
    """Initialize NLTK data and stemmer - cached to avoid re-loading"""
    try:
        # Check if data exists first
        try:
            nltk.data.find('tokenizers/punkt_tab')
            stopwords.words('english')  # Test stopwords access
            print("NLTK data found, skipping download")
        except LookupError:
            print("Downloading NLTK data...")
            nltk.download('punkt_tab', quiet=True)
            nltk.download('stopwords', quiet=True)
        
        # Initialize and return stemmer
        return PorterStemmer(), True
    except Exception as e:
        print(f"NLTK initialization error: {e}")
        return None, False

@st.cache_resource(show_spinner=False)
def load_models():
    """Load models once and cache them"""
    try:
        with open('vectorizer.pkl', 'rb') as f:
            tfidf = pickle.load(f)
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        return tfidf, model, True
    except Exception as e:
        return None, None, False

# Initialize everything at startup (cached)
ps, nltk_ready = initialize_nltk()
tfidf, model, models_ready = load_models()

def transform_text(text):
    """Fast text transformation"""
    if not nltk_ready or ps is None:
        # Fallback: simple processing
        return " ".join(word.lower() for word in text.split() if word.isalnum())
    
    try:
        text = text.lower()
        text = nltk.word_tokenize(text)
        
        # Filter alphanumeric only
        text = [word for word in text if word.isalnum()]
        
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        text = [word for word in text if word not in stop_words]
        
        # Apply stemming
        text = [ps.stem(word) for word in text]
        
        return " ".join(text)
    except:
        # Fallback on any error
        return " ".join(word.lower() for word in text.split() if word.isalnum())

# UI with loading indicator
def main():
    st.title("📧 Email/SMS Spam Classifier")
    
    # Show status
    if not models_ready:
        st.error("❌ Models not loaded properly")
        st.stop()
    
    if not nltk_ready:
        st.warning("⚠️ Using simplified text processing")
    
    # Input
    input_sms = st.text_area(
        "Enter your message:",
        placeholder="Type your message here...",
        height=100
    )
    
    # Predict button
    if st.button('🔍 Analyze Message', type="primary"):
        if input_sms.strip():
            # Show progress
            with st.spinner('Processing...'):
                try:
                    # Transform text
                    transformed_sms = transform_text(input_sms)
                    
                    # Vectorize and predict
                    vector_input = tfidf.transform([transformed_sms])
                    result = model.predict(vector_input)[0]
                    
                    # Get probability if available
                    try:
                        prob = model.predict_proba(vector_input)[0]
                        confidence = prob[result]
                    except:
                        confidence = None
                    
                    # Display result
                    if result == 1:
                        st.error("🚨 SPAM DETECTED")
                        if confidence:
                            st.write(f"Confidence: {confidence:.1%}")
                    else:
                        st.success("✅ NOT SPAM")
                        if confidence:
                            st.write(f"Confidence: {confidence:.1%}")
                            
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        else:
            st.warning("Please enter a message to analyze")

if __name__ == "__main__":
    main()
