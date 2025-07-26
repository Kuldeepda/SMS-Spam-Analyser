import streamlit as st
import pickle
import string
import nltk
import os
import ssl
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Handle SSL certificate issues for NLTK downloads
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Set up NLTK data directory for Render
nltk_data_dir = os.path.join(os.getcwd(), 'nltk_data')
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)
nltk.data.path.append(nltk_data_dir)

# Download required NLTK data with error handling
@st.cache_resource
def download_nltk_data():
    """Download NLTK data only once"""
    resources_to_download = [
        ('punkt_tab', 'tokenizers/punkt_tab'),
        ('stopwords', 'corpora/stopwords')
    ]
    
    for resource_name, resource_path in resources_to_download:
        try:
            nltk.data.find(resource_path)
            print(f"{resource_name} already exists")
        except LookupError:
            print(f"Downloading {resource_name}...")
            try:
                nltk.download(resource_name, download_dir=nltk_data_dir, quiet=True)
                print(f"Successfully downloaded {resource_name}")
            except Exception as e:
                print(f"Error downloading {resource_name}: {e}")
                # Fallback: try downloading to default location
                try:
                    nltk.download(resource_name, quiet=True)
                except Exception as e2:
                    print(f"Fallback download also failed for {resource_name}: {e2}")

# Download NLTK data
download_nltk_data()

# Initialize stemmer
ps = PorterStemmer()

def transform_text(text):
    """Transform text with proper error handling"""
    try:
        text = text.lower()
        
        # Tokenize with error handling
        try:
            text = nltk.word_tokenize(text)
        except LookupError:
            # Fallback: download punkt_tab again
            nltk.download('punkt_tab', quiet=True)
            text = nltk.word_tokenize(text)
        
        # Keep only alphanumeric tokens
        y = []
        for i in text:
            if i.isalnum():
                y.append(i)
        text = y[:]
        y.clear()
        
        # Remove stopwords and punctuation
        try:
            stop_words = stopwords.words('english')
        except LookupError:
            # Fallback: download stopwords again
            nltk.download('stopwords', quiet=True)
            stop_words = stopwords.words('english')
            
        for i in text:
            if i not in stop_words and i not in string.punctuation:
                y.append(i)
        text = y[:]
        y.clear()
        
        # Apply stemming
        for i in text:
            y.append(ps.stem(i))
            
        return " ".join(y)
    
    except Exception as e:
        st.error(f"Error in text transformation: {e}")
        return text.lower()  # Return basic lowercase as fallback

@st.cache_resource
def load_models():
    """Load pickle models with error handling"""
    try:
        tfidf = pickle.load(open('vectorizer.pkl','rb'))
        model = pickle.load(open('model.pkl','rb'))
        return tfidf, model
    except FileNotFoundError as e:
        st.error(f"Model files not found: {e}")
        return None, None
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

# Load models
tfidf, model = load_models()

# UI
st.title("Email/SMS Spam Classifier")

if tfidf is None or model is None:
    st.error("Failed to load models. Please ensure 'vectorizer.pkl' and 'model.pkl' are in the correct directory.")
else:
    input_sms = st.text_area("Enter the message")
    
    if st.button('Predict'):
        if input_sms.strip():
            try:
                # 1. preprocess
                transformed_sms = transform_text(input_sms)
                
                # 2. vectorize
                vector_input = tfidf.transform([transformed_sms])
                
                # 3. predict
                result = model.predict(vector_input)[0]
                
                # 4. Display
                if result == 1:
                    st.header("🚨 Spam")
                    st.error("This message appears to be spam!")
                else:
                    st.header("✅ Not Spam")
                    st.success("This message appears to be legitimate!")
                    
            except Exception as e:
                st.error(f"Error during prediction: {e}")
        else:
            st.warning("Please enter a message to classify.")
