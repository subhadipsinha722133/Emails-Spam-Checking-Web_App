import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import string
import nltk
from nltk.corpus import stopwords
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Download stopwords
try:
    nltk.download('stopwords')
except:
    st.warning("Could not download NLTK stopwords. Using basic stopword list.")
    # Fallback to basic stopwords if NLTK download fails
    stopwords_list = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", 
                     "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 
                     'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 
                     'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 
                     'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 
                     'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 
                     'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 
                     'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 
                     'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 
                     'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 
                     'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 
                     'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 
                     'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 
                     'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', 
                     "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', 
                     "won't", 'wouldn', "wouldn't"]
else:
    stopwords_list = stopwords.words('english')

# Set page configuration
st.set_page_config(
    page_title="Spam Email Classifier",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
<style>
    .main-header {font-size: 3rem; color: #FF4B4B;}
    .section-header {font-size: 2rem; color: #1F77B4; border-bottom: 2px solid #1F77B4; padding-bottom: 0.3rem;}
    .positive {color: #00CC96;}
    .negative {color: #EF553B;}
    .info-box {background-color: #F0F2F6; padding: 20px; border-radius: 10px; margin-bottom: 20px;}
</style>
""", unsafe_allow_html=True)

# App title
st.markdown('<h1 class="main-header">📧 Spam Email Classifier</h1>', unsafe_allow_html=True)
st.write("This application uses a neural network to classify emails as spam or ham (not spam).")
st.write("Made By Subhadip 😎")

# Sidebar
st.sidebar.title("Navigation")
app_section = st.sidebar.radio("Go to", ["Data Overview", "Data Preprocessing", "Model Training", "Make Prediction"])

# Load data
@st.cache_data
def load_data():
    # Create a more comprehensive sample dataset
    sample_data = {
        'text': [
            "Subject: Free offer win money prize call now limited time",
            "Subject: Meeting reminder tomorrow at 10am conference room",
            "Subject: Your account statement is ready for review",
            "Subject: Limited time offer buy now get discount",
            "Subject: Project update from the team progress report",
            "Subject: Congratulations you won a lottery claim prize",
            "Subject: Weekly newsletter from company updates news",
            "Subject: Special offer cheap prices limited stock",
            "Subject: Lunch meeting next Wednesday restaurant booking",
            "Subject: Urgent request transfer funds bank account",
            "Subject: Investment opportunity high returns guaranteed",
            "Subject: Team building event Friday afternoon",
            "Subject: Make money fast easy work from home",
            "Subject: Quarterly report financial performance analysis",
            "Subject: Free trial subscription cancel anytime",
            "Subject: Client presentation materials preparation",
            "Subject: Discount coupon code expire soon",
            "Subject: Department meeting agenda items discussion",
            "Subject: Earn extra income part time job",
            "Subject: Vacation schedule approval time off"
        ],
        'label': ['spam', 'ham', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam',
                 'spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham']
    }
    return pd.DataFrame(sample_data)

data = load_data()

# Data Overview Section
if app_section == "Data Overview":
    st.markdown('<h2 class="section-header">Dataset Overview</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**First 5 rows of the dataset:**")
        st.dataframe(data.head())
        
    with col2:
        st.write("**Dataset Information:**")
        st.write(f"Number of emails: {len(data)}")
        st.write(f"Number of spam emails: {len(data[data['label'] == 'spam'])}")
        st.write(f"Number of ham emails: {len(data[data['label'] == 'ham'])}")
        
    # Original distribution plot
    st.write("**Original Class Distribution:**")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(x='label', data=data, ax=ax)
    ax.set_title("Original Distribution of Spam and Ham Emails")
    ax.set_xticklabels(['Ham (Not Spam)', 'Spam'])
    st.pyplot(fig)
    
    # Balance the data
    st.markdown('<h2 class="section-header">Balancing the Dataset</h2>', unsafe_allow_html=True)
    st.write("To address class imbalance, we downsample the majority class to match the number of emails in the minority class.")
    
    ham_msg = data[data['label'] == 'ham']
    spam_msg = data[data['label'] == 'spam']
    ham_msg_balanced = ham_msg.sample(n=len(spam_msg), random_state=42)
    balanced_data = pd.concat([ham_msg_balanced, spam_msg]).reset_index(drop=True)
    
    # Display balanced distribution
    st.write("**Balanced Class Distribution:**")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(x='label', data=balanced_data, ax=ax)
    ax.set_title("Balanced Distribution of Spam and Ham Emails")
    ax.set_xticklabels(['Ham (Not Spam)', 'Spam'])
    st.pyplot(fig)
    
    # Store balanced_data in session state for use in other sections
    st.session_state.balanced_data = balanced_data

# Data Preprocessing Section
elif app_section == "Data Preprocessing":
    st.markdown('<h2 class="section-header">Data Preprocessing</h2>', unsafe_allow_html=True)
    
    if 'balanced_data' not in st.session_state:
        st.warning("Please go to the Data Overview section first to load and balance the data.")
    else:
        balanced_data = st.session_state.balanced_data
        
        st.write("**Text Preprocessing Steps:**")
        st.write("1. Remove the word 'Subject' from emails")
        st.write("2. Remove punctuation")
        st.write("3. Remove stopwords")
        
        # Step 1: Remove 'Subject'
        balanced_data['text'] = balanced_data['text'].str.replace('Subject', '')
        
        # Step 2: Remove punctuation
        punctuations_list = string.punctuation
        def remove_punctuations(text):
            temp = str.maketrans('', '', punctuations_list)
            return text.translate(temp)
        
        balanced_data['text'] = balanced_data['text'].apply(lambda x: remove_punctuations(x))
        
        # Step 3: Remove stopwords
        def remove_stopwords(text):
            imp_words = []
            
            for word in str(text).split():
                word = word.lower()
                if word not in stopwords_list:
                    imp_words.append(word)
            
            return " ".join(imp_words)
        
        balanced_data['text'] = balanced_data['text'].apply(lambda text: remove_stopwords(text))
        
        # Display processed data
        st.write("**Processed Data (First 5 rows):**")
        st.dataframe(balanced_data.head())
        
        # Most frequent words visualization (alternative to wordcloud)
        st.write("**Most Frequent Words:**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Non-Spam Emails:")
            ham_text = " ".join(balanced_data[balanced_data['label'] == 'ham']['text'])
            ham_words = ham_text.split()
            ham_word_counts = Counter(ham_words).most_common(10)
            
            if ham_word_counts:
                fig, ax = plt.subplots(figsize=(8, 6))
                words, counts = zip(*ham_word_counts)
                ax.barh(words, counts)
                ax.set_title('Top 10 Words in Non-Spam Emails')
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.write("No words available for visualization")
        
        with col2:
            st.write("Spam Emails:")
            spam_text = " ".join(balanced_data[balanced_data['label'] == 'spam']['text'])
            spam_words = spam_text.split()
            spam_word_counts = Counter(spam_words).most_common(10)
            
            if spam_word_counts:
                fig, ax = plt.subplots(figsize=(8, 6))
                words, counts = zip(*spam_word_counts)
                ax.barh(words, counts, color='red')
                ax.set_title('Top 10 Words in Spam Emails')
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.write("No words available for visualization")
        
        # Store processed data in session state
        st.session_state.processed_data = balanced_data

# Model Training Section
elif app_section == "Model Training":
    st.markdown('<h2 class="section-header">Model Training</h2>', unsafe_allow_html=True)
    
    if 'processed_data' not in st.session_state:
        st.warning("Please go to the Data Preprocessing section first to process the data.")
    else:
        balanced_data = st.session_state.processed_data
        
        # Prepare the data
        train_X, test_X, train_Y, test_Y = train_test_split(
            balanced_data['text'], balanced_data['label'], test_size=0.2, random_state=42
        )
        
        # Tokenization
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(train_X)
        
        train_sequences = tokenizer.texts_to_sequences(train_X)
        test_sequences = tokenizer.texts_to_sequences(test_X)
        
        max_len = 20  # Reduced for our small dataset
        train_sequences = pad_sequences(train_sequences, maxlen=max_len, padding='post', truncating='post')
        test_sequences = pad_sequences(test_sequences, maxlen=max_len, padding='post', truncating='post')
        
        train_Y = (train_Y == 'spam').astype(int)
        test_Y = (test_Y == 'spam').astype(int)
        
        # Store tokenizer and sequences for prediction
        st.session_state.tokenizer = tokenizer
        st.session_state.train_sequences = train_sequences
        st.session_state.test_sequences = test_sequences
        st.session_state.train_Y = train_Y
        st.session_state.test_Y = test_Y
        
        # Model architecture
        st.write("**Model Architecture:**")
        model = tf.keras.models.Sequential([
            tf.keras.layers.Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=16, input_length=max_len),
            tf.keras.layers.LSTM(8),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            loss=tf.keras.losses.BinaryCrossentropy(),
            optimizer='adam',
            metrics=['accuracy']
        )
        
        # Display model summary
        st.text("Model Summary:")
        summary_list = []
        model.summary(print_fn=lambda x: summary_list.append(x))
        st.text("\n".join(summary_list))
        
        # Train the model
        st.write("**Training the Model:**")
        epochs = st.slider("Number of epochs", min_value=5, max_value=50, value=10)
        
        if st.button("Train Model"):
            with st.spinner("Training in progress..."):
                # Since we have a small sample dataset, we'll use simple training
                # without callbacks to avoid import issues
                history = model.fit(
                    train_sequences, train_Y,
                    validation_data=(test_sequences, test_Y),
                    epochs=epochs,
                    batch_size=4,
                    verbose=1
                )
                
                # Store model and history
                st.session_state.model = model
                st.session_state.history = history
                
                # Evaluate the model
                test_loss, test_accuracy = model.evaluate(test_sequences, test_Y, verbose=0)
                st.session_state.test_loss = test_loss
                st.session_state.test_accuracy = test_accuracy
                
                st.success("Training completed!")
        
        # Display results if model is trained
        if 'model' in st.session_state:
            st.write("**Training Results:**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Test Loss", f"{st.session_state.test_loss:.4f}")
            
            with col2:
                st.metric("Test Accuracy", f"{st.session_state.test_accuracy:.4f}")
            
            # Plot accuracy
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(st.session_state.history.history['accuracy'], label='Training Accuracy')
            ax.plot(st.session_state.history.history['val_accuracy'], label='Validation Accuracy')
            ax.set_title('Model Accuracy')
            ax.set_ylabel('Accuracy')
            ax.set_xlabel('Epoch')
            ax.legend()
            st.pyplot(fig)

# Make Prediction Section
elif app_section == "Make Prediction":
    st.markdown('<h2 class="section-header">Make a Prediction</h2>', unsafe_allow_html=True)
    
    if 'model' not in st.session_state:
        st.warning("Please train the model first in the Model Training section.")
    else:
        # Input text
        input_text = st.text_area("Enter email text to classify:", height=200, 
                                 value="Congratulations! You've won a free gift. Claim your prize now by calling this number.")
        
        if st.button("Classify Email"):
            if not input_text:
                st.error("Please enter some text to classify.")
            else:
                # Preprocess the input text
                input_text_processed = input_text.replace('Subject', '')
                
                # Remove punctuation
                punctuations_list = string.punctuation
                def remove_punctuations(text):
                    temp = str.maketrans('', '', punctuations_list)
                    return text.translate(temp)
                
                input_text_processed = remove_punctuations(input_text_processed)
                
                # Remove stopwords
                def remove_stopwords(text):
                    imp_words = []
                    
                    for word in str(text).split():
                        word = word.lower()
                        if word not in stopwords_list:
                            imp_words.append(word)
                    
                    return " ".join(imp_words)
                
                input_text_processed = remove_stopwords(input_text_processed)
                
                # Display processed text
                with st.expander("View processed text"):
                    st.write(input_text_processed)
                
                # Tokenize and pad the sequence
                tokenizer = st.session_state.tokenizer
                sequence = tokenizer.texts_to_sequences([input_text_processed])
                padded_sequence = pad_sequences(sequence, maxlen=20, padding='post', truncating='post')
                
                # Make prediction
                model = st.session_state.model
                prediction = model.predict(padded_sequence)
                
                # Display result
                if prediction[0][0] > 0.5:
                    st.error(f"This email is classified as **SPAM** with {prediction[0][0]*100:.2f}% confidence.")
                else:
                    st.success(f"This email is classified as **HAM** (not spam) with {(1-prediction[0][0])*100:.2f}% confidence.")
                
                # Show confidence
                st.write("**Confidence:**")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Spam Confidence", f"{prediction[0][0]*100:.2f}%")
                
                with col2:
                    st.metric("Ham Confidence", f"{(1-prediction[0][0])*100:.2f}%")

# Footer
st.markdown("---")
st.markdown("### About")
st.markdown("""
This spam classification app uses:
- **TensorFlow/Keras** for building and training the neural network
- **NLTK** for text preprocessing
- **Streamlit** for the web interface

The model architecture includes:
- Embedding layer
- LSTM layer
- Dense layers with ReLU and Sigmoid activations

""")
