import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import os

# Configure Streamlit page
st.set_page_config(
    page_title="Emotion Classifier",
    page_icon="😊",
    layout="wide"
)

# Create sample data if no dataset is available
@st.cache_data
def create_sample_data():
    """Create sample emotion data for demonstration"""
    sample_texts = [
        "I am so happy today!", "This is wonderful news!", "I love this!",
        "I am really angry about this situation", "This makes me furious", "I hate when this happens",
        "I feel so sad about this", "This is heartbreaking", "I'm feeling down",
        "I'm scared of what might happen", "This is terrifying", "I'm worried about the outcome",
        "What a surprise!", "I didn't expect this at all", "This is shocking",
        "I don't really care about this", "This is okay I guess", "Nothing special here"
    ]
    
    sample_labels = [
        "joy", "joy", "joy",
        "anger", "anger", "anger", 
        "sadness", "sadness", "sadness",
        "fear", "fear", "fear",
        "surprise", "surprise", "surprise",
        "neutral", "neutral", "neutral"
    ]
    
    return pd.DataFrame({'text': sample_texts, 'label': sample_labels})

@st.cache_data
def load_data():
    """Load emotion dataset or create sample data"""
    try:
        # Try to load the actual dataset
        if os.path.exists("emotion_dataset.csv"):
            df = pd.read_csv("emotion_dataset.csv")
            st.success("✅ Loaded emotion_dataset.csv")
        else:
            df = create_sample_data()
            st.info("📝 Using sample data (emotion_dataset.csv not found)")
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return create_sample_data()

@st.cache_resource
def train_model(df):
    """Train the emotion classification model"""
    try:
        # Prepare labels - ensure we have consistent mapping
        unique_labels = sorted(df['label'].unique())
        label_to_code = {label: i for i, label in enumerate(unique_labels)}
        code_to_label = {i: label for i, label in enumerate(unique_labels)}
        
        # Map labels to codes
        df['label_code'] = df['label'].map(label_to_code)
        
        # Prepare features
        X = df['text']
        y = df['label_code']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Vectorize text
        vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)
        
        # Train model
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train_vec, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test_vec)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Get unique labels for classification report
        unique_test_labels = sorted(np.unique(np.concatenate((y_test, y_pred))))
        target_names = [code_to_label[i] for i in unique_test_labels]
        
        # Generate classification report
        report = classification_report(
            y_test, y_pred, 
            target_names=target_names,
            zero_division=0,
            output_dict=True
        )
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred, labels=unique_test_labels)
        
        return {
            'model': model,
            'vectorizer': vectorizer,
            'label_map': code_to_label,  # This should now be properly formatted
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm,
            'target_names': target_names,
            'test_data': (X_test, y_test, y_pred)
        }
        
    except Exception as e:
        st.error(f"Error training model: {e}")
        return None

def plot_confusion_matrix(cm, target_names):
    """Plot confusion matrix"""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', 
                xticklabels=target_names, 
                yticklabels=target_names,
                cmap='Blues', ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    return fig

def main():
    st.title("🎭 Emotion Classification App")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", 
                               ["Emotion Classifier", "Model Performance", "Dataset Info"])
    
    # Load data and train model
    with st.spinner("Loading data and training model..."):
        df = load_data()
        model_data = train_model(df)
    
    if model_data is None:
        st.error("Failed to train model. Please check your data.")
        return
    
    if page == "Emotion Classifier":
        st.header("🔮 Predict Emotion")
        
        # Display emotion mapping
        st.subheader("📊 Emotion Code Mapping")
        mapping_df = pd.DataFrame(list(model_data['label_map'].items()), 
                                 columns=['Code', 'Emotion'])
        st.dataframe(mapping_df, use_container_width=True)
        
        # Text input
        user_input = st.text_area(
            "Enter text to classify emotion:",
            placeholder="Type your text here...",
            height=100
        )
        
        if st.button("Classify Emotion", type="primary"):
            if user_input.strip():
                # Make prediction
                user_vec = model_data['vectorizer'].transform([user_input])
                prediction_code = model_data['model'].predict(user_vec)[0]
                probabilities = model_data['model'].predict_proba(user_vec)[0]
                
                # Debug: Check what we're getting
                st.write(f"Debug - prediction_code: {prediction_code}, type: {type(prediction_code)}")
                st.write(f"Debug - label_map: {model_data['label_map']}")
                
                # Convert prediction code to emotion string
                predicted_emotion = model_data['label_map'].get(prediction_code, f"unknown_{prediction_code}")
                
                # Ensure predicted_emotion is a string
                if not isinstance(predicted_emotion, str):
                    predicted_emotion = str(predicted_emotion)
                
                confidence = probabilities[prediction_code]
                
                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.success(f"**Predicted Emotion:** {predicted_emotion.upper()}")
                    st.info(f"**Confidence:** {confidence:.2%}")
                
                with col2:
                    # Create probability chart
                    prob_df = pd.DataFrame({
                        'Emotion': [model_data['label_map'][i] for i in range(len(probabilities))],
                        'Probability': probabilities
                    }).sort_values('Probability', ascending=True)
                    
                    fig, ax = plt.subplots(figsize=(8, 4))
                    bars = ax.barh(prob_df['Emotion'], prob_df['Probability'])
                    ax.set_xlabel('Probability')
                    ax.set_title('Emotion Probabilities')
                    
                    # Color the predicted emotion bar differently
                    for i, bar in enumerate(bars):
                        if prob_df.iloc[i]['Emotion'] == predicted_emotion:
                            bar.set_color('red')
                        else:
                            bar.set_color('lightblue')
                    
                    st.pyplot(fig)
                    plt.close()
                
                # Show detailed probabilities
                st.markdown("---")
                st.subheader("All Emotion Probabilities")
                detailed_df = pd.DataFrame({
                    'Emotion': [model_data['label_map'][i] for i in range(len(probabilities))],
                    'Probability': [f"{p:.4f}" for p in probabilities],
                    'Percentage': [f"{p:.2%}" for p in probabilities]
                }).sort_values('Probability', ascending=False, key=lambda x: x.astype(float))
                
                st.dataframe(detailed_df, use_container_width=True)
            else:
                st.warning("Please enter some text to classify.")
    
    elif page == "Model Performance":
        st.header("📊 Model Performance")
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Accuracy", f"{model_data['accuracy']:.3f}")
        
        with col2:
            avg_precision = model_data['classification_report']['macro avg']['precision']
            st.metric("Avg Precision", f"{avg_precision:.3f}")
        
        with col3:
            avg_recall = model_data['classification_report']['macro avg']['recall']
            st.metric("Avg Recall", f"{avg_recall:.3f}")
        
        # Confusion Matrix
        st.subheader("Confusion Matrix")
        fig = plot_confusion_matrix(model_data['confusion_matrix'], model_data['target_names'])
        st.pyplot(fig)
        plt.close()
        
        # Classification Report
        st.subheader("Detailed Classification Report")
        report_df = pd.DataFrame(model_data['classification_report']).transpose()
        st.dataframe(report_df.round(3), use_container_width=True)
    
    elif page == "Dataset Info":
        st.header("📋 Dataset Information")
        
        # Dataset overview
        st.subheader("Dataset Overview")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Samples", len(df))
            st.metric("Unique Emotions", df['label'].nunique())
        
        with col2:
            st.metric("Average Text Length", f"{df['text'].str.len().mean():.1f}")
            st.metric("Text Columns", len([col for col in df.columns if df[col].dtype == 'object']))
        
        # Emotion distribution
        st.subheader("Emotion Distribution")
        emotion_counts = df['label'].value_counts()
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(8, 6))
            emotion_counts.plot(kind='bar', ax=ax, color='skyblue')
            ax.set_title('Distribution of Emotions')
            ax.set_xlabel('Emotion')
            ax.set_ylabel('Count')
            plt.xticks(rotation=45)
            st.pyplot(fig)
            plt.close()
        
        with col2:
            st.dataframe(emotion_counts.reset_index().rename(columns={'index': 'Emotion', 'label': 'Count'}))
        
        # Sample data
        st.subheader("Sample Data")
        st.dataframe(df.head(10), use_container_width=True)

if __name__ == "__main__":
    main()