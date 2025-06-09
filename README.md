# Emotion Classification from Text

This project classifies text into four emotions: **Happy**, **Sad**, **Angry**, and **Neutral**, using a machine learning pipeline.

## 📁 Dataset
The dataset contains labeled text samples for emotion classification. Each entry has a sentence and its corresponding emotion label.  
(*e.g., from Kaggle or a custom CSV*)

## 🔍 Approach Summary
- **Preprocessing**: Cleaned text and converted to numerical features using TF-IDF.
- **Modeling**: Trained a **Logistic Regression** classifier.
- **Evaluation**: Measured performance with **accuracy**, **confusion matrix**, and **classification report**.
- **Interface**: Built an interactive **Streamlit app** for real-time prediction.

## 📦 Dependencies
- Python 3.x  
- pandas  
- scikit-learn  
- matplotlib, seaborn  
- streamlit  # Emotion_classifier_model
