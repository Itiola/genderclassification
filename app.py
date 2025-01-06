import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import pandas as pd
import tensorflow as tf

# Check for GPU Support
def check_gpu():
    if tf.config.list_physical_devices('GPU'):
        return "✅ GPU is available and being used for inference."
    else:
        return "❌ No GPU detected. Using CPU for inference."

# Load the Trained Model
model = load_model("gender_classifier_efficientnetb0.keras")  # Replace with your model file name

# Define the Class Names
class_names = ["Female", "Male"]

# Custom CSS for Light Mode Styling
st.markdown("""
    <style>
    body {
        background-color: #f9f9f9;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: #333;
    }
    .main {
        background-color: #ffffff;
        padding: 30px;
        border-radius: 12px;
        box-shadow: 0px 8px 12px rgba(0, 0, 0, 0.1);
        margin-top: 20px;
    }
    h1 {
        color: #4CAF50;
        font-size: 2.5em;
    }
    h2 {
        color: #333;
        font-size: 1.6em;
    }
    h3 {
        color: #555;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 12px 24px;
        cursor: pointer;
        font-size: 1em;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stProgress>div>div {
        background-color: #4CAF50;
    }
    .stImage>img {
        border-radius: 12px;
        box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
    }
    .stProgress {
        margin-top: 20px;
    }
    footer {
        text-align: center;
        padding: 20px;
        color: #888;
        font-size: 0.9em;
    }
    .card {
        background-color: #f7f7f7;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .card-header {
        color: #4CAF50;
        font-size: 1.5em;
        margin-bottom: 15px;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.title("Enhanced Gender Classification with Analytics")
st.subheader("Upload one or more images to classify them as Male or Female.")
st.markdown("This app uses a pre-trained EfficientNetB0 model for classification.")
st.markdown(f"**GPU Status:** {check_gpu()}")

# Drag-and-Drop File Uploader for Batch Upload
uploaded_files = st.file_uploader("Choose images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    st.write("### Uploaded Images:")
    for uploaded_file in uploaded_files:
        st.image(uploaded_file, caption=uploaded_file.name, use_column_width=True)

    # Initialize Results List
    results = []
    male_count, female_count = 0, 0
    
    # Progress Bar with Animation
    progress_bar = st.progress(0)
    total_files = len(uploaded_files)

    st.write("### Predictions:")
    for idx, uploaded_file in enumerate(uploaded_files):
        # Process each image
        img = Image.open(uploaded_file)
        img = img.resize((224, 224))  # Resize to model input size

        # Preprocess Image
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array = img_array / 255.0  # Normalize pixel values

        # Perform Prediction
        predictions = model.predict(img_array)
        confidence = predictions[0][0]

        # Determine Class
        if confidence > 0.5:
            result = class_names[1]  # Male
            confidence_score = confidence * 100
            male_count += 1
        else:
            result = class_names[0]  # Female
            confidence_score = (1 - confidence) * 100
            female_count += 1

        # Store Results
        results.append({
            "Image": uploaded_file.name,
            "Prediction": result,
            "Confidence (%)": round(confidence_score, 2)
        })

        # Update Progress Bar
        progress_bar.progress((idx + 1) / total_files)

        # Display Prediction in Cards
        with st.container():
            st.markdown(f"<div class='card'><div class='card-header'>{uploaded_file.name}</div><div>**Prediction:** {result}</div><div>**Confidence:** {confidence_score:.2f}%</div></div>", unsafe_allow_html=True)

    # Convert Results to DataFrame
    results_df = pd.DataFrame(results)

    # Interactive Bar Chart with Tooltips
    st.write("### Prediction Distribution")
    chart_data = pd.DataFrame({
        "Class": ["Male", "Female"],
        "Count": [male_count, female_count]
    })
    st.bar_chart(chart_data.set_index("Class"))

    # Allow Download of CSV Report
    st.write("### Download Predictions as CSV:")
    csv = results_df.to_csv(index=False)
    st.download_button(
        label="Download Predictions as CSV",
        data=csv,
        file_name="gender_classification_predictions.csv",
        mime="text/csv"
    )

# Footer with Credits
st.markdown("""
    <footer>
        Developed with ❤️ using Streamlit and TensorFlow. Powered by Modern UI Design.
    </footer>
""", unsafe_allow_html=True)
