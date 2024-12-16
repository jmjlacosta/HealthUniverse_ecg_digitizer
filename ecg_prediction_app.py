import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import io
from digitization.StreamingDigitizer import Digitizer
from utils.ecg.Lead import Lead
from tensorflow.keras.models import load_model

# Load the pre-trained model
model = load_model('model.hdf5')

# ECG labels
labels = [
    "1st degree AV block (1dAVb)",
    "Right bundle branch block (RBBB)",
    "Left bundle branch block (LBBB)",
    "Sinus bradycardia (SB)",
    "Atrial fibrillation (AF)",
    "Sinus tachycardia (ST)"
]

# Description
st.write("""
# EKG Digitizer and Interpreter

This tool digitizes EKG images, extracts signals, and provides predictions for potential diagnoses.  
It builds on established methods for automated EKG interpretation. For more information, refer to:  
- **[Automated detection of cardiac arrhythmias using deep neural networks](https://www.sciencedirect.com/science/article/pii/S016926072400049X)**, published in *Computers in Biology and Medicine*.  
- **[Artificial intelligence-enhanced electrocardiography](https://www.nature.com/articles/s41467-020-15432-4)**, published in *Nature Communications*.
""")

# User inputs
rhythm = st.selectbox("Select Rhythm Lead", options=["Lead I", "Lead II", "Lead III"], index=1)
rp_at_right = st.checkbox("RP at Right?", value=False)
cabrera = st.checkbox("Use Cabrera Format?", value=False)
force_second_contour = st.checkbox("Force ECG Contour Re-Try", value=False)

# Allow user to define scaling factor
scaling_factor = st.number_input(
    "Scaling Factor for Signals", min_value=-1000, max_value=1000, value=10,
    step=1, help="Adjust the scaling factor for the ECG signals. The model expects the values to be on a scale of 1e-4V. Default is 10."
)

# Map rhythm selection to Lead enum
rhythm_map = {"Lead I": Lead.I, "Lead II": Lead.II, "Lead III": Lead.III}
selected_rhythm = rhythm_map[rhythm]

# Initialize the Digitizer with dynamic options
digitizer = Digitizer(
    layout=(3, 4),
    rhythm=[selected_rhythm],
    rp_at_right=rp_at_right,
    cabrera=cabrera,
    outpath="",
    ocr=False,
    interpolation=16384
)

def digitize_image(image_path):
    """
    Wrapper around the Digitizer to process the uploaded image.

    Args:
        image_path: Path to the uploaded image.

    Returns:
        digitized_image: The processed image (with trace overlay).
        data: Extracted signal data as a pandas DataFrame.
    """
    ecg, data = digitizer.digitize(image_path, return_values=True, force_second_contour=force_second_contour)

    # Preprocess data
    data_I = data['I'].iloc[:4096].reset_index(drop=True)
    data_II = data['II'].rolling(window=4).mean().iloc[3::4].reset_index(drop=True)
    data_III = data['III'].iloc[:4096].reset_index(drop=True)
    data_aVR_aVF = data[['aVR', 'aVL', 'aVF']].iloc[4096:8192].reset_index(drop=True)
    data_V1_V3 = data[['V1', 'V2', 'V3']].iloc[8192:12288].reset_index(drop=True)
    data_V4_V6 = data[['V4', 'V5', 'V6']].iloc[12288:16384].reset_index(drop=True)

    # Concatenate slices and scale signals
    data_final = pd.concat([data_I, data_II, data_III, data_aVR_aVF, data_V1_V3, data_V4_V6], axis=1)
    data_final *= scaling_factor  # Apply user-defined scaling factor
    
    return ecg, data_final

def get_predictions(data):
    """
    Get predictions from the model based on the extracted ECG data.

    Args:
        data: The extracted ECG signal data as a pandas DataFrame.

    Returns:
        predictions: List of predicted labels.
    """
    model_input = np.expand_dims(data.to_numpy(), axis=0)
    model_input = np.nan_to_num(model_input)
    predictions = model.predict(model_input)
    return [(labels[i], predictions[0][i]) for i in range(len(labels))]

def main():
    st.write("Upload an EKG image to digitize it, extract data, and view predictions for potential diagnoses.")

    # File uploader
    uploaded_file = st.file_uploader("Choose an EKG image file", type=["png", "jpg", "jpeg", "tif", "tiff"])

    if uploaded_file is not None:
        # Save the uploaded image to a temporary path
        temp_path = "temp_uploaded_image.png"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Display the uploaded image
        st.image(uploaded_file, caption="Uploaded EKG Image", use_column_width=True)

        # Process the image
        st.write("Processing image...")
        digitized_image, data = digitize_image(temp_path)

        digitized_image_data = digitized_image.data
        if digitized_image_data.dtype != np.uint8:
            digitized_image_data = (255 * (digitized_image_data / np.max(digitized_image_data))).astype(np.uint8)

        # Convert array to PIL image
        pil_image = Image.fromarray(digitized_image_data, mode="L" if len(digitized_image_data.shape) == 2 else "RGB")
        st.image(pil_image, caption="Digitized Image", use_column_width=True)

        # Display extracted data
        st.write("Extracted Data:")
        st.dataframe(data)

        # Create CSV download link (Move above predictions)
        csv_buffer = io.StringIO()
        data.to_csv(csv_buffer, index=False)
        csv_bytes = csv_buffer.getvalue().encode('utf-8')
        st.download_button(
            label="Download Extracted Data CSV",
            data=csv_bytes,
            file_name="digitized_data.csv",
            mime="text/csv"
        )

        # Display predictions
        predictions = get_predictions(data)
        likely_labels = [(label, prediction) for label, prediction in predictions if prediction > 0.5]
        max_label, max_prediction = max(predictions, key=lambda x: x[1])

        # st.markdown("### Predictions")
        # for label, prediction in predictions:
        #     if prediction > 0.5:
        #         st.write(f"**Likely {label}: {round(100 * prediction, 2)}%**")
            # else:
            #     st.write(f"{label}: {round(100 * prediction, 2)}%")

        # Highlight final prediction
        if likely_labels:
            st.markdown(f"### **Prediction: Likely {likely_labels[0][0]}**")
        else:
            st.markdown(f"### **Prediction: Possible {max_label}**")

if __name__ == "__main__":
    main()
