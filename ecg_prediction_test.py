from fastapi import FastAPI, Request, Form, HTTPException, UploadFile, File
from digitization.StreamingDigitizer import Digitizer
from fastapi.middleware.cors import CORSMiddleware
from typing import Annotated, Literal, Optional
from tensorflow.keras.models import load_model
from pydantic import BaseModel, Field, HttpUrl
from fastapi.responses import FileResponse
from utils.ecg.Lead import Lead
from PIL import Image
import pandas as pd
import numpy as np

app = FastAPI(
    title="EKG Digitizer and Interpreter",
    description="""This tool digitizes EKG images, extracts signals, and provides predictions for potential diagnoses.  
It builds on established methods for automated EKG interpretation. For more information, refer to:  
- **[Automated detection of cardiac arrhythmias using deep neural networks](https://www.sciencedirect.com/science/article/pii/S016926072400049X)**, published in *Computers in Biology and Medicine*.  
- **[Artificial intelligence-enhanced electrocardiography](https://www.nature.com/articles/s41467-020-15432-4)**, published in *Nature Communications*.""",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class EKGFormInput(BaseModel):
    """Form-based input schema for appropriately reading an EKG image."""

    rhythm: Literal["Lead I", "Lead II", "Lead III"] = Field(
        default="Lead II",
        title="Rhythm Lead",
        examples=["Lead II"],
        description="Rhythm leads monitor the heart's electrical activity over time, with Lead II being the most common in an EKG, offering a clear view of the heart's electrical activity along its natural conduction pathway.",
    )

    reference_pulse: Literal["Left", "Right"] = Field(
        default="Left",
        title="Reference Pulse",
        examples=["Left"],
        description="The reference pulse in an EKG is a calibration marker indicating voltage and time standards, usually seen on the left side at the start of the recording.",
    )

    ekg_format: Literal["Standard", "Cabrera"] = Field(
        default="Standard",
        title="Format",
        examples=["Standard"],
        description="The EKG format refers to the arrangement of the 12 leads used to record the heart's electrical activity, with the standard format placing limb leads (I, II, III, aVR, aVL, aVF) and precordial leads (V1–V6) in a specific order. The Cabrera format rearranges the limb leads into a sequential anatomical order (aVL, I, -aVR, II, aVF, III) to emphasize the heart's electrical axis and facilitate the identification of conduction abnormalities.",
    )

    force_second_contour: bool = Field(
        default=False,
        title="Force to Re-Contour",
        examples=[False],
        description="Sometimes the EKG detenction doesn't work. This allows for a quick way to re-try.",
    )

    scaling_factor: float = Field(
        default=10,
        title="Scaling Factor",
        examples=[10],
        description="The prediction model expects the EKG values to be on a scale of 100 µV per mm. For a standard 10 mm/mV EKG, we need to divide the values by 10 to match the expected scale.",
        gt=0,
    )

    image_upload: Literal["Example Image I", "Example Image II"] = Field(
        default="Example Image I",
        title="Example Image",
        examples=["Example Image I"],
        description="Example EKG to use for Demo.",
    )

class EKGFormOutput(BaseModel):
    """Form-based output schema for result from reading an EKG image."""
    prediction: str = Field(
        title="Predictions",
        examples=["Prediction: Likely 1st degree AV block (1dAVb)"],
        description="The prediction of the possible conditions.",
        format="display",
    )

    download_link: HttpUrl = Field(
        title="Download Link",
        examples=["http://127.0.0.1:8000/download_processed_image"],
        description="A link to download the processed EKG image.",
    )


@app.post(
    "/process_ekg_image/",
    response_model=EKGFormOutput,
    summary="Process EKG Image",
    description="Digitize EKG image, extract signals, and provide predictions for potential diagnoses.",
)

def process_ekg_image(
    data: Annotated[EKGFormInput, Form()],
    request: Request,
) -> EKGFormOutput:
    """Digitize EKG image, extract signals, and provide predictions for potential diagnoses.

    Args:
        data: EKGFormInput - input data containing EKG format information.

    Returns:
        EKGFormOutput: prediction on digitized EKG and interpretation
    """
    image = None

    if data.image_upload == "Example Image I":
        file_location = "data/example_1.png"
    else:
        file_location = "data/example_2.png"
        
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

    # Map rhythm selection to Lead enum
    rhythm_map = {"Lead I": Lead.I, "Lead II": Lead.II, "Lead III": Lead.III}
    selected_rhythm = rhythm_map[data.rhythm]
    rp_at_right = data.reference_pulse == "Right"
    cabrera = data.ekg_format == "Cabrera"
    scaling_factor = data.scaling_factor

    # Initialize the Digitizer with dynamic options
    digitizer = Digitizer(
        layout=(3, 4),
        rhythm=[selected_rhythm],
        rp_at_right=rp_at_right,
        cabrera=cabrera,
        outpath="data/",
        ocr=False,
        interpolation=16384
    )

    ecg, data = digitizer.digitize(file_location, return_values=True, force_second_contour=data.force_second_contour)

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

    model_input = np.expand_dims(data_final.to_numpy(), axis=0)
    model_input = np.nan_to_num(model_input)
    preds = model.predict(model_input)

    output_image_path = "data/processed_ecg.png"
    ecg.save(output_image_path)
    # if ecg.dtype != np.uint8:
    #     ecg = (255 * (ecg / np.max(ecg))).astype(np.uint8)
    # pil_image = Image.fromarray(ecg, mode="L" if len(ecg.shape) == 2 else "RGB")
    # pil_image.save(output_image_path)

    # Display predictions
    predictions = [(labels[i], preds[0][i]) for i in range(len(labels))]
    likely_labels = [(label, prediction) for label, prediction in predictions if prediction > 0.5]
    max_label, max_prediction = max(predictions, key=lambda x: x[1])

    # Highlight final prediction
    if likely_labels:
        prediction = f"### **Prediction: Likely {likely_labels[0][0]}**"
    else:
        prediction = f"### **Prediction: Possible {max_label}**"

    base_url = request.base_url
    return EKGFormOutput(
        prediction=prediction,
        download_link=str(base_url) + "download_processed_image",
    )

@app.get("/download_processed_image", summary="Download Processed ECG Image")
async def download_processed_image():
    """Serve the processed ECG image for download."""
    return FileResponse("data/processed_ecg.png", media_type="image/png", filename="processed_ecg.png")