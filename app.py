import sqlite3
from pathlib import Path
from datetime import datetime
from huggingface_hub import hf_hub_download
import streamlit as st
import torch
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms

from model import CNNModel


# =========================================================
# 1. APP CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Tomato Disease Detection",
    page_icon="🍅",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = hf_hub_download(
    repo_id="Deepanshu765/tomato-disease-cnn",
    filename="tomato_disease_cnn.pth"
)
DB_NAME = str(BASE_DIR / "prediction_history.db")

device = torch.device("cpu")


# =========================================================
# 2. DISEASE CLASS NAMES
# IMPORTANT: Must match ImageFolder class_to_idx order
# =========================================================

CLASS_NAMES = [
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato__Tomato_mosaic_virus",
    "Tomato_healthy"
]


# =========================================================
# 3. IMAGE TRANSFORMATION
# Must match the transformations used during training
# =========================================================

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =========================================================
# 4. DISEASE INFORMATION
# =========================================================

DISEASE_INFO = {
    "Tomato_Bacterial_spot": {
        "name": "Bacterial Spot",
        "description": (
            "Bacterial spot can cause small, dark lesions on tomato "
            "leaves and may affect fruits."
        ),
        "prevention": (
            "Use disease-free seeds, avoid working with wet plants, "
            "and follow locally recommended disease-management practices."
        )
    },

    "Tomato_Early_blight": {
        "name": "Early Blight",
        "description": (
            "Early blight commonly produces brown spots with "
            "concentric rings on older leaves."
        ),
        "prevention": (
            "Remove infected plant debris, improve airflow, avoid "
            "overhead watering, and follow local agricultural guidance."
        )
    },

    "Tomato_Late_blight": {
        "name": "Late Blight",
        "description": (
            "Late blight can cause rapidly expanding, water-soaked "
            "or brown lesions on leaves and stems."
        ),
        "prevention": (
            "Monitor plants frequently, avoid prolonged leaf wetness, "
            "remove affected material safely, and seek local agricultural advice."
        )
    },

    "Tomato_Leaf_Mold": {
        "name": "Leaf Mold",
        "description": (
            "Leaf mold may cause pale or yellow areas on the upper "
            "leaf surface and mold growth underneath."
        ),
        "prevention": (
            "Improve ventilation, reduce humidity around plants, "
            "and avoid keeping foliage wet for long periods."
        )
    },

    "Tomato_Septoria_leaf_spot": {
        "name": "Septoria Leaf Spot",
        "description": (
            "Septoria leaf spot often appears as small circular spots "
            "with pale centers and dark margins."
        ),
        "prevention": (
            "Remove infected leaves when appropriate, manage plant debris, "
            "and avoid splashing water onto foliage."
        )
    },

    "Tomato_Spider_mites_Two_spotted_spider_mite": {
        "name": "Two-Spotted Spider Mites",
        "description": (
            "Spider mites can cause tiny pale speckles, leaf discoloration, "
            "and sometimes fine webbing."
        ),
        "prevention": (
            "Inspect the undersides of leaves and use locally recommended "
            "integrated pest-management methods."
        )
    },

    "Tomato__Target_Spot": {
        "name": "Target Spot",
        "description": (
            "Target spot can produce brown lesions with concentric "
            "rings on leaves and other plant parts."
        ),
        "prevention": (
            "Improve airflow, avoid excessive leaf wetness, and remove "
            "infected plant debris where appropriate."
        )
    },

    "Tomato__Tomato_YellowLeaf__Curl_Virus": {
        "name": "Tomato Yellow Leaf Curl Virus",
        "description": (
            "This viral disease can cause yellowing, upward leaf curling, "
            "stunted growth, and reduced fruit production."
        ),
        "prevention": (
            "Use healthy seedlings, monitor whiteflies, and follow "
            "local recommendations for managing virus transmission."
        )
    },

    "Tomato__Tomato_mosaic_virus": {
        "name": "Tomato Mosaic Virus",
        "description": (
            "Tomato mosaic virus may cause mottled leaf patterns, "
            "distorted growth, and reduced plant vigor."
        ),
        "prevention": (
            "Use clean tools, maintain good hygiene when handling plants, "
            "and remove infected plants according to local guidance."
        )
    },

    "Tomato_healthy": {
        "name": "Healthy Tomato Leaf",
        "description": (
            "The model classified this image as belonging to the "
            "healthy tomato leaf class."
        ),
        "prevention": (
            "Continue regular plant monitoring, provide suitable growing "
            "conditions, and inspect leaves for changes."
        )
    }
}


# =========================================================
# 5. DATABASE FUNCTIONS
# =========================================================

def create_database():
    """Create the SQLite database and prediction table."""

    with sqlite3.connect(DB_NAME) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                predicted_class TEXT NOT NULL,
                confidence REAL NOT NULL
            )
        """)


def save_prediction(predicted_class, confidence):
    """Save a prediction to the database."""

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with sqlite3.connect(DB_NAME) as conn:
        conn.execute(
            """
            INSERT INTO predictions
            (timestamp, predicted_class, confidence)
            VALUES (?, ?, ?)
            """,
            (timestamp, predicted_class, float(confidence))
        )


def get_prediction_history():
    """Retrieve all saved predictions."""

    with sqlite3.connect(DB_NAME) as conn:
        df = pd.read_sql_query(
            """
            SELECT
                id AS 'ID',
                timestamp AS 'Date & Time',
                predicted_class AS 'Predicted Disease',
                confidence AS 'Confidence (%)'
            FROM predictions
            ORDER BY id DESC
            """,
            conn
        )

    return df


def get_total_predictions():
    """Return the number of saved predictions."""

    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.execute(
            "SELECT COUNT(*) FROM predictions"
        )
        return cursor.fetchone()[0]


def clear_prediction_history():
    """Delete all prediction records."""

    with sqlite3.connect(DB_NAME) as conn:
        conn.execute("DELETE FROM predictions")


create_database()


# =========================================================
# 6. LOAD TRAINED MODEL
# =========================================================

@st.cache_resource
def load_model():
    """Load the trained CNN model once."""

    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH.name}"
        )

    model = CNNModel()

    try:
        state_dict = torch.load(
            MODEL_PATH,
            map_location=device,
            weights_only=True
        )
    except TypeError:
        # Compatibility with older PyTorch versions
        state_dict = torch.load(
            MODEL_PATH,
            map_location=device
        )

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model


# =========================================================
# 7. PREDICTION FUNCTION
# =========================================================

def predict_image(image, model):
    """Predict the disease class for an uploaded image."""

    image = image.convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.inference_mode():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)

        confidence, predicted_index = torch.max(
            probabilities, dim=1
        )

    predicted_index = int(predicted_index.item())
    confidence = float(confidence.item()) * 100

    if predicted_index >= len(CLASS_NAMES):
        raise ValueError(
            "Model output count does not match CLASS_NAMES."
        )

    predicted_class = CLASS_NAMES[predicted_index]

    return predicted_class, confidence


# =========================================================
# 8. CUSTOM CSS
# =========================================================

st.markdown("""
<style>
    .stApp {
        background-color: #f5f8f6;
    }

    [data-testid="stSidebar"] {
        background-color: #102a23;
    }

    [data-testid="stSidebar"] * {
        color: #ffffff;
    }

    .hero {
        padding: 28px;
        border-radius: 18px;
        background: linear-gradient(120deg, #12372a, #26734d);
        color: white;
        margin-bottom: 24px;
    }

    .hero h1 {
        font-size: 34px;
        margin-bottom: 8px;
    }

    .hero p {
        font-size: 16px;
        margin-bottom: 0;
        color: #e2f5e9;
    }

    .info-card {
        padding: 20px;
        border-radius: 14px;
        background: white;
        border: 1px solid #e1e9e3;
        margin-bottom: 16px;
    }

    .info-card h3 {
        color: #17613b;
        margin-top: 0;
    }

    div[data-testid="stMetric"] {
        background: white;
        padding: 18px;
        border-radius: 12px;
        border: 1px solid #e1e9e3;
    }

    .small-note {
        color: #65756b;
        font-size: 13px;
    }

    .footer {
        text-align: center;
        color: #65756b;
        font-size: 13px;
        padding-top: 24px;
        padding-bottom: 10px;
    }

    div.stButton > button {
        border-radius: 9px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# =========================================================
# 9. SIDEBAR NAVIGATION
# =========================================================

with st.sidebar:
    st.markdown("# Tomato AI")
    st.caption("CNN-Based Disease Classification")

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "Dashboard",
            "Disease Detection",
            "Disease Guide",
            "Prediction History"
        ],
        label_visibility="collapsed"
    )

    st.divider()

    st.markdown("### Model Information")
    st.write("Architecture: Custom CNN")
    st.write("Framework: PyTorch")
    st.write("Input Size: 256 × 256")
    st.write("Device: CPU")

    st.divider()

    st.caption("AI-assisted plant leaf classification")


# =========================================================
# 10. LOAD MODEL
# =========================================================

try:
    model = load_model()
    model_error = None

except Exception as error:
    model = None
    model_error = str(error)


# =========================================================
# 11. DASHBOARD PAGE
# =========================================================

if page == "Dashboard":

    st.markdown("""
    <div class="hero">
        <h1>Tomato Disease Detection</h1>
        <p>
            Analyze tomato leaf images using a custom
            Convolutional Neural Network built with PyTorch.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Supported Classes",
            len(CLASS_NAMES)
        )

    with col2:
        st.metric(
            "Total Predictions",
            get_total_predictions()
        )

    with col3:
        st.metric(
            "Model Input",
            "256 × 256"
        )

    st.markdown("## About This Project")

    st.markdown("""
    This application uses a trained Convolutional Neural Network
    to classify tomato leaf images into one of ten categories.

    The application provides:

    - Image upload and disease prediction
    - Model confidence scores
    - Basic disease information and prevention guidance
    - Prediction history stored in SQLite
    - CSV export of prediction records
    """)

    left, right = st.columns(2)

    with left:
        st.markdown("""
        <div class="info-card">
            <h3>How It Works</h3>
            <p>1. Upload a tomato leaf image.</p>
            <p>2. The image is resized and normalized.</p>
            <p>3. The CNN generates class scores.</p>
            <p>4. The application displays the predicted class.</p>
        </div>
        """, unsafe_allow_html=True)

    with right:
        st.markdown("""
        <div class="info-card">
            <h3>Important Note</h3>
            <p>
                Predictions are model estimates, not a professional
                agricultural diagnosis. Results may be unreliable
                for unclear images, other plant species, or unfamiliar
                conditions.
            </p>
        </div>
        """, unsafe_allow_html=True)

    if model_error:
        st.error(
            "The model could not be loaded. Open the Disease Detection "
            "page after fixing the model file or configuration."
        )
        with st.expander("Technical details"):
            st.code(model_error)

    st.markdown("""
    <div class="footer">
        Tomato Disease Detection | PyTorch + Streamlit
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# 12. DISEASE DETECTION PAGE
# =========================================================

elif page == "Disease Detection":

    st.markdown("""
    <div class="hero">
        <h1>Disease Detection</h1>
        <p>Upload a tomato leaf image to analyze it with the CNN model.</p>
    </div>
    """, unsafe_allow_html=True)

    if model_error:
        st.error("Unable to load the trained model.")

        with st.expander("Show technical details"):
            st.code(model_error)

        st.info(
            "Check that model.py and tomato_disease_cnn.pth "
            "are present and compatible."
        )

    else:
        uploaded_file = st.file_uploader(
            "Upload a leaf image",
            type=["jpg", "jpeg", "png"],
            help="Use a clear image showing the tomato leaf."
        )

        if uploaded_file is not None:

            try:
                image = Image.open(uploaded_file).convert("RGB")

                left, right = st.columns([1, 1])

                with left:
                    st.markdown("### Uploaded Image")
                    st.image(
                        image,
                        caption="Image selected for analysis",
                        width="stretch"
                    )

                with right:
                    st.markdown("### Image Details")
                    st.write(f"Width: {image.width} pixels")
                    st.write(f"Height: {image.height} pixels")
                    st.write("Color mode: RGB")

                    st.markdown(
                        '<p class="small-note">'
                        'For best results, use a clear, well-lit '
                        'tomato leaf image.</p>',
                        unsafe_allow_html=True
                    )

                if st.button(
                    "Analyze Leaf",
                    type="primary",
                    use_container_width=True
                ):

                    with st.spinner("Analyzing image..."):

                        predicted_class, confidence = predict_image(
                            image, model
                        )

                        disease_details = DISEASE_INFO.get(
                            predicted_class,
                            {
                                "name": predicted_class.replace("_", " "),
                                "description": (
                                    "No additional disease information "
                                    "is available for this class."
                                ),
                                "prevention": (
                                    "Consult local agricultural guidance."
                                )
                            }
                        )

                        save_prediction(
                            predicted_class,
                            confidence
                        )

                        st.session_state["latest_result"] = {
                            "disease_name": predicted_class,
                            "display_name": disease_details["name"],
                            "confidence": confidence,
                            "description": disease_details["description"],
                            "prevention": disease_details["prevention"]
                        }

                result = st.session_state.get("latest_result")

                if result is not None:

                    st.divider()
                    st.markdown("## Prediction Result")

                    if result["disease_name"] == "Tomato_healthy":
                        st.success(
                            f"Prediction: {result['display_name']}"
                        )
                    else:
                        st.warning(
                            f"Prediction: {result['display_name']}"
                        )

                    result_col1, result_col2 = st.columns(2)

                    with result_col1:
                        st.metric(
                            "Model Confidence",
                            f"{result['confidence']:.2f}%"
                        )

                        st.progress(
                            min(
                                max(result["confidence"] / 100, 0.0),
                                1.0
                            )
                        )

                    with result_col2:
                        st.markdown("### Classification")
                        st.write(result["display_name"])
                        st.caption(
                            "Confidence is the model's softmax score; "
                            "it is not necessarily a calibrated probability."
                        )

                    st.markdown("### About This Class")
                    st.write(result["description"])

                    st.markdown("### Prevention / Care Guidance")
                    st.write(result["prevention"])

                    st.caption(
                        "Confirm important plant-health decisions with "
                        "a qualified agricultural expert."
                    )

            except Exception as error:
                st.error("The image could not be processed.")

                with st.expander("Technical details"):
                    st.code(str(error))


# =========================================================
# 13. DISEASE GUIDE PAGE
# =========================================================

elif page == "Disease Guide":

    st.markdown("""
    <div class="hero">
        <h1>Tomato Disease Guide</h1>
        <p>Learn about the classes supported by the model.</p>
    </div>
    """, unsafe_allow_html=True)

    search_query = st.text_input(
        "Search disease",
        placeholder="For example, Early Blight or Healthy"
    )

    matching_diseases = []

    for class_name, details in DISEASE_INFO.items():

        searchable_text = (
            class_name + " " +
            details["name"] + " " +
            details["description"]
        ).lower()

        if search_query.lower().strip() in searchable_text:
            matching_diseases.append((class_name, details))

    if matching_diseases:

        for class_name, details in matching_diseases:

            with st.expander(details["name"]):

                st.markdown("**Model class**")
                st.code(class_name)

                st.markdown("**Description**")
                st.write(details["description"])

                st.markdown("**Prevention / Care**")
                st.write(details["prevention"])

    else:
        st.info("No matching disease was found.")


# =========================================================
# 14. PREDICTION HISTORY PAGE
# =========================================================

elif page == "Prediction History":

    st.markdown("""
    <div class="hero">
        <h1>Prediction History</h1>
        <p>Review and export predictions saved by the application.</p>
    </div>
    """, unsafe_allow_html=True)

    history_df = get_prediction_history()

    if not history_df.empty:

        st.metric(
            "Total Saved Predictions",
            len(history_df)
        )

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )

        csv_data = history_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="Download History as CSV",
            data=csv_data,
            file_name="tomato_prediction_history.csv",
            mime="text/csv",
            type="primary"
        )

        st.divider()

        st.markdown("### Delete History")

        confirm_delete = st.checkbox(
            "I understand that deleting history is permanent."
        )

        if st.button(
            "Delete All History",
            disabled=not confirm_delete
        ):

            clear_prediction_history()

            st.session_state.pop("latest_result", None)

            st.success("Prediction history deleted.")
            st.rerun()

    else:
        st.info(
            "No predictions found. Open Disease Detection "
            "and analyze an image to create your first record."
        )


# =========================================================
# 15. FOOTER
# =========================================================

st.markdown("""
<div class="footer">
    Built with PyTorch and Streamlit |
    Tomato Disease Detection CNN
</div>
""", unsafe_allow_html=True)