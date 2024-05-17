# Python In-built packages
import os
import tempfile
from pathlib import Path
import PIL

# External packages
import streamlit as st

# Local Modules
import settings
import helper

# Setting page layout
st.set_page_config(
    page_title="Traffic Sign and Drowsiness Detection  ",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main page heading
st.title("Traffic Sign and Drowsiness Detection ")

# Sidebar
st.sidebar.header("ML Model Config")


# Model Options
model_type = st.sidebar.radio(
    "Select Task", ['Traffic Sign', 'Drowsiness Detection',])
print("model type ", model_type)
# confidence = float(st.sidebar.slider(
#     "Select Model Confidence", 25, 100, 40)) / 100

# Selecting Detection Or Segmentation
if model_type == 'Traffic Sign':
    model_path = Path(settings.TRAFFIC_MODEL)
elif model_type == 'Drowsiness Detection':
    model_path = Path(settings.DROWSINESS_MODEL)

# Load Pre-trained ML Model
try:
    print("model path", model_path)
    model = helper.load_model(model_path)
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)

st.sidebar.header("Image/Video Config")
source_radio = st.sidebar.radio(
    "Select Source", settings.SOURCES_LIST)

source_img = None
# If image is selected
if source_radio == settings.IMAGE:
    source_img = st.sidebar.file_uploader(
        "Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

    col1, col2 = st.columns(2)

    with col1:
        try:
            if source_img is None:
                default_image_path = str(settings.DEFAULT_IMAGE)
                default_image = PIL.Image.open(default_image_path)
                st.image(default_image_path, caption="Default Image",
                         use_column_width=True)
            else:
                uploaded_image = PIL.Image.open(source_img)
                st.image(source_img, caption="Uploaded Image",
                         use_column_width=True)
        except Exception as ex:
            st.error("Error occurred while opening the image.")
            st.error(ex)

    with col2:
        if source_img is None:
            default_detected_image_path = str(settings.DEFAULT_DETECT_IMAGE)
            default_detected_image = PIL.Image.open(
                default_detected_image_path)
            st.image(default_detected_image_path, caption='Detected Image',
                     use_column_width=True)
        else:
            if st.sidebar.button('Detect Objects'):
                res = model.predict(uploaded_image,

                                    )
                boxes = res[0].boxes
                res_plotted = res[0].plot()[:, :, ::-1]
                st.image(res_plotted, caption='Detected Image',
                         use_column_width=True)
                try:
                    with st.expander("Detection Results"):
                        for box in boxes:
                            st.write(box.data)
                except Exception as ex:
                    # st.write(ex)
                    st.write("No image is uploaded yet!")

elif source_radio == settings.VIDEO:
    video = st.file_uploader("Choose a video", type=["mp4","avi"])
    if video is not None:
        # Temporary file to save the uploaded video
        temp_video = tempfile.NamedTemporaryFile(delete=False)
        temp_video.write(video.read())

        temp_video.close()

        os.rename(temp_video.name,temp_video.name+".mp4")

        helper.play_stored_video(temp_video.name,model)

elif source_radio == settings.WEBCAM:
    helper.play_webcam(model_path, model)

elif source_radio == settings.RTSP:
    helper.play_rtsp_stream( model)

elif source_radio == settings.YOUTUBE:
    helper.play_youtube_video(model)

else:
    st.error("Please select a valid source type!")
