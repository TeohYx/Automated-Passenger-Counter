import torch
import yolov9.detect as detect
import os
import requests
from mainlit import Streamlit
import streamlit as st
from datetime import datetime
import json

from track.track_info import TrackInfo
from util.source_info import SourceInfo
from track.counter import Counter
from database import Database

# Constant
YOLOV9 = "yolov9"
WEIGHTS = "weights"
YOLOV9_WEIGHT = "https://github.com/WongKinYiu/yolov9/releases/download/v0.1"
HOME = os.getcwd()
weight = "gelan-c"
streamlit_session_state_json = "util/streamlit_session_state.json"
preset_line_json = "presetline/preset_line.json"
preset_line_path = "presetline"
screenshot_file = "presetline/screenshot.png"

def download_model_weights(weight):
    """
    Input: model weight (yolov9-s, gelan-c, etc.)
    Output: downloaded model weight in folder
    
    *if the .pt file already downloaded, it will return immediately
    """
    if not WEIGHTS in os.listdir():
        print(f"{WEIGHTS} folder not found, creating one.")
        os.mkdir(WEIGHTS)

    model_weight = f"{weight}.pt"
    weights_url = f"{YOLOV9_WEIGHT}/{model_weight}"
    weights_path = f"{WEIGHTS}/{model_weight}"

    if model_weight in os.listdir(WEIGHTS):
        print(f"{model_weight} already downloaded.")
        return weights_path

    response = requests.get(weights_url, stream=True)

    if response.status_code == 200:
        with open(weights_path, 'wb') as f:
            for chunk in response.iter_content(1024):
                if chunk:
                    f.write(chunk)
        print(f"{model_weight} downloaded successfully to: {weights_path}")    
    else:
        print(f"Download failed with status code: {response.status_code}")

    return weights_path

def initialize_session_state(streamlit_session_state_json):
    with open(streamlit_session_state_json, 'r') as file:
        data = json.load(file)

    for key, value in data.items():
        if key not in st.session_state:
            st.session_state[key] = value

    print("Streamlit session state initialized.")

def main():
    weights_path = download_model_weights(weight)   # Get model weights
    initialize_session_state(streamlit_session_state_json)  # Initialize streamlit session state
    database_name = f"{st.session_state.database} ({datetime.now().strftime('%Y%m%d_%H%M%S')})"
    
    strlit = Streamlit()        # Initiate Streamlit instance
    strlit.json_file = preset_line_json
    mongodb = Database(
        collection_name=database_name, client_host="mongodb+srv://TeohYx:Xian-28-0605@apc.agqhpn3.mongodb.net/"
        )                       # Initiate Database instance

    # mongodb = Database(
    #     collection_name=database_name,
    #     )                       # Initiate Database instance


    # streamlit
    st.title("Automated Passenger Counter")

    # Set up streamlit frontend
    # Have two selection, therefore initializing two big container
    strlit.container.append(st.container())
    strlit.container.append(st.container())

    st.session_state.selectbox_sidebar = st.sidebar.selectbox("Sidebar selection", (strlit.selection_current, strlit.selection_history))

    if st.session_state.selectbox_sidebar == strlit.selection_current:

        media_choser = strlit.container[0].empty()  # to hold the media input tabs

        if not st.session_state.source:
            st.session_state.source = strlit.select_source(media_choser) # return source when chosen

        if st.session_state['is_media_chosen']:
            if st.session_state['streaming_url']:
                media_choser.text(f"Streaming: {st.session_state.source}")
            else:
                media_choser.text(f"Recorded video: {st.session_state.source}")
    elif st.session_state.selectbox_sidebar == strlit.selection_history:
        strlit.display_history_visualization(mongodb)

    # If media source is chosen
    if st.session_state.source:

        counter_obj = Counter()  # Initiate Counter instance
        source_info_obj = SourceInfo(st.session_state.source, screenshot_file)  # Initiate SourceInfo instance
        track_info_obj = TrackInfo(preset_line_json, preset_line_path)  # Initiate TrackInfo instance

        track_info_obj.store_plain_image(screenshot_file)

        # # Selectbox in Streamlit
        if st.session_state.selectbox_sidebar == strlit.selection_current:
            strlit.display_current_interface()
        # elif st.session_state.selectbox_sidebar == strlit.selection_history:
        #     strlit.display_history_interface()

        # Detect screen type
        # dets = {"weights": weights_path, "conf_thres": 0.1 \
        #         , "source": st.session_state.source, "device": torch.cuda.current_device() \
        #         , "classes": 0, "vid_stride": 5, "strlit": strlit \
        #         , "mongodb": mongodb, "counter_obj": counter_obj, "source_info_obj": source_info_obj \
        #         , "track_info_obj": track_info_obj}
        dets = {"weights": weights_path, "conf_thres": 0.1 \
                , "source": st.session_state.source, "device": torch.cuda.current_device() \
                , "classes": 0, "vid_stride": 2, "strlit": strlit \
                , "mongodb": mongodb, "counter_obj": counter_obj, "source_info_obj": source_info_obj \
                , "track_info_obj": track_info_obj}
        detect.run(**dets)

if __name__ == "__main__":
    main()