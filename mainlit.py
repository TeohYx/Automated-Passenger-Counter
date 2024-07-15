import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import json
import re

import cv2
import tempfile
import yolov9.detect as detect
import os
from datetime import datetime
# selection_current = "Current"
# selection_history = "History"

# st.set_page_config(layout="wide")
import time
from database.mongo import Database

from PIL import Image
import numpy as np
from streamlit_drawable_canvas import st_canvas

class Streamlit:
    selection_current = "Current"
    selection_history = "History"

    def __init__(self):
        self.json_file = None
        self.image_path = {}
        self.container = [] # each index indicate container for new content
        self.temp_empty = []

    def set_json_file(self, json_file):
        self.json_file = json_file

    @staticmethod
    def css_caption(text, style, color=None):
        css = f"""
        <style>
        .{style} {{
            margin-top: -15px;
            margin-bottom: -15px;
            color: {color};
        }}
        </style>
        <div class="{style}">
            {text}
        </div>
        """

        return css

    def select_preset_line(self):
        """
        Select the preset line in "current" container
        """
        # Get all the path and its json file name
        def extract_path(data, parent_key=''):
            if isinstance(data, dict):
                for key, value in data.items():
                    if key == "path":
                        # image_path.append((parent_key, value))
                        self.image_path[parent_key] = value
                    elif isinstance(value, (dict, list)):
                        extract_path(value, parent_key=parent_key if parent_key else key)
            elif isinstance(data, list):
                for item in data:
                    extract_path(item, parent_key=parent_key)

        def display_streamlit_canvas(data, jsonf):
            bg_image = data['plain_image']
            height = data['source_size']['height']
            width = data['source_size']['width']

            if bg_image:
                background_image = Image.open(bg_image)
            else:
                background_image = None

            canvas_width = 600
            canvas_height = 600*9/16      #16:9

            # Create a canvas component
            canvas_result = st_canvas(
                fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
                stroke_width=1,
                stroke_color="black",
                background_color="",
                background_image=background_image,
                update_streamlit=True,
                height=canvas_height,
                width=canvas_width,
                drawing_mode="line",
                display_toolbar=True,
                key="full_app",
            )

            # canvas = self.container[0].empty()

            if canvas_result.json_data is not None:
                objects = pd.json_normalize(canvas_result.json_data["objects"])
                for col in objects.select_dtypes(include=["object"]).columns:
                    objects[col] = objects[col].astype("str")

                if len(objects) == 2:
                    btn = self.container[0].button("Select the customized preset line")

                    if btn:
                        st.session_state["drawing_canvas"] = False
                        #  Center + half of the line
                        x11 = objects.iloc[0]['left'] + objects.iloc[0]['x1']
                        y11 = objects.iloc[0]['top'] + objects.iloc[0]['y1']
                        x12 = objects.iloc[0]['left'] + objects.iloc[0]['x2']
                        y12 = objects.iloc[0]['top'] + objects.iloc[0]['y2']

                        x21 = objects.iloc[1]['left'] + objects.iloc[0]['x1']
                        y21 = objects.iloc[1]['top'] + objects.iloc[0]['y1']
                        x22 = objects.iloc[1]['left'] + objects.iloc[0]['x2']
                        y22 = objects.iloc[1]['top'] + objects.iloc[0]['y2']

                        actual_x11 = int((x11 / canvas_width) * width)
                        actual_y11 = int((y11 / canvas_height) * height)
                        actual_x12 = int((x12 / canvas_width) * width)
                        actual_y12 = int((y12 / canvas_height) * height)

                        actual_x21 = int((x21 / canvas_width) * width)
                        actual_y21 = int((y21 / canvas_height) * height)
                        actual_x22 = int((x22 / canvas_width) * width)
                        actual_y22 = int((y22 / canvas_height) * height)

                        print(f"{actual_x11}, {actual_y11}, {actual_x12}, {actual_y12}")
                        print(f"{actual_x21}, {actual_y21}, {actual_x22}, {actual_y22}")

                        data['lastsaved']['inner']['p1'] = [actual_x11, actual_y11]
                        data['lastsaved']['inner']['p2'] = [actual_x12, actual_y12]
                        data['lastsaved']['outer']['p1'] = [actual_x21, actual_y21]
                        data['lastsaved']['outer']['p2'] = [actual_x22, actual_y22]

                        with open(jsonf, 'w') as file:
                            json.dump(data, file, indent=4)

                        st.rerun()
                elif len(objects) < 2:
                    self.container[0].caption("Draw one line first for :green[Boarding line], and second line for :red[Alighting line]")
                elif len(objects) > 2:
                    self.container[0].caption("Too much line drawn, only two is needed, which are :green[Boarding line] and :red[Alighting line]")

            if self.container[0].button("Back"):
                st.session_state["drawing_canvas"] = False
                st.rerun()
        # Show the container, prompting user to select the preset line.
        # with first_container:
        # Display text
        self.container[0].header("Choose the preset line.")
        self.container[0].caption("The preset line will be used as counting purpose in the video source.")
        self.container[0].caption(
            Streamlit.css_caption("Red line - Alighting line", "red-color", "red"),
            unsafe_allow_html=True
            )
        self.container[0].caption(
            Streamlit.css_caption("Green line - Boarding line", "green-color", "green"),
            unsafe_allow_html=True
            )
        self.container[0].markdown("---")

        # jsonf = "presetline/preset_line.json"      
        jsonf = self.json_file  

        with open(jsonf, 'r') as file:
            data = json.load(file)

        extract_path(data)

        if 'drawing_canvas' not in st.session_state:
            st.session_state['drawing_canvas'] = False

        if not st.session_state['drawing_canvas']:
            selected_image = self.container[0].selectbox("Select", options=list(self.image_path.keys()))
            st.session_state.line_selection = (selected_image, self.image_path[selected_image])
            self.container[0].image(self.image_path[selected_image], caption=selected_image)

            if self.container[0].button("Customize preset line"):
                # When click the button, display the canvas to draw
                st.session_state['drawing_canvas'] = True
                st.rerun()

            if self.container[0].button("Select"):
                print("isClicked")
                st.session_state.show_container = not st.session_state.show_container

                data["selection"] = selected_image

                with open(jsonf, 'w') as file:
                    json.dump(data, file)

                st.rerun()
                # if st.session_state.line_selection is None:
                #     pass
                # else:
                #     print(st.session_state.show_container)
        else:
            display_streamlit_canvas(data, jsonf)

    @staticmethod
    def select_source(media_choser):
        """
        Prompt user to select source mode
        """
        media_choser.header("Choose the media source")

        def store_video():
            """Store file in temporary folder"""
            # Get the file extension (eg: mp4)
            file_extension = os.path.splitext(st.session_state['uploaded_file'].name)[1]
            # Creates a temperory file and returns a file object
            tfile = tempfile.NamedTemporaryFile(suffix=file_extension, delete=False)
            # Writes the content of the uploaded file to the temporary file
            tfile.write(st.session_state['uploaded_file'].read())
            return tfile

        if not st.session_state['is_media_chosen']:
            tab1, tab2 = media_choser.tabs(['Recorded video', 'Real time'])
            with tab1:
                st.header("Recorded video")
                uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi", "mkv"])
                
                if uploaded_file is not None:
                    st.session_state['uploaded_file'] = uploaded_file

                # if uploaded_file:
                if st.session_state['uploaded_file'] is not None:
                    # print(st.session_state['uploaded_file'])

                    tfile = store_video()


                    # print(tfile)
                    st.session_state['is_media_chosen'] = True
                    return tfile.name
                  
            with tab2:
                st.header("Real-time source")
                
                current_value = st.text_input("Real-time URL (http / https / rtsp / etc.)")
                if current_value != st.session_state['streaming_url']:
                    st.session_state['is_media_chosen'] = True
                
                st.session_state['streaming_url'] = current_value
                return st.session_state['streaming_url']

    def display_current_visualization(self, mongodb):
        """
        Display the visualization regarding to the playing source in "current" container
        """
        if not self.temp_empty:
            for _ in range(3):
                self.temp_empty.append(self.container[0].empty())
        
        lists = list(mongodb.get_collection_col().find())

        cleaned_data = [{key:value for key, value in record.items() if key != '_id'} for record in lists]

        df = pd.DataFrame(cleaned_data)
        # print(lists)
        df['time'] = df['time'].apply(lambda x: datetime.strptime(x, '%Y%m%d_%H%M%S').strftime("%H:%M:%S") if not isinstance(x, int) else x)
        df['net'] = df['in'] - df['out']
        num_rows = df.shape[0]

        # self.temp_empty[0].bar_chart(data=df, x='time', y='in')
        # self.temp_empty[1].bar_chart(data=df, x='time', y='out')
        # self.temp_empty[2].bar_chart(data=df, x='time', y='net')

        def display_general(tab1):
            col1, col2, col3 = tab1.container().columns(3)
            col1.title(df['in'].sum())
            col1.title(":green[IN]")
            col2.title(df['out'].sum())
            col2.title(":red[OUT]")
            col3.title(df['net'].sum())
            col3.title(":blue[NET]")

        def custom_agg(group):
            return pd.Series({
                'time': group['time'].iloc[0],
                'in': group['in'].sum(),
                'out': group['out'].sum(),
                'net': group['net'].sum()
            })

        def adjust_chart(fig, max):
            fig.update_yaxes(range=[0, max + 1])  # Adjust the range as needed
            fig.update_layout(yaxis_tickformat='d')
            return fig
        
        def set_chart(container, df):
            fig_in = px.bar(df, x="time", y="in", title="in")
            fig_out = px.bar(df, x="time", y="out", title="out")
            fig_net = px.bar(df, x="time", y="net", title="net")

            fig_in = adjust_chart(fig_in, df['in'].max())
            fig_out = adjust_chart(fig_out, df['out'].max())
            # fig_net = adjust_chart(fig_net, df['net'].max())
            
            container.plotly_chart(fig_in)
            container.plotly_chart(fig_out)
            container.plotly_chart(fig_net)  

        if st.session_state['have_visualization']:
            tab1, tab2 = self.temp_empty[0].container().tabs(["General", "Interactive"])
            with tab1:
                display_general(tab1)
            with tab2:
                tab2.caption("Select in between 2min, 10min, 30min and 1hour for diffrent visualization.")

                subtab1, subtab2, subtab3, subtab4 = tab2.tabs(["2m", "10m", "30m", "1h"])
                
                with subtab1:
                    set_chart(subtab1, df)

                with subtab2:
                    if num_rows > 5:
                        sum = df.groupby(df.index // 5).apply(custom_agg).reset_index(drop=True)
                        set_chart(subtab2, sum)
                    else:
                        subtab2.caption("At least 10 minutes of data are needed to display the visual")
                with subtab3:
                    if num_rows > 15:
                        sum = df.groupby(df.index // 15).apply(custom_agg).reset_index(drop=True)
                        set_chart(subtab3, sum)
                    else:
                        subtab3.caption("At least 30 minutes of data are needed to display the visual")
                with subtab4:
                    if num_rows > 30:
                        sum = df.groupby(df.index // 30).apply(custom_agg).reset_index(drop=True)
                        set_chart(subtab4, sum)
                    else:
                        subtab4.caption("At least 1 hour of data are needed to display the visual")
        else:
            text_section = self.temp_empty[0].container()
            text_section.caption("The source media is less than 120 seconds, interactive section is unavailable.")
            text_section.caption("The following display the number of passenger boarded, alighted, as well as the passenger in the public bus.")
            display_general(text_section)
        
        # time.sleep(0.1)

    def display_history_visualization(self, mongodb):
        """
        Display the history visualization in "history" selection
        """
        self.container[1].caption("This page visualize the history counting.")
        dis_tab1, dis_tab2 = self.container[1].tabs(["Full day", "Every run"])

        collections = mongodb.db.list_collection_names()

        date_format = "%Y%m%d_%H%M%S"

        pattern = r"\(([^)]+)\)"
        new_format = "%Y%m%d"

        collection_list = {}

        def extract_day(collection):
            match = re.search(pattern, collection)

            if match:
                collection_time = match.group(1)

            date_object = datetime.strptime(collection_time, date_format)  
            date = date_object.strftime(new_format)

            return date
            # print(date)

        def display_general(tab1):
            col1, col2, col3 = tab1.container().columns(3)
            col1.title(df['in'].sum())
            col1.title(":green[IN]")
            col2.title(df['out'].sum())
            col2.title(":red[OUT]")
            col3.title(df['net'].sum())
            col3.title(":blue[NET]")

        def custom_agg(group):
            return pd.Series({
                'time': group['time'].iloc[0],
                'in': group['in'].sum(),
                'out': group['out'].sum(),
                'net': group['net'].sum()
            })

        def adjust_chart(fig, max):
            fig.update_yaxes(range=[0, max + 1])  # Adjust the range as needed
            fig.update_layout(yaxis_tickformat='d')
            return fig

        def visual_collection(df, name, dis_tab2):
            dis_tab2.text(name)
            tab1, tab2 = dis_tab2.tabs(["General", "Interactive"])
            with tab1:
                display_general(tab1)
            with tab2:
                tab2.caption("Select in between 2min, 10min, 30min and 1hour for diffrent visualization.")

                subtab1, subtab2, subtab3, subtab4 = tab2.tabs(["2m", "10m", "30m", "1h"])
                
                with subtab1:
                    fig_in = px.bar(df, x="time", y="in", title="in")
                    fig_out = px.bar(df, x="time", y="out", title="out")
                    fig_net = px.bar(df, x="time", y="net", title="net")

                    fig_in = adjust_chart(fig_in, df['in'].max())
                    fig_out = adjust_chart(fig_out, df['out'].max())
                    # fig_net = adjust_chart(fig_net, df['net'].max())
                    
                    subtab1.plotly_chart(fig_in)
                    subtab1.plotly_chart(fig_out)
                    subtab1.plotly_chart(fig_net)

                with subtab2:
                    if num_rows > 5:
                        df = df.groupby(df.index // 5).apply(custom_agg).reset_index(drop=True)
                        
                        fig_in = px.bar(df, x="time", y="in", title="in")
                        fig_out = px.bar(df, x="time", y="out", title="out")
                        fig_net = px.bar(df, x="time", y="net", title="net")

                        fig_in = adjust_chart(fig_in, df['in'].max())
                        fig_out = adjust_chart(fig_out, df['out'].max())
                        # fig_net = adjust_chart(fig_net, sum['net'].max())
                        
                        subtab2.plotly_chart(fig_in)
                        subtab2.plotly_chart(fig_out)
                        subtab2.plotly_chart(fig_net)
                    else:
                        subtab2.caption("At least 10 minutes of data are needed to display the visual")
                with subtab3:
                    if num_rows > 15:
                        df = df.groupby(df.index // 15).apply(custom_agg).reset_index(drop=True)

                        fig_in = px.bar(df, x="time", y="in", title="in")
                        fig_out = px.bar(df, x="time", y="out", title="out")
                        fig_net = px.bar(df, x="time", y="net", title="net")

                        fig_in = adjust_chart(fig_in, df['in'].max())
                        fig_out = adjust_chart(fig_out, df['out'].max())
                        # fig_net = adjust_chart(fig_net, sum['net'].max())
                        
                        subtab3.plotly_chart(fig_in)
                        subtab3.plotly_chart(fig_out)
                        subtab3.plotly_chart(fig_net)

                    else:
                        subtab3.caption("At least 30 minutes of data are needed to display the visual")
                with subtab4:
                    if num_rows > 30:
                        df = df.groupby(df.index // 30).apply(custom_agg).reset_index(drop=True)
                        
                        fig_in = px.bar(df, x="time", y="in", title="in")
                        fig_out = px.bar(df, x="time", y="out", title="out")
                        fig_net = px.bar(df, x="time", y="net", title="net")

                        fig_in = adjust_chart(fig_in, df['in'].max())
                        fig_out = adjust_chart(fig_out, df['out'].max())
                        # fig_net = adjust_chart(fig_net, sum['net'].max())
                        
                        subtab4.plotly_chart(fig_in)
                        subtab4.plotly_chart(fig_out)
                        subtab4.plotly_chart(fig_net)

                    else:
                        subtab4.caption("At least 1 hour of data are needed to display the visual")
            dis_tab2.markdown(
                """
                <hr style="border: 1px solid white;">
                """,
                unsafe_allow_html=True
            )

        
        for collection in collections:
            date = extract_day(collection)

            if date not in collection_list.keys():
                collection_list[date] = None
                # print(collection_list)
            # Get the document of the collection
            documents = list(mongodb.db[collection].find())
            cleaned_data = [{key:value for key, value in record.items() if key != '_id'} for record in documents]

            df = pd.DataFrame(cleaned_data)
            df['time'] = df['time'].apply(lambda x: datetime.strptime(x, '%Y%m%d_%H%M%S').strftime("%H:%M:%S") if not isinstance(x, int) else x)
            df['net'] = df['in'] - df['out']
            num_rows = df.shape[0]

            previous_df = collection_list[date]

            if previous_df is None:
                collection_list[date] = df
            else:
                df = pd.concat([previous_df, df], ignore_index=True)
                collection_list[date] = df
        # print(collection_list)
        # # Display visual
        # for key, coll in collection_list.items():
            
            with dis_tab2:
                visual_collection(df, collection, dis_tab2)
            
            # tab1, tab2 = dis_tab2.tabs(["General", "Interactive"])
        keys = collection_list.keys()
        tabs = dis_tab1.tabs(keys)
        with dis_tab1:
            for tab, name in zip(tabs, keys):
                print(f"name is {name}")
                with tab:         
                    for key, coll in collection_list.items():
                        
                        if name == key:
                            print(f"key is {key}")
                            fig_in = px.bar(coll, x="time", y="in", title="in")
                            fig_out = px.bar(coll, x="time", y="out", title="out")
                            fig_net = px.bar(coll, x="time", y="net", title="net")

                            tab.plotly_chart(fig_in)
                            tab.plotly_chart(fig_out)
                            tab.plotly_chart(fig_net)

    def display_current_interface(self):
        """
        Display the camera source and visualization when "current" selection is selected
        """
        # def toggle_container():
        #     st.session_state.show_container = not st.session_state.show_container

        if st.session_state.show_container:
            # first_container.empty()
            self.select_preset_line()
        else:
            self.container[0].text(f"Selected line: {st.session_state.line_selection[0]}")
            self.container[0].markdown("---")
            self.container[0].header("Media source")

            # self.display_current_visualization(self.container[0])

    # @staticmethod
    # def set_containers():
    #     """
    #     Set the empty container for better interface
    #     """
    #     first_container = st.container()
    #     second_container = st.container()
    #     third_container = st.container()
    #     fourth_container = st.container()

    #     return first_container, second_container, third_container, fourth_container

def main():
    strlit = Streamlit()
    strlit.set_json_file("presetline/preset_line.json")
    st.title("Automated Passenger Counter")
    st.markdown("---")
    
    first_container, second_container, third_container, fourth_container = strlit.set_containers()

    # Setting for visualization viewing, with current and history selection
    selection = st.sidebar.selectbox("Sidebar selection", (strlit.selection_current, strlit.selection_history))

    if selection == strlit.selection_current:
        strlit.display_current_interface(first_container, second_container, third_container, fourth_container)
    elif selection == strlit.selection_history:
        strlit.display_history_interface(first_container, second_container)


if __name__ == "__main__":
   main()