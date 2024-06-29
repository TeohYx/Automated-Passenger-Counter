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

    def select_preset_line(self, container, track_info_obj, source_info_obj):
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

        # Show the container, prompting user to select the preset line.
        # with first_container:
        # Display text
        container.header("Choose the preset line.")
        container.caption("The preset line will be used as counting purpose in the video source.")
        container.caption(
            Streamlit.css_caption("Red line - Alighting line", "red-color", "red"),
            unsafe_allow_html=True
            )
        container.caption(
            Streamlit.css_caption("Green line - Boarding line", "green-color", "green"),
            unsafe_allow_html=True
            )

        # jsonf = "presetline/preset_line.json"      
        jsonf = self.json_file  

        with open(jsonf, 'r') as file:
            data = json.load(file)

        extract_path(data)
        # print(image_path)

        # selected_image = container.selectbox(
        #     "Select", 
        #     options=list(self.image_path.keys()),
        #     # index=list(self.image_path.keys()).index(st.session_state.selectbox_value),
        #     key="selectbox",
        #     on_change=on_selectbox_change
        # )
        selected_image = container.selectbox("Select", options=list(self.image_path.keys()))
        st.session_state.line_selection = (selected_image, self.image_path[selected_image])
        container.image(self.image_path[selected_image], caption=selected_image)

        if container.button("Customize preset line"):
            # Feed in opencv draw line window
            detect.draw_image(source_info_obj.screenshot_file, track_info_obj, source_info_obj)
            st.rerun()
            # pass

        if container.button("Select"):
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

    @staticmethod
    def select_source(media_choser, container):
        """
        Prompt user to select source mode
        """
        container.header("Choose the media source")

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
                    file_extension = os.path.splitext(st.session_state['uploaded_file'].name)[1]
                    tfile = tempfile.NamedTemporaryFile(suffix=file_extension, delete=False)
                    tfile.write(st.session_state['uploaded_file'].read())
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
 
    def display_source(self, container):
        """
        Display the source in "current" container
        """
        container.markdown("---")
        container.header("Media source")

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

        if st.session_state['have_visualization']:
            tab1, tab2 = self.temp_empty[0].container().tabs(["General", "Interactive"])
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
                        sum = df.groupby(df.index // 5).apply(custom_agg).reset_index(drop=True)
                        
                        fig_in = px.bar(sum, x="time", y="in", title="in")
                        fig_out = px.bar(sum, x="time", y="out", title="out")
                        fig_net = px.bar(sum, x="time", y="net", title="net")

                        fig_in = adjust_chart(fig_in, sum['in'].max())
                        fig_out = adjust_chart(fig_out, sum['out'].max())
                        # fig_net = adjust_chart(fig_net, sum['net'].max())
                        
                        subtab2.plotly_chart(fig_in)
                        subtab2.plotly_chart(fig_out)
                        subtab2.plotly_chart(fig_net)
                    else:
                        subtab2.caption("At least 10 minutes of data are needed to display the visual")
                with subtab3:
                    if num_rows > 15:
                        sum = df.groupby(df.index // 15).apply(custom_agg).reset_index(drop=True)
 
                        fig_in = px.bar(sum, x="time", y="in", title="in")
                        fig_out = px.bar(sum, x="time", y="out", title="out")
                        fig_net = px.bar(sum, x="time", y="net", title="net")

                        fig_in = adjust_chart(fig_in, sum['in'].max())
                        fig_out = adjust_chart(fig_out, sum['out'].max())
                        # fig_net = adjust_chart(fig_net, sum['net'].max())
                        
                        subtab3.plotly_chart(fig_in)
                        subtab3.plotly_chart(fig_out)
                        subtab3.plotly_chart(fig_net)

                    else:
                        subtab3.caption("At least 30 minutes of data are needed to display the visual")
                with subtab4:
                    if num_rows > 30:
                        sum = df.groupby(df.index // 30).apply(custom_agg).reset_index(drop=True)
                        
                        fig_in = px.bar(sum, x="time", y="in", title="in")
                        fig_out = px.bar(sum, x="time", y="out", title="out")
                        fig_net = px.bar(sum, x="time", y="net", title="net")

                        fig_in = adjust_chart(fig_in, sum['in'].max())
                        fig_out = adjust_chart(fig_out, sum['out'].max())
                        # fig_net = adjust_chart(fig_net, sum['net'].max())
                        
                        subtab4.plotly_chart(fig_in)
                        subtab4.plotly_chart(fig_out)
                        subtab4.plotly_chart(fig_net)

                    else:
                        subtab4.caption("At least 1 hour of data are needed to display the visual")
                    
                # self.temp_empty[1].text("lel")
        else:
            text_section = self.temp_empty[0].container()
            text_section.caption("The source media is less than 120 seconds, interactive section is unavailable.")
            text_section.caption("The following display the number of passenger boarded, alighted, as well as the passenger in the public bus.")
            display_general(text_section)
        
        # time.sleep(0.1)

    def display_history_visualization(self):
        """
        Display the history visualization in "history" selection
        """
        self.container[1].caption("This page visualize the history counting.")
        dis_tab1, dis_tab2 = self.container[1].tabs(["Full day", "Every run"])

        db = Database()

        collections = db.db.list_collection_names()

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
            documents = list(db.db[collection].find())
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

    def display_current_interface(self, track_info_obj, source_info_obj):
        """
        Display the camera source and visualization when "current" selection is selected
        """
        # def toggle_container():
        #     st.session_state.show_container = not st.session_state.show_container

        if st.session_state.show_container:
            # first_container.empty()
            self.select_preset_line(self.container[0], track_info_obj, source_info_obj)
        else:
            self.container[0].text(f"Selected line: {st.session_state.line_selection[0]}")
            self.display_source(self.container[0])
            # self.display_current_visualization(self.container[0])

    @staticmethod
    def set_containers():
        """
        Set the empty container for better interface
        """
        first_container = st.container()
        second_container = st.container()
        third_container = st.container()
        fourth_container = st.container()

        return first_container, second_container, third_container, fourth_container

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