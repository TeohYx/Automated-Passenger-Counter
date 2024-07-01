import cv2
import os
import numpy as np

class SourceInfo():
    """
    Store all information related to the video
    """
    def __init__(self, source, screenshot_file, resized_height=720, resized_width=1280):
        self.source = source
        self.source_duration = None # video duration

        self.height = None
        self.width = None
        self.channels = None
        self.recording_first_frame = None
        self.screenshot_file = screenshot_file

        self.resized_height = resized_height
        self.resized_width = resized_width

        self.screenshot_first_frame()
        self.get_source_duration()

    def get_source_duration(self):
        """
        Get duration of the video
        """
        cap = cv2.VideoCapture(self.source)
        total_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_per_second = cap.get(cv2.CAP_PROP_FPS)
        self.source_duration = total_frame_count / frame_per_second

    def screenshot_first_frame(self):
        """
        Get screenshot of first frame. 
        This screenshot will be useful in drawing inner_line and outer_line 
        """
        cap = cv2.VideoCapture(self.source)

        ret, frame = cap.read()
        if not ret:
            raise Exception("Some error occured while taking screenshot.")

        self.recording_first_frame = frame
        self.height, self.width, self.channels = frame.shape  
        print(f"Shape is: {self.height, self.width}")
        # if os.path.exists(screenshot_file):
        #     print(f"The screenshot is existed and is in {screenshot_file}.")

        #     # frame = cv2.imread(screenshot_file)
        #     return 
        
        cv2.imwrite(self.screenshot_file, frame)
        print(f"File saved successfully: {self.screenshot_file}")

        # print(f"Image size: Width = {width}, Height = {height}, Channels = {channels}")

        cap.release()