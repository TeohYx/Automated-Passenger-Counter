import cv2
import os
import json
import numpy as np

class TrackInfo():
    """
    Store the infomation related to tracking, such as the frame screenshot for tracking purpose, inner line indicating boarding and outer line indicating alighting.
    
    """
    preset_line_json = None
    preset_line_path = None

    def __init__(self, preset_line_json, preset_line_path):
        TrackInfo.preset_line_json = preset_line_json
        TrackInfo.preset_line_path = preset_line_path

        self.inner_line = None  # ((x1, y1, x2, y2), BGR)    # This is based on original resolution
        self.outer_line = None  # This is based on original resolution
        self.start_point = None
        self.end_point = None
        self.drawing = False

        self.state = None
        self.have_line = False
        self.set_line()

    def set_start_point(self, start_point):
        self.start_point = start_point

    def set_end_point(self, end_point):
        self.end_point = end_point

    def set_inner_line(self, inner_line):
        self.inner_line = inner_line

    def set_outer_line(self, outer_line):
        self.outer_line = outer_line

    def store_plain_image(self, screenshot_path):
        with open(TrackInfo.preset_line_json, 'r') as file:
            data = json.load(file)      

        data['plain_image'] = screenshot_path 

        # Save json file
        with open(TrackInfo.preset_line_json, 'w') as file:
            json.dump(data, file, indent=4)

    def set_line(self):
        """
        Set up object's inner and outer line is it is specified in the json file, return is none is set
        """
        with open(TrackInfo.preset_line_json, 'r') as file:
            data = json.load(file)
        
        selection = data['selection']
        if not selection:   # means line not yet set
            return

        # Check if the line had value
        self.have_line = bool(data['lastsaved']['path'])

        # Set the line for inner and outer from json file
        inner_line = int(data[selection]['inner']['p1'][0]), int(data[selection]['inner']['p1'][1]), int(data[selection]['inner']['p2'][0]), int(data[selection]['inner']['p2'][1])
        outer_line = int(data[selection]['outer']['p1'][0]), int(data[selection]['outer']['p1'][1]), int(data[selection]['outer']['p2'][0]), int(data[selection]['outer']['p2'][1])

        self.set_inner_line((inner_line, tuple(data[selection]['inner']['color'])))
        self.set_outer_line((outer_line, tuple(data[selection]['outer']['color'])))

    @staticmethod
    def set_json_information(data, key, inner_p1, inner_p2, outer_p1, outer_p2):
        """
        Set the json information correctly with points
        """
        data[key]["inner"]['p1'] = inner_p1
        data[key]["inner"]['p2'] = inner_p2
        data[key]["outer"]['p1'] = outer_p1
        data[key]["outer"]['p2'] = outer_p2

    @staticmethod
    def create_and_store_image(data, key, screenshot_file):
        """
        Store screenshot image with given preset line
        """
        source_image = cv2.imread(screenshot_file)

        img = cv2.line(source_image, tuple(data[key]["inner"]["p1"]), tuple(data[key]["inner"]["p2"]), data[key]["inner"]["color"], 2)
        img = cv2.line(img, tuple(data[key]["outer"]["p1"]), tuple(data[key]["outer"]["p2"]), data[key]["outer"]["color"], 2)

        # Store image path in json file
        data[key]["path"] = f"{TrackInfo.preset_line_path}/{key}.png"
        
        # Save image
        cv2.imwrite(data[key]["path"], img)
        print("Chekcer")

    @staticmethod
    def store_lastsaved_line(screenshot_file):
        """
        Set the lastsaved preset line 
        Then create the image for each of the preset line and save them in the given json file
        """
        try:
            with open(TrackInfo.preset_line_json, 'r') as file:
                data = json.load(file)
        except FileNotFoundError:
            print("File not found.")
            # do streamlit line here
            return       

        TrackInfo.create_and_store_image(data, "lastsaved", screenshot_file)

        # Save json file
        with open(TrackInfo.preset_line_json, 'w') as file:
            json.dump(data, file, indent=4)

        print(f"Successfully saved lastsaved line in {TrackInfo.preset_line_json}")

    @staticmethod
    def store_default_line(source_size, screenshot_file):
        """
        Set the default preset line (vertical, vertical inverse, horizontal, horizontal inverse)
        Then create the image for each of the preset lines and save them both in the given json file

        It checks for the source_size to decide whether to run the function. If the source size are different when comparing with the json file source size, then it will run again.
        """
        try:
            with open(TrackInfo.preset_line_json, 'r') as file:
                data = json.load(file)
        except FileNotFoundError:
            print("File not found.")
            # do streamlit line here
            return

        height, width = source_size[0], source_size[1]

        print(f"Source shape: {(height, width)}")
        print(f"json shape: {(data['source_size']['height'], data['source_size']['width'])}")

        if data['source_size']['height'] != height or data['source_size']['width'] != width:
            # Calculate the line proportion, in 45% and 55%
            height_upper = int(height * 0.45)
            height_lower = int(height * 0.55)
            width_left = int(width * 0.45)
            width_right = int(width * 0.55)

            # Red line at left, green line at right, in vertical line
            TrackInfo.set_json_information(data, "vertical_left_boarding", [width_left, 0], [width_left, height], [width_right, 0], [width_right, height])

            # Red line at left, green line at right, in vertical line
            TrackInfo.set_json_information(data, "vertical_right_boarding", [width_right, 0], [width_right, height], [width_left, 0], [width_left, height])

            # Red line at above, green line at bottom, in horizontal line
            TrackInfo.set_json_information(data, "horizontal_above_boarding", [0, height_upper], [width, height_upper], [0, height_lower], [width, height_lower])

            # Red line at bottom, green line at above, in horizontal line
            TrackInfo.set_json_information(data, "horizontal_below_boarding", [0, height_lower], [width, height_lower], [0, height_upper], [width, height_upper])

            data['source_size']['height'] = height
            data['source_size']['width'] = width
        else:
            print("The default preset line with this resolution had already created.")
        
        # Create image and lines for vertical left boarding 
        TrackInfo.create_and_store_image(data, "vertical_left_boarding", screenshot_file)

        # Create image and lines for vertical right boarding 
        TrackInfo.create_and_store_image(data, "vertical_right_boarding", screenshot_file)

        # Create image and lines for horizontal left boarding 
        TrackInfo.create_and_store_image(data, "horizontal_above_boarding", screenshot_file)

        # Create image and lines for horizontal right boarding 
        TrackInfo.create_and_store_image(data, "horizontal_below_boarding", screenshot_file)

        # Save json file
        with open(TrackInfo.preset_line_json, 'w') as file:
            json.dump(data, file, indent=4)

        print(f"Successfully saved default file in {TrackInfo.preset_line_json}")

    @staticmethod
    def reset_dict(data, key):
        data[key]["inner"]["color"] = [0, 255, 0]
        data[key]["outer"]["color"] = [0, 0, 255]
        data[key]["inner"]["p1"] = [-1, -1]
        data[key]["inner"]["p2"] = [-1, -1]
        data[key]["outer"]["p1"] = [-1, -1]
        data[key]["outer"]["p2"] = [-1, -1]
        data[key]["path"] = ""

    @staticmethod
    def reset_json():
        """
        This is for developer uses only. Reset the json file to its initial value.
        """
        try:
            with open(TrackInfo.preset_line_json, 'r') as file:
                data = json.load(file)
        except FileNotFoundError:
            print("File not found.")
            # do streamlit line here
            return
        
        data['source_size']['height'] = -1 
        data['source_size']['width'] = -1
        data['selection'] = ""
        data['plain_image'] = ""
        TrackInfo.reset_dict(data, "vertical_left_boarding")
        TrackInfo.reset_dict(data, "vertical_right_boarding")
        TrackInfo.reset_dict(data, "horizontal_above_boarding")
        TrackInfo.reset_dict(data, "horizontal_below_boarding")
        TrackInfo.reset_dict(data, "lastsaved")

        with open(TrackInfo.preset_line_json, "w") as file:
            json.dump(data, file, indent=4)

        print("Successfully reset the json file.")

    @staticmethod
    def reset_lastsaved_line():
        """
        This is for developer uses only. Reset the last saved json file to its initial value.
        """
        try:
            with open(TrackInfo.preset_line_json, 'r') as file:
                data = json.load(file)
        except FileNotFoundError:
            print("File not found.")
            # do streamlit line here
            return

        TrackInfo.reset_dict(data, "lastsaved")

        with open(TrackInfo.preset_line_json, "w") as file:
            json.dump(data, file, indent=4)

        print("Successfully reset the json file.")

    def store_line_info(self, source_info_obj):
        """
        Store line information in self.inner_line and self.outer_line
        In addition, it also save the line file with name based on self.inner_line_file or self.outer_line_file
        """
        # Revert preset lines back to original resolution 
        start_point_x = int((self.start_point[0] / source_info_obj.resized_width) * source_info_obj.width)
        start_point_y = int((self.start_point[1] / source_info_obj.resized_height) * source_info_obj.height)
        end_point_x = int((self.end_point[0] / source_info_obj.resized_width) * source_info_obj.width)
        end_point_y = int((self.end_point[1] / source_info_obj.resized_height) * source_info_obj.height)

        if self.state == "inner":
            color = self.inner_line[1]
            self.inner_line = (start_point_x, start_point_y, end_point_x, end_point_y), color

            with open(TrackInfo.preset_line_json, 'r') as file:
                data = json.load(file)

            data['lastsaved']['inner']['p1'] = [start_point_x, start_point_y]
            data['lastsaved']['inner']['p2'] = [end_point_x, end_point_y]

            with open(TrackInfo.preset_line_json, 'w') as file:
                json.dump(data, file, indent=4)

            # with open(self.inner_line_file, 'w') as file:
            #     file.write(f"{start_point_x}, {start_point_y}, {end_point_x}, {end_point_y}")
            self.state = None
        elif self.state == "outer":
            color = self.outer_line[1]
            self.outer_line = (start_point_x, start_point_y, end_point_x, end_point_y), color

            with open(TrackInfo.preset_line_json, 'r') as file:
                data = json.load(file)

            data['lastsaved']['outer']['p1'] = [start_point_x, start_point_y]
            data['lastsaved']['outer']['p2'] = [end_point_x, end_point_y]

            with open(TrackInfo.preset_line_json, 'w') as file:
                json.dump(data, file, indent=4)
 
            self.state = None  
        else:
            if self.state == None:
                return
            raise ValueError("The 'state' argument in draw_line() function only takes 'inner' or 'outer'!")

    def draw_line_control(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.start_point = (x, y)
            self.drawing = True
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.end_point = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.end_point = (x, y)
            self.drawing = False

    def draw_line_on_frame(self, image):
        """
        Draw the inner and outer line on the frame
        """
        cv2.line(image, tuple(self.inner_line[0][:2]), tuple(self.inner_line[0][2:]), self.inner_line[1], 2)
        cv2.line(image, tuple(self.outer_line[0][:2]), tuple(self.outer_line[0][2:]), self.outer_line[1], 2)
