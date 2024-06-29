import cv2
import sys
import keyboard
import os

class Bounding_box:
    def __init__(self, path=None, frame=None):
        self.path = path                    # Path of the video
        self.frame = frame                  # The first frame captured as image
        self.roi_value = []                 # The temporary roi value
        self.roi_value_all = []             # All roi value 
        self.height = None
        self.width = None
        self.start_point = None
        self.end_point = None
        self.drawing = False

    #  Extract the first frame as image
    def screenshot(self):
        if self.path is None:
            return
        video_path = self.path
        cap = cv2.VideoCapture(video_path)

        ret, frame = cap.read()
        if not ret:
            return

        cv2.imwrite('screenshot.jpg', frame)

        # Get the image of the frame
        self.frame = cv2.imread('screenshot.jpg')
        # print("width is: ", self.frame.Width)
        height, width, channels = self.frame.shape
        self.height = height
        self.width = width
        print(f"Image size: Width = {width}, Height = {height}, Channels = {channels}")

        cap.release()
        cv2.destroyAllWindows()

    # Define a ROI
    """
    Format: 
    [x, y, w, h]
    x - Top left x coordinate
    y - Top left y coordinate
    w - Width
    h - Height
    """
    def region_of_interest(self):
        roi = cv2.selectROI("Select ROIs", self.frame, showCrosshair=True)
        self.roi_value = list(roi)

        # print(roi)
        cv2.destroyAllWindows()

    def draw_line(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.start_point = (x, y)
            self.drawing = True
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.end_point = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.end_point = (x, y)
            self.drawing = False
# Do a normalization so that it fits on the video
            
def resize(value, bb):
    result = []
    for c, x, y, w, h in value:
        xn = round(x/bb.width, 4)
        yn = round(y/bb.height, 4)
        wn = round(w /bb.width, 4)
        hn = round(h/bb.height, 4)

        result.append((c, xn, yn, wn, hn))

    return result


def get_line(source, frame=None, line=None):
    if line in os.listdir():
        print("Line already drawn.")
        return

    bb = Bounding_box(source, cv2.imread(frame))
    bb.screenshot()
    image = bb.frame
    # print(image)
    # Set the window name
    window_name = 'Draw Line'

    # Display the image
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, bb.draw_line)

    while True:
        # Display the image
        temp_image = image.copy()
        if bb.start_point is not None and bb.end_point is not None:
            cv2.line(temp_image, bb.start_point, bb.end_point, color=(255, 255, 255), thickness=2)
        cv2.imshow(window_name, temp_image)

        # Check for ESC key press
        if cv2.waitKey(1) == 27:
            # bb.start_point = start_point
            # bb.end_point = end_point
            break

    cv2.destroyAllWindows()

    with open(line, 'w') as file:
        file.write(f"{bb.start_point[0]}, {bb.start_point[1]}, {bb.end_point[0]}, {bb.end_point[1]}")
    # return (bb.start_point, bb.end_point)