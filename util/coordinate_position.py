import cv2
import os
import json

script_directory = os.path.dirname(os.path.realpath(__file__))

save_file = os.path.join(script_directory, "time_position.json")
save_image_file = os.path.join(script_directory, "time_position.png")
video_file = "camera.png"
start_x = None
start_y = None
end_x = None
end_y = None
drawing = False

def json_extractor(path):
    with open(path, 'r') as json_file:
        data = json.load(json_file)

    x1 = min(data['x1'], data['x2'])
    x2 = max(data['x1'], data['x2'])
    y1 = min(data['y1'], data['y2'])
    y2 = max(data['y1'], data['y2'])

    return x1, y1, x2, y2

def save_point():
    global save_file, start_x, start_y, end_x, end_y

    coordinate = {
        "x1": start_x,
        "y1": start_y,
        "x2": end_x,
        "y2": end_y
    }

    with open(save_file, 'w') as json_file:
        json.dump(coordinate, json_file, indent=4)


def save_image(image):
    global save_image_file, start_x, start_y, end_x, end_y

    y1 = min(start_y, end_y)
    y2 = max(start_y, end_y)
    x1 = min(start_x, end_x)
    x2 = max(start_x, end_x)

    cropped_image = image[y1:y2, x1:x2]
    cv2.imwrite(save_image_file, cropped_image)


def draw_rectangle(event, x, y, flags, param):
    global start_x, start_y, end_x, end_y, drawing
    first_frame = param[0]
    if event == cv2.EVENT_LBUTTONDOWN:
        start_x, start_y = x, y
        drawing = True
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        frame_copy = first_frame.copy()
        cv2.rectangle(frame_copy, (start_x, start_y), (x, y), (255, 255, 255), 2)
        cv2.imshow("vid", frame_copy)
    elif event == cv2.EVENT_LBUTTONUP and drawing:
        x = 0 if x < 0 else x
        y = 0 if y < 0 else y

        end_x, end_y = x, y
        frame_copy = first_frame.copy()
        cv2.rectangle(frame_copy, (start_x, start_y), (end_x, end_y), (255, 255, 255), 2)
        cv2.imshow("vid", frame_copy)
        drawing = False
        

def main():
    if video_file.endswith(".mkv"):
        cap = cv2.VideoCapture(video_file)

        _, first_frame = cap.read()
    elif video_file.endswith("png"):
        first_frame = cv2.imread(video_file)

    cv2.imshow("vid", first_frame)
    print(first_frame.shape)
    cv2.setMouseCallback("vid", draw_rectangle, [first_frame])
    
    # if cv2.waitKey(0) & 0xFF == ord('d'):
    #     save_point()
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            return
        elif key == 13:
            save_point()
            save_image(first_frame)
            print(f"json file saved in {save_file}, while image saved in {save_image_file}.")
            return


if __name__ == "__main__":
    main()
