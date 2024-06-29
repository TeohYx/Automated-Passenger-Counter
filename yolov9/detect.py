import argparse
import os
import platform
import sys
from pathlib import Path
from datetime import datetime
import re

import torch
import streamlit as st
import pandas as pd

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLO root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode

#---------------Object Tracking---------------
from track.sort import *
# from track.bounding_box import get_line 


#-----------Object Blurring-------------------
blurratio = 40

#.................. Tracker Functions .................
'''Computer Color for every box and track'''
palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
def compute_color_for_labels(label):
    color = [int(int(p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)

def draw_boxes(img, bbox, identities=None, categories=None, 
                names=None, color_box=None, confidence=None,offset=(0, 0)):
    """
    Draw Bounding boxes of tracked object
    """
    # print(identities)
    # print(bbox)
    # print(f"img is {img} \n bbox is {bbox} \n categories is {categories} \n names is {names} \n color_box is {color_box} \n offset is {offset}")
    # print(confidence)
    
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        cat = int(categories[i]) if categories is not None else 0       # CATEGORY
        id = int(identities[i]) if identities is not None else 0        # UNIQUE ID
        conf = round(float(confidence[i]), 2) if confidence is not None else 0      # CONFIDENCE SCORE
        data = (int((box[0]+box[2])/2),(int((box[1]+box[3])/2)))

        """LABEL APPEARING ON TOP OF THE TRACKED BOX"""
        label = str(f"Id is: {id}, {names[cat]} Conf: {conf}")

        if color_box:
            color = compute_color_for_labels(id)
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(img, (x1, y1), (x2, y2),color, 2)
            cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), (255,191,0), -1)
            cv2.putText(img, label, (x1, y1 - 5),cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
            [255, 255, 255], 1)
            cv2.circle(img, data, 3, color,-1)
        else:
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(img, (x1, y1), (x2, y2),(255,191,0), 2)
            cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), (255,191,0), -1)
            cv2.putText(img, label, (x1, y1 - 5),cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
            [255, 255, 255], 1)
            cv2.circle(img, data, 3, (255,191,0),-1)
    return img
#..............................................................................

def create_save_folder(path):
    """
    Create new file and prevent from overriding existing files.

    Return unique file name in string
    """
    attempt = 0

    folder_path = f"{path}{attempt}"
    while os.path.exists(folder_path):
        attempt += 1
        folder_path = f"{path}{attempt}"

    return folder_path

def put_text(image, text, font=cv2.FONT_HERSHEY_SIMPLEX, scale=1, thickness=2, line_spacing=1.3):
    """
    Draw image
    """
    # print(image.shape)
    resized_y, resized_x, _ = image.shape

    text = text.split('\n')

    (_, text_height), _ = cv2.getTextSize(text[0], font, scale, thickness)
    y = text_height + 10

    for txt in text:
        (txt_width, txt_height), _ = cv2.getTextSize(txt, font, scale, thickness)
        x = resized_x - txt_width - 10
        cv2.putText(image, txt, (x, y), font, scale, (0, 255, 0), thickness)
        y += int(txt_height * line_spacing)


def image_preset_line_checker(screenshot_file, track_info_obj, source_info_obj):
    """
    Display the first frame image with drawn line
    """
    resized_x = source_info_obj.resized_width
    resized_y = source_info_obj.resized_height
    # Set the window name
    window_name = 'First frame'

    # Display the image
    image = cv2.imread(screenshot_file).copy()
    cv2.namedWindow(window_name)
    track_info_obj.draw_line_on_frame(image) # Draw prediction line on image

    while True:
        image = cv2.resize(image, (resized_x, resized_y))
        cv2.imshow(window_name, image)
            # Check for ESC key press
        
        put_text(image, "y - Continue with this preset line\nn - Redraw the line")

        # Ask for confirmation 
        key = cv2.waitKey(1) & 0xFF

        if key == ord('y'):  # Agree with the preset line
            break
        elif key == ord('n'):  # Not agree with the preset line
            draw_image(screenshot_file, track_info_obj, source_info_obj)
            break

    cv2.destroyAllWindows() 


def draw_image(screenshot_file, track_info_obj, source_info_obj):
    """
    Draw line for inner and outer line 
    Usually being called when new line needed to be draw on current source (img)
    """
    image = source_info_obj.recording_first_frame
    cv2.imwrite(source_info_obj.screenshot_file, image)

    # Manually draw inner line
    preset_tracking_line(screenshot_file, track_info_obj, source_info_obj, "inner", text="Draw inner line indicating boarding line")
    track_info_obj.store_line_info(source_info_obj)

    # Manually draw outer line
    preset_tracking_line(screenshot_file, track_info_obj, source_info_obj, "outer", text="Draw outer line indicating alighting line")
    track_info_obj.store_line_info(source_info_obj) 

    # Create and store the image with line 
    track_info_obj.store_lastsaved_line(screenshot_file)


def draw_and_display_image(screenshot_file, track_info_obj):
    """
    Draw and display the first frame image
    """
    # Set the window name
    window_name = 'Draw Line'

    # Display the image
    image = cv2.imread(screenshot_file)
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, track_info_obj.draw_line_control)

    while True:
        # Display the image
        temp_image = image.copy()
        if track_info_obj.start_point is not None and track_info_obj.end_point is not None:
            cv2.line(temp_image, track_info_obj.start_point, track_info_obj.end_point, color=(255, 255, 255), thickness=2)
        cv2.imshow(window_name, temp_image)

        # Check for ESC key press
        if cv2.waitKey(1) == 27:
            # bb.start_point = start_point
            # bb.end_point = end_point
            break

    cv2.destroyAllWindows()
    

def preset_tracking_line(screenshot_file, track_info_obj, source_info_obj, state, text=" "):
    """
    Draw and display the first frame image
    """
    # is_exit = False
    track_info_obj.state = state
    resized_res = source_info_obj.resized_width, source_info_obj.resized_height

    # Read the image
    image = source_info_obj.recording_first_frame.copy()

    # Draw inner line if already inputted
    if track_info_obj.inner_line[0] is not None:
        cv2.line(image, (track_info_obj.inner_line[0][0], track_info_obj.inner_line[0][1]),
                 (track_info_obj.inner_line[0][2], track_info_obj.inner_line[0][3]), color=track_info_obj.inner_line[1], thickness=2)

    # Draw outer line if already inputted
    if track_info_obj.outer_line[0] is not None:
        cv2.line(image, (track_info_obj.outer_line[0][0], track_info_obj.outer_line[0][1]),
                 (track_info_obj.outer_line[0][2], track_info_obj.outer_line[0][3]), color=track_info_obj.outer_line[1], thickness=2)

    # Set the window name
    window_name = 'Draw Line'
    # print(source_info_obj.recording_first_frame)

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, track_info_obj.draw_line_control)

    while True:
        resized_image = cv2.resize(image, resized_res)
        # Display the image
        if track_info_obj.start_point is not None and track_info_obj.end_point is not None:
            cv2.line(resized_image, track_info_obj.start_point, track_info_obj.end_point, color=(255, 255, 255), thickness=2)

        put_text(resized_image, text)   # Display text at top right corner
        cv2.imshow(window_name, resized_image)

        # Check for ESC key press
        if cv2.waitKey(1) == 27:
            print("exited")
            # bb.start_point = start_point
            # bb.end_point = end_point
            is_exit = True
            break

    cv2.destroyAllWindows()
    # return is_exit


def create_document(datetime_tesseract, mongodb):
    """
    Initialize document 
    """
    print(datetime_tesseract)
    if not isinstance(datetime_tesseract, int):
        document_time_str = datetime_tesseract.strftime("%Y%m%d_%H%M%S")
    else:
        document_time_str = datetime_tesseract
    mongodb.initialize_document(document_time_str)

def visualization(mongodb, in_bar, out_bar, net_bar):
    lists = list(mongodb.get_collection_col().find())

    cleaned_data = [{key:value for key, value in record.items() if key != '_id'} for record in lists]

    df = pd.DataFrame(cleaned_data)

    df['time'] = df['time'].apply(lambda x: datetime.strptime(x, '%Y%m%d_%H%M%S').strftime("%H:%M:%S"))

    df['net'] = df['in'] - df['out']

    in_bar.bar_chart(data=df, x='time', y='in')
    out_bar.bar_chart(data=df, x='time', y='out')
    net_bar.bar_chart(data=df, x='time', y='net')

@smart_inference_mode()
def run(
        weights=ROOT / 'yolo.pt',  # model path or triton URL
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
        color_box=False,
        strlit=None,  # Streamlit object
        mongodb=None,
        counter_obj=None,
        source_info_obj=None,
        track_info_obj=None
):
    #.... Initialize preset lines ....
    # Set up 4 default lines, update the image and source size into the json file
    track_info_obj.store_default_line((source_info_obj.height, source_info_obj.width), source_info_obj.screenshot_file)
    
    # Set up manual line in lastsaved section and update the image into the json file
    track_info_obj.store_lastsaved_line(source_info_obj.screenshot_file)
    
    last_frame_image = None
    #.........................

    #.... Initialize SORT .... 
    sort_max_age = 5 
    sort_min_hits = 2
    sort_iou_thresh = 0.2
    sort_tracker = Sort(max_age=sort_max_age,
                        min_hits=sort_min_hits,
                        iou_threshold=sort_iou_thresh) 
    track_color_id = 0
    #......................... 

    # Check for the format of sources, either if its file (img, vid)/dir/URL/glob/screen/0(webcam)
    source = str(source)    # file/dir/URL/glob/screen/0(webcam)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)    # Check is the file is Image or Video by the suffix
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Setting up saving folder that store counted object (with its current and previous frame)
    save_attept_path = "image_save/attempt"
    folder_path = create_save_folder(save_attept_path)

    # Load model
    device = select_device(device)

    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)

    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        # view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)

        st.session_state.media_mode = "streaming"   # Streamlit

        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)

        st.session_state.media_mode = "screenshot"   # Streamlit

    else:
        # Create a new video capture object to dataset
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)

        st.session_state.media_mode = "recorded video"   # Streamlit

    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())

    # When recorded video is selected, or real-time address is given, it hides the selection and prompt with the "Play media" button
    if st.session_state.show_container == False:
        btn = strlit.container[0].button("Play media")  # Button

        media_display = strlit.container[0].empty()

        cut_off_time = 120  # count passenger in 2min interval in database, for recorded video and streaming media
        cut_off_frame = None    

        start_time = time.time()    # For identify runtime FPS
        time_interval = 0
        frame = 0
        first_stream_checker = True

        temp_frame = 0
        time_temp = 0

        collector_frame = 0
    
        if btn:
            strlit.container[0].markdown("---")
            strlit.container[0].header("Visualization")

            # Loop, starts runnning the video or streaming media
            for path, im, im0s, vid_cap, s in dataset:
                temp_time = time.time()

                # Manage database MongoDB
                if st.session_state['streaming_url'] == "": # Manage collection documents for recorded video
                    pattern = re.search(r"\(\d+/\d+\)", s).group()
                    number_string = pattern.strip("()")
                    current_frame = int(number_string.split('/')[0])    # get the current frame

                    if not cut_off_frame and source_info_obj.source_duration > cut_off_time:    # Initialize cutofftime and make sure only run once
                        cut_off_frame = int((cut_off_time * int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT)) / source_info_obj.source_duration) / vid_stride)

                    if cut_off_frame:   # Video >2min, time is included in document
                        st.session_state['have_visualization'] = True
                        if first_stream_checker:
                            mongodb.document = datetime.now()
                            create_document(mongodb.document, mongodb)
                            first_stream_checker = False
                        # Create new document when exceeded 120 seconds, with 'key' being the time at the moment
                        if current_frame % cut_off_frame == 0:
                            print(f"these are {cut_off_frame} and {current_frame}")
                            mongodb.document = datetime.now()
                            create_document(mongodb.document, mongodb)
                    else:   # Vidoe <2min, only constant number is used, meaning one document each run
                        # Create only one document with 'key' = 1, cause of less than 120seconds (cutofftime)
                        if first_stream_checker:
                            mongodb.document = 1
                            create_document(mongodb.document, mongodb)
                            first_stream_checker = False
                else:   # Manage collection database for streaming media
                    st.session_state['have_visualization'] = True
                    if first_stream_checker:
                        # Create new document for the first time, with 'key' being the time at the moment
                        mongodb.document = datetime.now()
                        create_document(mongodb.document, mongodb)
                        first_stream_checker = False

                    time_interval = time.time() - start_time
                    if time_interval > cut_off_time:    # make sure only run when real time is selected
                        start_time = time.time()
                        # Create new document when exceeded 120 seconds, with 'key' being the time at the moment
                        mongodb.document = datetime.now()
                        create_document(mongodb.document, mongodb)

                with dt[0]:
                    im = torch.from_numpy(im).to(model.device)
                    im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
                    im /= 255  # 0 - 255 to 0.0 - 1.0
                    if len(im.shape) == 3:
                        im = im[None]  # expand for batch dim

                # Inference
                with dt[1]:
                    visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
                    pred = model(im, augment=augment, visualize=visualize)

                # NMS
                with dt[2]:
                    pred = pred[0][1] if isinstance(pred[0], list) else pred[0]
                    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

                # Second-stage classifier (optional)
                # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)
                
                #example: 
                # pred is [tensor([[499.17947, 181.21526, 621.33301, 341.38055,   0.91296,   0.00000],
                # [401.35309, 284.58063, 507.15344, 371.85614,   0.83781,   0.00000]], device='cuda:0')]
                # Process every detected object
                for i, det in enumerate(pred):  # per image
                    seen += 1
                    if webcam:  # batch_size >= 1
                        p, im0, frame = path[i], im0s[i].copy(), dataset.count
                        s += f'{i}: '
                    else:
                        p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

                    #..................DRAW SELECTED PRESET LINE....................
                    track_info_obj.draw_line_on_frame(im0)    

                    p = Path(p)  # to Path
                    save_path = str(save_dir / p.name)  # im.jpg
                    txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
                    s += '%gx%g ' % im.shape[2:]  # print string
                    gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                    imc = im0.copy() if save_crop else im0  # for save_crop
                    annotator = Annotator(im0, line_width=line_thickness, example=str(names))

                    #..................ANNOTATE BOUNDING BOX ON DETECTED OBJECT....................
                    if len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                        # Print results
                        for c in det[:, 5].unique():
                            n = (det[:, 5] == c).sum()  # detections per class
                            s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                        # Write results
                        for *xyxy, conf, cls in reversed(det):
                            if save_txt:  # Write to file
                                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                                line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                                with open(f'{txt_path}.txt', 'a') as f:
                                    f.write(('%g ' * len(line)).rstrip() % line + '\n')

                            if save_img or save_crop or view_img:  # Add bbox to image
                                c = int(cls)  # integer class
                                label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                                # annotator.box_label(xyxy, label, color=colors(c, True))
                            if save_crop:
                                save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

                    #..................USE TRACK FUNCTION....................
                    #pass an empty array to sort
                    dets_to_sort = np.empty((0,6))

                    confidence_score = []       # INITIALIZE CONFIDENCE SCORE ARRAY
                    # NOTE: We send in detected object class too
                    for x1,y1,x2,y2,conf,detclass in det.cpu().detach().numpy():
                        dets_to_sort = np.vstack((dets_to_sort, 
                                                    np.array([x1, y1, x2, y2, 
                                                            conf, detclass])))
                        confidence_score.append(conf)           # RETRIEVE CONF FOR DISPLAY IN EACH FRAME 

                    #..................RUN SORT ALGORITHM, therefore the object tracking....................
                    tracked_dets = sort_tracker.update(dets_to_sort)            # MIXTURE OF DATA 
                    tracks = sort_tracker.getTrackers()                      # GET EACH TRACKS IN LIST

                    #loop over tracks                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
                    for track in tracks:
                        # print(track)
                        if color_box:
                            color = compute_color_for_labels(track_color_id)
                            [cv2.line(im0, (int(track.centroidarr[i][0]),int(track.centroidarr[i][1])), 
                                    (int(track.centroidarr[i+1][0]),int(track.centroidarr[i+1][1])),
                                    color, thickness=3) for i,_ in  enumerate(track.centroidarr) 
                                    if i < len(track.centroidarr)-1 ] 
                            track_color_id = track_color_id+1
                        else:
                            """ CONSTRUCT A RECTANGLE ON TRACKED OBJECT"""
                            [cv2.line(im0, (int(track.centroidarr[i][0]),int(track.centroidarr[i][1])), 
                                    (int(track.centroidarr[i+1][0]),int(track.centroidarr[i+1][1])),
                                    (124, 252, 0), thickness=3) for i,_ in  enumerate(track.centroidarr) 
                                    if i < len(track.centroidarr)-1 ] 

                    # Declare None to avoid runtime error when 'if' below is not run
                    identities = None
                    cent = None
                    is_record = False
                    # -------------------------------

                    # -------------------------------
                    # Draw boxes for visualization
                    if len(tracked_dets)>0:
                        # print("content is: ", tracked_dets)
                        bbox_xyxy = tracked_dets[:,:4]
                        identities = tracked_dets[:, 8]
                        categories = tracked_dets[:, 4]
                        box_info = tracked_dets[:, [0, 1, 2, 3, 4, 8]]
                        conf = list(reversed(confidence_score))
                        cent = [(float((x1+x2)/2), float((y1+y2)/2)) for x1, y1, x2, y2 in bbox_xyxy]
                        draw_boxes(im0, bbox_xyxy, identities, categories, names, color_box, conf)

                    #..................PERFORM COUNTING....................
                    if cent is not None:
                        if seen == 1:
                            cent_iden_last_frame = None
                            cent_iden = [[cen, iden] for cen, iden in zip(cent, identities)]
                        else:
                            cent_iden_last_frame = cent_iden
                            cent_iden = [[cen, iden] for cen, iden in zip(cent, identities)]

                        # Update the counting 
                        segments = counter_obj.update_frame(cent_iden_last_frame, cent_iden)

                        # Check intersection on each tracked object
                        # document_time tbc
                        # is_record = counter_obj.check_intersect(segments, im0, track_info_obj, mongodb, document_time)
                        is_record = counter_obj.check_intersect(segments, im0, track_info_obj, mongodb, mongodb.document)
                    else:
                        print("No tracking detected")
                        # Set both to None in first frame
                        if seen == 1:
                            cent_iden_last_frame = None
                            cent_iden = None
                        # Set cent_iden to None as there is no detected object
                        else:
                            cent_iden_last_frame = cent_iden
                            cent_iden = None

                    # -------------------------------

                    #..................VISUALIZE IN STREAMLIT....................
                    if seen % 5 == 1:   # To improve loop processing time, as this takes long time to process (~0.1s)
                        strlit.display_current_visualization(mongodb)

                    im0 = annotator.result()
                    streamlit_im0 = im0.copy()
                    streamlit_im0 = cv2.cvtColor(streamlit_im0, cv2.COLOR_BGR2RGB)

                    #..................DISPLAY MEDIA IN STREAMLIT....................
                    if seen % 5 == 1:   # To improve loop processing time, as this takes long time to process (~0.1s)
                        media_display.image(streamlit_im0, channels="RGB")

                    # view_img=True
                    if view_img:
                        if platform.system() == 'Linux' and p not in windows:
                            # print("Test")
                            windows.append(p)
                            cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                            cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                        resized_image = cv2.resize(im0, (source_info_obj.resized_width, source_info_obj.resized_height))
                        cv2.imshow(str(p), resized_image)
                        cv2.waitKey(1)  # 1 millisecond

                    # Save results (image with detections)
                    if save_img:
                        if dataset.mode == 'image':
                            cv2.imwrite(save_path, im0)
                        else:  # 'video' or 'stream'
                            if vid_path[i] != save_path:  # new video
                                vid_path[i] = save_path
                                if isinstance(vid_writer[i], cv2.VideoWriter):
                                    vid_writer[i].release()  # release previous video writer
                                if vid_cap:  # video
                                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                                else:  # stream
                                    fps, w, h = 30, im0.shape[1], im0.shape[0]
                                save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                                vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                            vid_writer[i].write(im0)
                    
                    # Save information of the decision frame
                    if is_record:
                        frame_folder_path = f"{folder_path}/{frame}"
                        if not os.path.exists(frame_folder_path):
                            os.makedirs(frame_folder_path)

                        cv2.imwrite(f"{frame_folder_path}/curr.jpg", im0)
                        cv2.imwrite(f"{frame_folder_path}/prev.jpg", last_frame_image)
                    last_frame_image = im0
                
                temp_frame += 1
                temp_end_time = time.time()
                temp_time_used = temp_end_time - temp_time
                time_temp += temp_time_used
                # print(f"time used: {temp_time_used}")

                if time_temp >= 1:
                    print(f"Frame processed in one second: {temp_frame}")
                    temp_frame = 0
                    time_temp = 0

                # Print time (inference-only)
                LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

            # Print results
            t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
            LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
            if save_txt or save_img:
                s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
                LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
            if update:
                strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolo.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
