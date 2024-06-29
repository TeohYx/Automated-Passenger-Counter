from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
import cv2
# import yt_dlp

url = 'rtsp://localhost:8554/'
# source = check_file(url)  
# print(source)
# # Use yt-dlp to extract the best stream URL
# ydl_opts = {
#     'format': 'best',
#     'quiet': True,
# }

# with yt_dlp.YoutubeDL(ydl_opts) as ydl:
#     info_dict = ydl.extract_info(url, download=False)
#     print(info_dict)
#     stream_url = info_dict['url']

# print(stream_url)
# Open the stream URL with OpenCV
cap = cv2.VideoCapture(url)
print(cap)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    cv2.imshow('Live Stream', frame)
    print("here2")
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows() 