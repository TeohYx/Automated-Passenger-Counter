import pytesseract as tess
from PIL import Image
import os
import glob
import json
from datetime import datetime, timedelta
import time

script_directory = os.path.dirname(os.path.realpath(__file__))
png_file = glob.glob(os.path.join(script_directory, '*.png'))

def image_to_txt(png):
    # print(png)
    text = tess.image_to_string(png)
    # print(text)
    return text

def main():
    text = image_to_txt(png_file[0])

    print(type(text))
    date = text.split()[1]
    timen = text.split()[2]

    datet = f"{date} {timen}"
    # print(datet)

    datetime_datet = datetime.strptime(datet, "%Y-%m-%d %H:%M:%S") 
    document_time = datetime_datet.strftime("%Y%m%d_%H%M%S")
    print(document_time)

    fut = datetime_datet + timedelta(minutes=60, seconds=59)

    datetime_datet = datetime.now()
    print(datetime_datet)
    fut = datetime_datet + timedelta(minutes=60, seconds=59)

    # is_hour = timedelta(hours=1) > (fut - datetime_datet)
    # print(is_hour)

if __name__== "__main__":
    main()