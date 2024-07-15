# Automated Passenger Counter

## Prerequisite
1. Python 3.11 (recommended)

## Setup
To use the project, first clone it in a folder:

```bash
git clone https://github.com/TeohYx/Automated-Passenger-Counter.git
```

Go to the folder directory
```bash
cd Automated-Passenger-Counter
```

Create the virtual environment, so that the library to be installed is isolated from the local computer:
```bash
python -m venv .venv
```

Then, activate the virtual environment:
```bash
.venv\Scripts\activate
```

Install all the dependencies from requirements.txt
```bash
pip install -r requirements.txt
```

When success, run the application by inputting the following Streamlit command:
```bash
streamlit run main.py
```

## Application Inputs
The application takes two type of input:
1. Recorded Video - Provide a video file in a supported format.
2. Real-time Streaming - Input the RTSP link of the streaming feed.

## Description
The Automated Passenger Counter application processes video feeds to count passengers. Users can choose between recorded videos and real-time streaming by providing the appropriate input.

## Example Interface
![Counting Interface](pic/Interface1.png)

![Visualization](pic/Visual.png)