# HealthEcho (FastAPI)

HealthEcho is a Python-based application built with FastAPI that demonstrates AI-powered health and nutrition features. The system supports image uploads, food and object identification, basic face detection, and voice feedback using ElevenLabs. The project is designed as a functional prototype showcasing AI integration rather than a production-ready medical or clinical system.

---

## Features

- FastAPI backend
- Runs locally at `127.0.0.1:8000`
- Image upload support
- Face detection using `face-recognition`
- Object and food detection using YOLO (Ultralytics)
- Image processing with OpenCV
- Voice response generation using ElevenLabs
- CORS enabled for development

---

## Project Structure

```
healthecho/
│
├── main.py
├── README.md
├── requirements.txt
│
├── templates/
│   └── index.html
│
├── static/
│
└── uploads/
```

---

## Python Version

Recommended:

```
Python 3.9 – 3.11
```

Check version:

```bash
python --version
```

---

## Create a Virtual Environment

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

---

## Install Dependencies

Create `requirements.txt`:

```txt
fastapi
uvicorn
jinja2
python-multipart
opencv-python
numpy==1.26.4
face-recognition
ultralytics
elevenlabs
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Face Recognition System Requirements

### Ubuntu / Linux

```bash
sudo apt update
sudo apt install cmake build-essential
```

### Windows

- Install Visual Studio Build Tools
- Enable C++ Build Tools during installation

---

## ElevenLabs API Key Setup

### Linux / macOS

```bash
export ELEVENLABS_API_KEY="your_api_key_here"
```

### Windows (CMD)

```cmd
set ELEVENLABS_API_KEY=your_api_key_here
```

---

## Run the Application

```bash
uvicorn main:app --reload
```

Open in browser:

```
http://127.0.0.1:8000
```

---

## API Endpoints

### Home Page

```
GET /
```

Serves the web interface.

---

### Health Check

```
GET /health
```

Example response:

```json
{
  "status": "OK",
  "time": "2026-01-01T10:00:00"
}
```

---

### Upload Image

```
POST /upload-image
```

Form-data:

- `file` → Image file (jpg or png)

Example response:

```json
{
  "filename": "image123.jpg",
  "faces_detected": 1,
  "yolo_objects": 2
}
```

---

## YOLO Model

Default model:

```
yolov8n.pt
```

To use a custom model:

```python
YOLO("your_custom_model.pt")
```

---

## ElevenLabs Voice Generation Example

```python
from elevenlabs import generate, save

audio = generate(
    text="Your nutrition analysis is complete",
    voice="Rachel",
    model="eleven_multilingual_v2"
)

save(audio, "nutrition_feedback.mp3")
```

---

## Notes

- Uploaded images are stored in the `uploads` directory
- YOLO models are downloaded automatically on first run
- CORS is enabled for development purposes
- API documentation is available at:

```
http://127.0.0.1:8000/docs
```

---

## Future Improvements

- Improved food identification and portion estimation
- Nutrition summarization based on detected food items
- Live camera input support
- User authentication and profiles
- Docker containerization
- Cloud deployment

---

## Author

Devarsh Mistry  
HealthEcho
