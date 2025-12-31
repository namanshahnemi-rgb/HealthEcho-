from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import cv2
import dlib
import os
import json
import numpy as np
import face_recognition
from imutils import face_utils
from scipy.spatial import distance as dist

PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
USERS_FILE = "users.json"
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 3
REQUIRED_BLINKS = 2

app = FastAPI()
templates = Jinja2Templates(directory="templates")

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

if os.path.exists(USERS_FILE):
    with open(USERS_FILE, "r") as f:
        users_db = json.load(f)
else:
    users_db = {}

camera = None
camera_active = False
current_mode = None
current_user = None
blink_counter = 0
total_blinks = 0
auth_result = None



def calculate_ear(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)



def generate_frames():
    global camera, camera_active, blink_counter, total_blinks, auth_result

    camera = cv2.VideoCapture(0)

    while camera_active:
        success, frame = camera.read()
        if not success:
            break

        frame = cv2.resize(frame, None, fx=0.75, fy=0.75)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        rects = detector(rgb, 0)

        for rect in rects:
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            left_eye = shape[36:42]
            right_eye = shape[42:48]

            ear = (calculate_ear(left_eye) + calculate_ear(right_eye)) / 2

            if ear < EYE_AR_THRESH:
                blink_counter += 1
            else:
                if blink_counter >= EYE_AR_CONSEC_FRAMES:
                    total_blinks += 1
                blink_counter = 0

            cv2.putText(frame, f"Blinks: {total_blinks}/{REQUIRED_BLINKS}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            if total_blinks >= REQUIRED_BLINKS and not auth_result:
                face_locations = [
                    (rect.top(), rect.right(), rect.bottom(), rect.left())]
                encodings = face_recognition.face_encodings(
                    rgb, face_locations)

                if encodings:
                    encoding = encodings[0]

                    if current_mode == "register":
                        users_db[current_user] = {
                            "encoding": encoding.tolist()}
                        with open(USERS_FILE, "w") as f:
                            json.dump(users_db, f, indent=2)
                        auth_result = f"REGISTER SUCCESS: {current_user}"

                    elif current_mode == "login":
                        known_encodings = [
                            np.array(v["encoding"]) for v in users_db.values()]
                        known_names = list(users_db.keys())

                        distances = face_recognition.face_distance(
                            known_encodings, encoding)
                        idx = np.argmin(distances)

                        auth_result = (
                            f"LOGIN SUCCESS: {known_names[idx]}"
                            if distances[idx] < 0.6 else "ACCESS DENIED"
                        )

               
                    camera_active = False

        cv2.putText(frame, "Blink Twice to Authenticate",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        ret, buffer = cv2.imencode(".jpg", frame)
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" +
               buffer.tobytes() + b"\r\n")

    if camera:
        camera.release()




@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})


@app.get("/video")
def video():
    if not camera_active:
        return JSONResponse({"error": "Camera not active"}, status_code=400)

    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.post("/register")
def register(username: str):
    global current_mode, current_user, total_blinks, auth_result, camera_active
    if username in users_db:
        return JSONResponse({"error": "User already exists"}, status_code=400)

    current_mode = "register"
    current_user = username
    total_blinks = 0
    auth_result = None
    camera_active = True
    return {"status": "Camera started for registration"}


@app.post("/login")
def login():
    global current_mode, total_blinks, auth_result, camera_active
    current_mode = "login"
    total_blinks = 0
    auth_result = None
    camera_active = True
    return {"status": "Camera started for login"}


@app.get("/result")
def result():
    return {"result": auth_result}
