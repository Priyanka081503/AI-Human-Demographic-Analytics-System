import cv2
import time
import psycopg2
import threading
import numpy as np
import random
import string
from datetime import datetime
from ultralytics import YOLO
from insightface.app import FaceAnalysis
from collections import defaultdict, deque


SOURCE_URL = "rtsp://admin:password@IP/Streaming/Channels/102"

DB_PARAMS = {
    "host": "host",
    "database": "DB name",
    "user": "postgres",
    "password": "password",
    "port": "5432"
}

model = YOLO("yolo11n.pt")

print("Loading InsightFace...")
app = FaceAnalysis(providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(320, 320))


id_mapping = {}
entry_time = {}
crossed_ingress = set()
crossed_egress = set()
last_positions = {}

gender_history = defaultdict(lambda: deque(maxlen=15))
final_gender = {}
gender_saved = set()   

frame_counter = 0

def generate_id(length=20):
    chars = string.ascii_letters + string.digits
    return ''.join(random.choice(chars) for _ in range(length))


def log_to_db(unique_id, gender=None, ingress=0, egress=0, dwell_time=0.0):
    try:
        conn = psycopg2.connect(**DB_PARAMS)
        cur = conn.cursor()

        cur.execute("""
            INSERT INTO table
            (capture_time, location, unique_id, gender, ingress_count, egress_count, dwell_time)
            VALUES (%s, %s, %s, %s, %s, %s, %s)

            ON CONFLICT (unique_id)
            DO UPDATE SET
                gender = COALESCE(EXCLUDED.gender, public.demographics.gender),
                ingress_count = public.demographics.ingress_count + EXCLUDED.ingress_count,
                egress_count = public.demographics.egress_count + EXCLUDED.egress_count,
                dwell_time = CASE
                                WHEN EXCLUDED.dwell_time > 0
                                THEN EXCLUDED.dwell_time
                                ELSE public.demographics.dwell_time
                             END,
                capture_time = EXCLUDED.capture_time;
        """, (
            datetime.now(),
            "Main Entrance",
            unique_id,
            gender,
            ingress,
            egress,
            dwell_time
        ))

        conn.commit()
        cur.close()
        conn.close()

        print("DB Updated:", unique_id, "Gender:", gender)

    except Exception as e:
        print("DB Error:", e)


class VideoStream:
    def __init__(self, src):
        self.cap = cv2.VideoCapture(src)
        self.ret, self.frame = self.cap.read()
        self.stopped = False

    def start(self):
        threading.Thread(target=self.update, daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if ret:
                self.ret, self.frame = ret, frame

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True

vs = VideoStream(SOURCE_URL).start()
time.sleep(2)

line_x = 400

while True:

    frame = vs.read()
    if frame is None:
        continue

    frame_counter += 1

    frame = cv2.resize(frame, (800, 600))
    cv2.line(frame, (line_x, 0), (line_x, 600), (0, 255, 255), 2)

    results = model.track(
        frame,
        persist=True,
        classes=[0],
        tracker="botsort.yaml",
        conf=0.4,
        verbose=False
    )

    if results[0].boxes.id is not None:

        boxes = results[0].boxes.xyxy.int().cpu().tolist()
        ids = results[0].boxes.id.int().cpu().tolist()

        for box, track_id in zip(boxes, ids):

            x1, y1, x2, y2 = box
            cx = int((x1 + x2) / 2)

            if track_id not in id_mapping:
                id_mapping[track_id] = generate_id()

            unique_id = id_mapping[track_id]

        

            if unique_id not in final_gender and frame_counter % 3 == 0:

                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:

                    face_input = cv2.resize(crop, (320, 320))
                    faces = app.get(face_input)

                    if len(faces) > 0:
                        gender_val = faces[0].gender
                        gender_history[unique_id].append(gender_val)

                        avg = sum(gender_history[unique_id]) / len(gender_history[unique_id])

                        if avg > 0.6:
                            final_gender[unique_id] = "Male"
                        elif avg < 0.4:
                            final_gender[unique_id] = "Female"

        
            if unique_id in final_gender and unique_id not in gender_saved:
                log_to_db(
                    unique_id,
                    gender=final_gender[unique_id],
                    ingress=0,
                    egress=0,
                    dwell_time=0.0
                )
                gender_saved.add(unique_id)

            gender_display = final_gender.get(unique_id, "Detecting")

            

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame,
                        f"{unique_id[:6]} | {gender_display}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2)

        

            if cx > line_x and unique_id not in crossed_ingress:
                entry_time[unique_id] = time.time()

                log_to_db(
                    unique_id,
                    gender=final_gender.get(unique_id),
                    ingress=1,
                    egress=0,
                    dwell_time=0.0
                )

                crossed_ingress.add(unique_id)

        

            if unique_id in last_positions:
                prev_x = last_positions[unique_id]

                if prev_x >= line_x and cx < line_x:
                    if unique_id in entry_time and unique_id not in crossed_egress:

                        dwell_minutes = round(
                            (time.time() - entry_time[unique_id]) / 60, 4
                        )

                        log_to_db(
                            unique_id,
                            gender=final_gender.get(unique_id),
                            ingress=0,
                            egress=1,
                            dwell_time=dwell_minutes
                        )

                        crossed_egress.add(unique_id)

            last_positions[unique_id] = cx

    cv2.imshow("Final Demographic Tracker", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vs.stop()
cv2.destroyAllWindows()