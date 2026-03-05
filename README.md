# AI-Human-Demographic-Analytics-System

A **real-time computer vision system** that detects people from RTSP camera streams and performs **demographic analytics**, including **gender detection, ingress/egress counting, and dwell time tracking**.
The system uses **YOLO for human detection**, **InsightFace for gender analysis**, and stores analytics data in a **PostgreSQL database**.

---

## 📌 Project Overview

This project is designed for **retail stores, malls, offices, and smart surveillance systems** where understanding **customer movement and demographics** is important.

The system processes **live RTSP camera streams**, detects humans, tracks them across frames, and analyzes their demographic information.

It also calculates:

* Entry count (Ingress)
* Exit count (Egress)
* Time spent inside the monitored area (Dwell Time)

All analytics data is stored in a **PostgreSQL database** for further analysis.

---

## 🖥 System Architecture

```
RTSP Camera Stream
        │
        ▼
Video Capture
        │
        ▼
Human Detection
        │
        ▼
Multi-Object Tracking
        │
        ▼
Face Analysis
        │
        ▼
Gender Prediction
        │
        ▼
Line Crossing Logic
        │
        ├── Ingress Count
        ├── Egress Count
        └── Dwell Time
        │
        ▼
PostgreSQL Database Storage
```

###  Install dependencies

```
pip install -r requirements.txt
```

## 🗄 Database Configuration

The system logs analytics data into **PostgreSQL**.

---

## 🎯 Use Cases

This system can be used in:

* Retail analytics
* Shopping malls
* Smart buildings
* Office entry monitoring
* Crowd analytics
* Security systems

---

## 👩‍💻 Author

**Priyanka**

Background: Electrical and Electronics Engineering
Skills: Python, SQL, Computer Vision, Web Development

---

## 📜 License

This project is open-source and available under the **MIT License**.

---

