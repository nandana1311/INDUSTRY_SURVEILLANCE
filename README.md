# 🏗️ Industrial Surveillance System 

### AI-Based Construction Site Safety Monitoring System

An AI-powered construction site safety monitoring system developed using
Computer Vision and Deep Learning. The system detects Personal
Protective Equipment (PPE) violations, fire, smoke, and potential
worker--machinery proximity hazards to support construction site safety
monitoring.

It combines YOLO-based object detection with a Flask web application to
process images, videos, and live webcam feeds, providing visual
detection results, safety statistics, and alert history.

------------------------------------------------------------------------

## 🚀 Key Features

### 1. PPE Detection and Compliance Monitoring

-   Detects workers and PPE equipment using YOLO.
-   Identifies Hardhat, Mask, and Safety Vest.
-   Detects missing PPE using dedicated violation classes.
-   Evaluates PPE compliance individually for each detected person.
-   Classifies worker safety status into:
    -   **Compliant:** All required PPE detected.
    -   **Partial:** One or more required PPE items missing.
    -   **No PPE:** All required PPE items missing.

### 2. Fire and Smoke Detection

-   Uses a separate YOLO model for fire and smoke detection.
-   Runs fire/smoke inference alongside PPE detection.
-   Highlights detected fire and smoke using bounding boxes.
-   Generates severity-based safety alerts.

### 3. Worker--Machinery Proximity Analysis

-   Identifies machinery and vehicles as potential proximity hazards.
-   Evaluates proximity for workers who are not fully PPE-compliant.
-   Calculates centre-to-centre pixel distance between workers and
    detected hazards.
-   Assigns alert severity based on configurable distance thresholds.

  Distance           Severity
  ------------------ --------------------
  Below 250 pixels   CRITICAL
  250--450 pixels    HIGH
  Above 450 pixels   No proximity alert

*Note: These thresholds are image pixel distances, not actual physical
measurements.*

### 4. Alert Management

-   Generates alerts for PPE violations, fire, smoke, and proximity
    hazards.
-   Supports multiple alert severity levels: Critical, High, Medium, and
    Low.
-   Implements an alert cooldown mechanism to reduce repetitive alerts.
-   Maintains alert history.
-   Saves alert records in JSON format.
-   Supports optional audio alerts in standalone mode.

### 5. Flask Web Dashboard

-   Image upload and processing.
-   Video upload and processing.
-   Live webcam monitoring.
-   Processing progress and status tracking.
-   Detection statistics and safety indicators.
-   Alert history.
-   Processed image and video preview.

------------------------------------------------------------------------

## 🛠️ Technologies Used

  Technology              Purpose
  ----------------------- -------------------------------
  Python                  Core programming
  YOLO (Ultralytics)      Object detection
  OpenCV                  Image and video processing
  PyTorch                 Deep learning framework
  NumPy                   Numerical operations
  Flask                   Web application and backend
  Flask-CORS              Cross-origin resource sharing
  Python Threading        Concurrent model inference
  HTML, CSS, JavaScript   Web interface

------------------------------------------------------------------------

## 🧠 System Architecture

The system integrates two independent YOLO detection models.

**PPE Detection Model** - Person - Hardhat - Mask - Safety Vest -
NO-Hardhat - NO-Mask - NO-Safety Vest - Machinery - Vehicle - Safety
Cone

**Fire and Smoke Detection Model** - Fire - Smoke

### Processing Workflow

1.  Capture input from an image, video, or webcam.
2.  Perform PPE and fire/smoke detection concurrently using separate
    threads.
3.  Associate detected PPE items and violations with individual persons.
4.  Determine the PPE compliance status of each worker.
5.  Analyse proximity between non-compliant workers and nearby machinery
    or vehicles.
6.  Generate severity-based alerts.
7.  Annotate the output with bounding boxes, safety indicators, and
    detection information.
8.  Display results through the web dashboard or standalone application.

------------------------------------------------------------------------

## 📂 Project Structure

``` text
INDUSTRY_SURVEILLANCE/
│
├── app.py
├── surveillance.py
│
├── models/
│   ├── best.pt
│   └── fire_smoke_detection.pt
│
├── templates/
│   ├── index.html
│   ├── history.html
│   └── webcam.html
│
├── static/
│
├── uploads/
├── outputs/
│
├── requirements.txt
└── README.md
```

*The model weights and generated output folders may not be included in
the public repository.*

------------------------------------------------------------------------

## ⚙️ Installation and Setup

### 1. Clone the repository

``` bash
git clone https://github.com/nandana1311/INDUSTRY_SURVEILLANCE.git
```

``` bash
cd INDUSTRY_SURVEILLANCE
```

### 2. Create a virtual environment

``` bash
python -m venv venv
```

Activate the environment:

**Windows**

``` bash
venv\Scripts\activate
```

**Linux / macOS**

``` bash
source venv/bin/activate
```

### 3. Install dependencies

Install the required Python libraries:

``` bash
pip install flask flask-cors opencv-python numpy ultralytics torch
```

For optional audio functionality:

``` bash
pip install pygame
```

### 4. Add trained model weights

Place the trained YOLO model files inside the `models` directory.

``` text
models/
├── best.pt
└── fire_smoke_detection.pt
```

The model weights are required to run the complete detection system.

### 5. Run the application

``` bash
python app.py
```

Open the application in your browser:

**http://127.0.0.1:5001**

------------------------------------------------------------------------

## 🖥️ Standalone Execution

The surveillance module can also run independently without the Flask
dashboard.

**Image processing**

``` bash
python surveillance.py --ppe-model models/best.pt --fire-model models/fire_smoke_detection.pt --image path/to/image.jpg
```

**Video processing**

``` bash
python surveillance.py --ppe-model models/best.pt --fire-model models/fire_smoke_detection.pt --video path/to/video.mp4 --output output.mp4
```

**Live webcam monitoring**

``` bash
python surveillance.py --ppe-model models/best.pt --fire-model models/fire_smoke_detection.pt --camera 0
```

------------------------------------------------------------------------

## 🔧 Configurable Parameters

The system provides configurable parameters for:

-   PPE and fire detection confidence thresholds.
-   Intersection over Union (IoU).
-   Maximum detection limits.
-   PPE-to-person association.
-   Worker--machinery proximity thresholds.
-   Alert cooldown duration.
-   CPU/GPU device selection.

The application automatically selects CUDA when available and otherwise
uses CPU processing.

------------------------------------------------------------------------

## ⚠️ Limitations

-   Proximity analysis uses pixel-based distances and does not represent
    calibrated real-world distances.
-   Detection performance depends on lighting, camera quality, object
    visibility, and model accuracy.
-   Processing speed depends on available hardware and input resolution.
-   Trained model weights are required for detection.
-   The system is intended to support safety monitoring and does not
    replace human supervision or established safety procedures.

------------------------------------------------------------------------

## 🔮 Future Enhancements

-   Camera calibration for improved real-world proximity estimation.
-   Improved detection under challenging lighting conditions.
-   Additional construction-site hazard detection.
-   Enhanced safety reporting and monitoring capabilities.

------------------------------------------------------------------------

## 👩‍💻 Author

**Nandana S. Nair**

AI & Machine Learning \| Computer Vision \| Python

GitHub: [nandana1311](https://github.com/nandana1311)

------------------------------------------------------------------------

*An AI-based construction site safety monitoring project focused on PPE
compliance and hazard detection.*
