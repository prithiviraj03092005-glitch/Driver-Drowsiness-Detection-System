# Driver Drowsiness Detection System

A real-time hybrid monitoring system that uses computer vision to detect driver drowsiness through Eye Aspect Ratio (EAR), texture analysis (CNN), and head pose orientation.

## 🚀 Features
- **Hybrid Detection**: Combines geometric EAR benchmarks with a deep learning eye status classifier.
- **Auto-Calibration**: Automatically calculates baseline EAR and head orientation during the first 2-3 seconds of operation.
- **Fail-Safe Mechanism**: Automatically falls back to EAR-based detection if the CNN model is unavailable.
- **Real-time Dashboard**: Web-based monitoring UI with live video feed, telemetry charts, and analytics.
- **Audible Alerts**: Integrated alarm system to warn drivers upon detecting prolonged eye closure.

## 🛠 Tech Stack
- **Backend**: Python, Flask, OpenCV
- **AI Frameworks**: TensorFlow (Keras), Mediapipe
- **Frontend**: HTML5, Vanilla CSS, Chart.js

## 📦 Installation

1. **Clone the repository**:
   ```bash
   git clone <repo-url>
   cd DrowsinessProject
   ```

2. **Set up virtual environment**:
   ```bash
   python -m venv driveenv
   source driveenv/bin/activate  # Windows: driveenv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 🚦 Usage

### Running the Dashboard
Start the Flask application:
```bash
python app.py
```
Open your browser and navigate to `http://localhost:5000`.

### Calibration
When the system starts, it will enter a 50-frame calibration window. **Look straight at the camera with eyes open** during this time to establish baseline values for:
- EAR (Normal open eye benchmark)
- Head Pitch/Yaw (Center orientation)

## 📁 Project Structure
- `app.py`: Main Flask application server.
- `src/main.py`: Core detection engine (Mediapipe + CNN).
- `src/alarm.py`: Sound management system.
- `src/calibration.py`: CLI-based calibration utility.
- `templates/`: HTML dashboard templates.
- `model/`: Pre-trained weight files for eye classification.

## ⚠️ Important Notes
- Ensure your camera is well-lit for optimal landmark detection.
- The system defaults to Camera Index 0. You can change this in `main.start_detector(0)` within `app.py`.
