import os
import time
import json

from flask import Flask, Response, render_template, jsonify, request, send_file, make_response

from src import main

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")

app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)

# Start detector once
main.start_detector(0)


@app.route("/")
def index():
    return render_template("index.html")


def mjpeg_generator():
    while True:
        frame = main.get_mjpeg_frame()
        if frame:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            )
        else:
            time.sleep(0.05)


@app.route("/video_feed")
def video_feed():
    return Response(mjpeg_generator(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/status")
def status():
    _, state = main.run_frame()
    return jsonify(state)


@app.route("/logs")
def logs():
    logpath = getattr(main, "LOGFILE", os.path.join("logs", "drowsiness.log"))
    lines = []
    if os.path.exists(logpath):
        with open(logpath, "r", encoding="utf-8") as f:
            lines = f.readlines()[-500:]
    cleaned = [l.rstrip("\n") for l in lines]
    return render_template("logs.html", lines=cleaned)


@app.route("/logs_csv")
def logs_csv():
    """
    Export logs as CSV: timestamp,message
    """
    logpath = getattr(main, "LOGFILE", os.path.join("logs", "drowsiness.log"))
    if not os.path.exists(logpath):
        resp = make_response("timestamp,message\n")
        resp.headers["Content-Type"] = "text/csv"
        resp.headers["Content-Disposition"] = "attachment; filename=logs.csv"
        return resp

    rows = ["timestamp,message"]
    with open(logpath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.startswith("[") or "]" not in line:
                continue
            ts, msg = line.split("]", 1)
            ts = ts.strip("[]")
            msg = msg.strip()
            rows.append(f"{ts},{msg.replace(',', ' ')}")

    csv_data = "\n".join(rows)
    resp = make_response(csv_data)
    resp.headers["Content-Type"] = "text/csv"
    resp.headers["Content-Disposition"] = "attachment; filename=drowsiness_logs.csv"
    return resp


@app.route("/calibration", methods=["GET", "POST"])
def calibration():
    if request.method == "POST":
        data = request.get_json(force=True)
        if not data:
            return jsonify({"ok": False, "error": "No JSON body"}), 400
        ok = main.save_calibration(data)
        return jsonify({"ok": ok})

    # GET – show current config
    cal = {
        "ear_threshold": main.SHARED_STATE.get("ear_threshold", main.DEFAULT_EAR_THRESHOLD),
        "pitch0": main.SHARED_STATE.get("pitch0", 0.0),
        "yaw0": main.SHARED_STATE.get("yaw0", 0.0),
    }
    return render_template("settings.html", cal=cal)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)