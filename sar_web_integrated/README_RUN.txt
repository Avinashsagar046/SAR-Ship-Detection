SAR SHIP DETECTOR — UPDATED WEB INTEGRATION

This package contains the updated SAR ship detection web interface and Flask backend.

1. Recommended layout inside D:\SAR_SHIP:
   web_app\
     app.py
     requirements.txt
     templates\index.html
     static\css\style.css
     static\js\script.js
     outputs\

2. Put app.py/templates/static/outputs under D:\SAR_SHIP\web_app.

3. Activate the existing project environment:
   cd D:\SAR_SHIP
   .venv\Scripts\activate

4. Install missing web dependencies if required:
   pip install -r web_app\requirements.txt

5. Start:
   python web_app\app.py

6. Open:
   http://127.0.0.1:5000

FINAL DETECTOR SETTINGS
- CA-CFAR: 21x21 window, guard 5x5, PFA 0.001, minimum connected area 10
- OS-CFAR: 21x21 window, guard 5x5, PFA 0.001, 80% ordered-statistics rank, minimum area 20
- YOLO11n: best.pt, image size 800, confidence 0.40 for the common evaluation operating point

METRIC DEFINITIONS SHOWN IN THE DASHBOARD
- Held-out YOLO11n test metrics: Precision 0.902, Recall 0.797, mAP@0.5 0.894, mAP@0.5:0.95 0.640
- Common evaluation at confidence 0.40: Precision 0.891, Recall 0.804, F1-score 0.846
- The two sets are kept distinct because they use different evaluation procedures.
- The HRSID held-out test set contains 1,962 images and 5,922 ship instances.

IMPORTANT
- The displayed image enhancement is visualization-only and is not fed into the evaluated CFAR or YOLO11n detectors.
- Image-level TP/FP/FN values are shown only when the uploaded filename exactly matches an HRSID test image with available ground truth.
