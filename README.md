# 🦺 PPE Safety Checker using YOLOv8

## Overview

PPE Safety Checker is an AI-powered computer vision application developed using **Python** and **YOLOv8** to monitor Personal Protective Equipment (PPE) compliance in construction and industrial environments.

The system detects workers, safety helmets, and safety vests, identifies PPE violations, classifies the inspection status, assigns a risk level, and generates inspection reports automatically.

---

# Objectives

- Detect workers in workplace images.
- Verify the presence of safety helmets and safety vests.
- Identify PPE violations automatically.
- Classify each inspection as SAFE or UNSAFE.
- Assess the associated risk level.
- Generate professional inspection reports.

---

# Features

- 👷 Worker Detection
- ⛑️ Helmet Detection
- 🦺 Safety Vest Detection
- ⚠️ Automatic PPE Violation Detection
- ✅ SAFE / UNSAFE Classification
- 📊 Risk Level Assessment (LOW / MEDIUM / HIGH)
- 🖼️ Save Annotated Detection Images
- 🚩 Save Unsafe Images Separately
- 📄 Generate CSV Inspection Logs
- 🌐 Generate HTML Safety Report

---

# Project Structure

```
PPE_Safety_Checker/
│
├── detect.py
├── README.md
├── requirements.txt
├── safety_report.html
├── violations_log.csv
│
├── images/
├── outputs/
├── flagged_images/
├── models/
└── yolov8n.pt
```

---

# Technologies Used

- Python
- YOLOv8 (Ultralytics)
- OpenCV
- NumPy
- Pandas
- HTML
- CSS

---

# Workflow

1. Load input images.
2. Detect workers using the YOLOv8 model.
3. Detect safety helmets and safety vests.
4. Identify PPE violations.
5. Classify the inspection as SAFE or UNSAFE.
6. Assign a risk level.
7. Save annotated images.
8. Save unsafe images separately.
9. Generate a CSV inspection log.
10. Generate an HTML safety report.

---

# Installation

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the project:

```bash
python detect.py
```

---

# Output

After execution, the system generates:

- Annotated images in the **outputs** folder.
- Unsafe images in the **flagged_images** folder.
- A CSV log containing all inspection results.
- An HTML safety report summarizing the inspection.

---

# Risk Classification

| Violations | Status | Risk Level |
|------------|--------|------------|
| None | SAFE | LOW |
| Missing Helmet | UNSAFE | MEDIUM |
| Missing Safety Vest | UNSAFE | MEDIUM |
| Missing Helmet & Safety Vest | UNSAFE | HIGH |

---

# Applications

This project can be applied in:

- Construction sites
- Industrial facilities
- Manufacturing plants
- Warehouse safety monitoring
- PPE compliance inspections

---

# Future Improvements

- Real-time webcam monitoring
- Video stream processing
- Email alert notifications
- Mobile application integration
- Live analytics dashboard
- Multi-worker tracking

---

# Author

**Developed by**

- Noon Abdelrahman

---

# License

This project was developed as an AI-based Computer Vision project for educational and internship demonstration purposes.

© 2026