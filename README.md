# 🦺 PPE Safety Checker using YOLOv8

## Overview

PPE Safety Checker is an AI-powered computer vision application developed using **Python** and **YOLOv8** to monitor Personal Protective Equipment (PPE) compliance in construction and industrial environments.

The system detects workers, safety helmets, and safety vests, identifies PPE violations, classifies the inspection status, assigns a risk level, and automatically generates inspection reports.

---
# VisionGuard AI - PPE Safety Checker

AI-powered PPE detection workflow using a Vision-Language Model.

## 🎥 Demo Video

Watch the demo here:

https://udksa-my.sharepoint.com/:v:/g/personal/2240009097_iau_edu_sa/IQDHhadkB3FAQat-UzIJqvvcARwAdJrzyCe1RjliRUZf2xE?e=uphMaB

# Objectives

* Detect workers in workplace images.
* Verify the presence of safety helmets and safety vests.
* Automatically identify PPE violations.
* Classify each inspection as **SAFE** or **UNSAFE**.
* Assess the associated risk level.
* Generate professional inspection reports.

---

# Features

* 👷 Worker Detection
* ⛑️ Helmet Detection
* 🦺 Safety Vest Detection
* ⚠️ Automatic PPE Violation Detection
* ✅ SAFE / UNSAFE Classification
* 📊 Risk Level Assessment (LOW / MEDIUM / HIGH)
* 🖼️ Save Annotated Detection Images
* 🚩 Save Unsafe Images Separately
* 📄 Generate CSV Inspection Logs
* 🌐 Generate HTML Safety Report

---

# Project Structure

```
PPE_Safety_Checker/
│
├── detect.py
├── README.md
├── requirements.txt
├── .gitignore
├── safety_report.html
├── violations_log.csv
├── yolov8n.pt
│
├── images/
├── outputs/
└── flagged_images/
```

---

# Technologies Used

* Python
* YOLOv8 (Ultralytics)
* OpenCV
* NumPy
* Pandas
* HTML
* CSS

---

# Workflow

1. Load input images.
2. Detect workers using YOLOv8.
3. Detect helmets and safety vests.
4. Identify PPE violations.
5. Classify the inspection as SAFE or UNSAFE.
6. Assign a risk level.
7. Save annotated images.
8. Save flagged images.
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

# Sample Data

The project was tested using sample construction site images containing workers with and without the required Personal Protective Equipment (PPE), including helmets and safety vests.

The sample images were used to verify worker detection, PPE compliance, risk classification, and report generation.

---

# Output

After execution, the system generates:

* Annotated detection images inside the **outputs** folder.
* Unsafe images inside the **flagged_images** folder.
* A CSV inspection log (**violations_log.csv**).
* A professional HTML report (**safety_report.html**).

---

# Risk Classification

| Violations                   | Status | Risk Level |
| ---------------------------- | ------ | ---------- |
| None                         | SAFE   | LOW        |
| Missing Helmet               | UNSAFE | MEDIUM     |
| Missing Safety Vest          | UNSAFE | MEDIUM     |
| Missing Helmet & Safety Vest | UNSAFE | HIGH       |

---

# Applications

This project can be applied in:

* Construction sites
* Industrial facilities
* Manufacturing plants
* Warehouse safety monitoring
* PPE compliance inspections

---

# AI Usage

AI tools (ChatGPT) were used throughout the development process to assist with planning, implementation, debugging, and documentation.

### AI Assisted With

* Planning the project workflow.
* Suggesting the initial code structure.
* Assisting with parts of the detection logic.
* Improving the HTML report layout.
* Writing and refining the README.
* Debugging implementation issues.

### My Contributions

* Built and tested the complete PPE detection workflow.
* Integrated worker, helmet, and safety vest detection.
* Implemented SAFE / UNSAFE classification.
* Added LOW / MEDIUM / HIGH risk classification.
* Generated CSV inspection logs.
* Created the HTML safety report.
* Saved unsafe images separately.
* Tested the project using multiple sample images.
* Fixed runtime errors and verified the complete pipeline.

### Challenges Solved

* Handling PPE violation detection correctly.
* Organizing project outputs.
* Improving report readability.
* Testing and validating generated results.

### What I Learned

Through this project I gained practical experience in:

* Computer Vision using YOLOv8.
* Image processing with OpenCV.
* AI-assisted software development.
* Debugging Python applications.
* Building complete end-to-end AI prototypes.

---

# Future Improvements

* Real-time webcam monitoring.
* Video stream processing.
* Email alert notifications.
* Mobile application integration.
* Live analytics dashboard.
* Multi-worker tracking.

---

# Author

**Developed by**

* Noon Abdelrahman

---

# License

This project was developed as an AI-based Computer Vision prototype for educational and internship demonstration purposes.

© 2026
