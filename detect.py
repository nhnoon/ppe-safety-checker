from ultralytics import YOLO
import cv2
import os
import shutil
import pandas as pd
from datetime import datetime
import webbrowser

MODEL_PATH = "runs/detect/train/weights/best.pt"
IMAGE_FOLDER = "images"
OUTPUT_FOLDER = "outputs"
FLAGGED_FOLDER = "flagged_images"
LOG_FILE = "violations_log.csv"
HTML_REPORT = "safety_report.html"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(FLAGGED_FOLDER, exist_ok=True)

for folder in [OUTPUT_FOLDER, FLAGGED_FOLDER]:
    for file_name in os.listdir(folder):
        file_path = os.path.join(folder, file_name)
        if os.path.isfile(file_path):
            os.remove(file_path)

for file_path in [LOG_FILE, HTML_REPORT]:
    if os.path.exists(file_path):
        os.remove(file_path)

model = YOLO(MODEL_PATH)

HELMET_LABELS = {"helmet", "hardhat", "hat"}
VEST_LABELS = {"safety vest", "vest"}
HUMAN_LABELS = {"human", "person", "worker"}

logs = []


def normalize_label(label):
    return label.strip().lower()


def classify_image(detected_labels):
    detected_set = set(detected_labels)

    has_human = any(label in HUMAN_LABELS for label in detected_set)
    has_helmet = any(label in HELMET_LABELS for label in detected_set)
    has_vest = any(label in VEST_LABELS for label in detected_set)

    violations = []

    if has_human:
        if not has_helmet:
            violations.append("Missing Helmet")
        if not has_vest:
            violations.append("Missing Safety Vest")
    else:
        violations.append("No Worker Detected")

    if not violations:
        status = "SAFE"
        risk = "LOW"
    elif "No Worker Detected" in violations:
        status = "CHECK"
        risk = "UNKNOWN"
    elif len(violations) >= 2:
        status = "UNSAFE"
        risk = "HIGH"
    else:
        status = "UNSAFE"
        risk = "MEDIUM"

    return has_human, has_helmet, has_vest, violations, status, risk


def get_safety_score(status, violations):
    if status == "SAFE":
        return 100
    if "No Worker Detected" in violations:
        return 0
    if len(violations) == 1:
        return 60
    return 20


def get_recommendation(violations):
    recommendations = []

    if "Missing Helmet" in violations:
        recommendations.append("Provide a certified hard hat before entering the site.")

    if "Missing Safety Vest" in violations:
        recommendations.append("Wear a reflective safety vest for visibility and compliance.")

    if "No Worker Detected" in violations:
        recommendations.append("Review the image manually because no worker was detected.")

    if not recommendations:
        recommendations.append("No action required. PPE compliance looks good.")

    return " | ".join(recommendations)


def draw_status(image_path, output_path, status, risk, violations, safety_score):
    image = cv2.imread(image_path)

    if image is None:
        return

    if status == "SAFE":
        title = "SAFE - PPE OK"
        color = (0, 180, 0)
    elif status == "CHECK":
        title = "CHECK - REVIEW IMAGE"
        color = (0, 165, 255)
    else:
        title = f"UNSAFE - {risk} RISK"
        color = (0, 0, 255)

    cv2.rectangle(image, (20, 20), (850, 135), (0, 0, 0), -1)
    cv2.putText(image, title, (35, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3)
    cv2.putText(image, f"Safety Score: {safety_score}%", (35, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)

    y = 170
    for violation in violations:
        cv2.putText(image, f"- {violation}", (35, y), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 2)
        y += 38

    cv2.imwrite(output_path, image)


def create_html_report(logs, summary):
    status_badge = {
        "SAFE": "safe",
        "UNSAFE": "unsafe",
        "CHECK": "check",
    }

    rows = ""
    for item in logs:
        image_link = item["Output Image"]
        badge_class = status_badge.get(item["Status"], "check")

        rows += f"""
        <tr>
            <td>{item["Image"]}</td>
            <td><span class="badge {badge_class}">{item["Status"]}</span></td>
            <td>{item["Risk Level"]}</td>
            <td>{item["Safety Score"]}%</td>
            <td>{item["Violations"]}</td>
            <td>{item["Recommendation"]}</td>
            <td>{item["Worker Detected"]}</td>
            <td>{item["Helmet Detected"]}</td>
            <td>{item["Vest Detected"]}</td>
            <td><a href="{image_link}" target="_blank">View Output</a></td>
            <td>{item["Time"]}</td>
        </tr>
        """

    html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>NEXTA Smart PPE Safety Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            background: #f4f6f8;
            margin: 0;
            padding: 30px;
            color: #1f2937;
        }}
        .container {{
            max-width: 1400px;
            margin: auto;
        }}
        h1 {{
            margin-bottom: 5px;
            color: #111827;
        }}
        .subtitle {{
            color: #6b7280;
            margin-bottom: 25px;
        }}
        .cards {{
            display: grid;
            grid-template-columns: repeat(5, 1fr);
            gap: 16px;
            margin-bottom: 28px;
        }}
        .card {{
            background: white;
            border-radius: 14px;
            padding: 20px;
            box-shadow: 0 4px 14px rgba(0,0,0,0.08);
        }}
        .card h2 {{
            margin: 0;
            font-size: 30px;
        }}
        .card p {{
            margin: 8px 0 0;
            color: #6b7280;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background: white;
            border-radius: 14px;
            overflow: hidden;
            box-shadow: 0 4px 14px rgba(0,0,0,0.08);
        }}
        th, td {{
            padding: 12px;
            border-bottom: 1px solid #e5e7eb;
            text-align: left;
            font-size: 13px;
            vertical-align: top;
        }}
        th {{
            background: #111827;
            color: white;
        }}
        .badge {{
            padding: 6px 10px;
            border-radius: 999px;
            font-weight: bold;
            color: white;
            font-size: 12px;
        }}
        .safe {{
            background: #16a34a;
        }}
        .unsafe {{
            background: #dc2626;
        }}
        .check {{
            background: #f59e0b;
        }}
        a {{
            color: #2563eb;
            font-weight: bold;
            text-decoration: none;
        }}
        .footer {{
            margin-top: 20px;
            color: #6b7280;
            font-size: 13px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>NEXTA Smart PPE Safety Inspector</h1>
        <div class="subtitle">
            Automatically generated PPE inspection report using a fine-tuned YOLOv8 model.
        </div>

        <div class="cards">
            <div class="card">
                <h2>{summary["total_images"]}</h2>
                <p>Total Images</p>
            </div>
            <div class="card">
                <h2>{summary["safe_images"]}</h2>
                <p>Safe Images</p>
            </div>
            <div class="card">
                <h2>{summary["unsafe_images"]}</h2>
                <p>Unsafe / Check Images</p>
            </div>
            <div class="card">
                <h2>{summary["total_violations"]}</h2>
                <p>Total Violations</p>
            </div>
            <div class="card">
                <h2>{summary["compliance_rate"]}%</h2>
                <p>Compliance Rate</p>
            </div>
        </div>

        <table>
            <thead>
                <tr>
                    <th>Image</th>
                    <th>Status</th>
                    <th>Risk</th>
                    <th>Safety Score</th>
                    <th>Violations</th>
                    <th>Recommendation</th>
                    <th>Worker</th>
                    <th>Helmet</th>
                    <th>Vest</th>
                    <th>Output</th>
                    <th>Time</th>
                </tr>
            </thead>
            <tbody>
                {rows}
            </tbody>
        </table>

        <div class="footer">
            Generated by NEXTA Smart PPE Safety Inspector prototype.
        </div>
    </div>
</body>
</html>
"""

    with open(HTML_REPORT, "w", encoding="utf-8") as file:
        file.write(html)


print("\n===================================")
print("   NEXTA SMART PPE SAFETY INSPECTOR")
print("===================================")
print("Model loaded successfully.")
print("Model classes:")
print(model.names)
print("===================================\n")


for image_name in os.listdir(IMAGE_FOLDER):
    if not image_name.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
        continue

    image_path = os.path.join(IMAGE_FOLDER, image_name)
    output_name = f"result_{image_name}"
    output_path = os.path.join(OUTPUT_FOLDER, output_name)

    results = model(image_path)
    result = results[0]

    detected_labels = []

    for box in result.boxes:
        class_id = int(box.cls[0])
        label = normalize_label(model.names[class_id])
        detected_labels.append(label)

    has_human, has_helmet, has_vest, violations, status, risk = classify_image(detected_labels)
    safety_score = get_safety_score(status, violations)
    recommendation = get_recommendation(violations)

    result.save(filename=output_path)
    draw_status(image_path, output_path, status, risk, violations, safety_score)

    if status != "SAFE":
        flagged_path = os.path.join(FLAGGED_FOLDER, f"flagged_{image_name}")
        shutil.copy(output_path, flagged_path)

    print("===================================")
    print(f"Image: {image_name}")
    print("-----------------------------------")
    print("Detected labels:")
    if detected_labels:
        for label in detected_labels:
            print(f"- {label}")
    else:
        print("- No detections")

    print("-----------------------------------")
    print(f"Helmet detected: {'Yes' if has_helmet else 'No'}")
    print(f"Vest detected: {'Yes' if has_vest else 'No'}")
    print(f"Worker detected: {'Yes' if has_human else 'No'}")
    print("-----------------------------------")
    print(f"Status: {status}")
    print(f"Risk Level: {risk}")
    print(f"Safety Score: {safety_score}%")
    print(f"Recommendation: {recommendation}")

    if violations:
        print("Violations:")
        for violation in violations:
            print(f"- {violation}")

    print("===================================\n")

    logs.append({
        "Image": image_name,
        "Worker Detected": has_human,
        "Helmet Detected": has_helmet,
        "Vest Detected": has_vest,
        "Violations": ", ".join(violations) if violations else "None",
        "Status": status,
        "Risk Level": risk,
        "Safety Score": safety_score,
        "Recommendation": recommendation,
        "Output Image": os.path.join(OUTPUT_FOLDER, output_name).replace("\\", "/"),
        "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })


total_images = len(logs)
safe_images = sum(1 for item in logs if item["Status"] == "SAFE")
unsafe_images = sum(1 for item in logs if item["Status"] != "SAFE")
total_violations = sum(
    0 if item["Violations"] == "None" else len(item["Violations"].split(","))
    for item in logs
)
compliance_rate = round((safe_images / total_images) * 100, 1) if total_images > 0 else 0

summary = {
    "total_images": total_images,
    "safe_images": safe_images,
    "unsafe_images": unsafe_images,
    "total_violations": total_violations,
    "compliance_rate": compliance_rate,
}

df = pd.DataFrame(logs)
df.to_csv(LOG_FILE, index=False, encoding="utf-8-sig")

create_html_report(logs, summary)

print("\n========== FINAL SUMMARY ==========")
print(f"Images checked: {total_images}")
print(f"Safe images: {safe_images}")
print(f"Unsafe/check images: {unsafe_images}")
print(f"Total violations: {total_violations}")
print(f"Compliance rate: {compliance_rate}%")
print(f"Annotated images saved to: {OUTPUT_FOLDER}")
print(f"Flagged images saved to: {FLAGGED_FOLDER}")
print(f"CSV log saved as: {LOG_FILE}")
print(f"HTML report saved as: {HTML_REPORT}")
print("===================================")
print("Finished successfully.")

webbrowser.open(HTML_REPORT)