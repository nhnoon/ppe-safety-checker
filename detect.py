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


def yes_no_icon(value):
    return "✅" if value else "❌"


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
    if not violations:
        return "No action required. PPE compliance looks good."

    recommendations = []

    if "Missing Helmet" in violations:
        recommendations.append("Provide a certified hard hat before entering the site.")

    if "Missing Safety Vest" in violations:
        recommendations.append("Wear a reflective safety vest for visibility and compliance.")

    if "No Worker Detected" in violations:
        recommendations.append("Review the image manually because no worker was detected.")

    return " | ".join(recommendations)


def get_action_label(violations, status):
    if status == "SAFE":
        return "No Action"
    if "No Worker Detected" in violations:
        return "Manual Review"
    if len(violations) >= 2:
        return "Stop Work"
    if "Missing Helmet" in violations:
        return "Wear Helmet"
    if "Missing Safety Vest" in violations:
        return "Wear Vest"
    return "Review"


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

    cv2.rectangle(image, (20, 20), (900, 145), (0, 0, 0), -1)
    cv2.putText(image, title, (35, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3)
    cv2.putText(image, f"Safety Score: {safety_score}%", (35, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)

    y = 180
    for violation in violations:
        cv2.putText(image, f"- {violation}", (35, y), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 2)
        y += 38

    cv2.imwrite(output_path, image)


def create_html_report(logs, summary):
    rows = ""

    for item in logs:
        status_class = item["Status"].lower()
        risk_class = item["Risk Level"].lower()
        action_class = item["Action Label"].lower().replace(" ", "-")
        image_link = item["Output Image"]

        rows += f"""
        <tr data-status="{item["Status"]}" data-risk="{item["Risk Level"]}">
            <td>
                <a href="{image_link}" target="_blank">
                    <img class="thumb" src="{image_link}" alt="output image">
                </a>
            </td>
            <td class="image-name">{item["Image"]}</td>
            <td><span class="badge status {status_class}">{item["Status"]}</span></td>
            <td><span class="badge risk {risk_class}">{item["Risk Level"]}</span></td>
            <td>
                <div class="score-wrap">
                    <span>{item["Safety Score"]}%</span>
                    <div class="score-bar">
                        <div class="score-fill" style="width:{item["Safety Score"]}%"></div>
                    </div>
                </div>
            </td>
            <td>{item["Violations"]}</td>
            <td><span class="action {action_class}">{item["Action Label"]}</span></td>
            <td>{item["Recommendation"]}</td>
            <td class="center">{yes_no_icon(item["Worker Detected"])}</td>
            <td class="center">{yes_no_icon(item["Helmet Detected"])}</td>
            <td class="center">{yes_no_icon(item["Vest Detected"])}</td>
            <td>{item["Time"]}</td>
        </tr>
        """

    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>NEXTA Smart PPE Safety Dashboard</title>
<style>
    * {{
        box-sizing: border-box;
    }}

    body {{
        margin: 0;
        font-family: "Segoe UI", Arial, sans-serif;
        background: #0f172a;
        color: #e5e7eb;
    }}

    .page {{
        max-width: 1500px;
        margin: auto;
        padding: 30px;
    }}

    .hero {{
        background: linear-gradient(135deg, #111827, #1e3a8a);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 24px;
        padding: 32px;
        margin-bottom: 24px;
        box-shadow: 0 20px 50px rgba(0,0,0,0.25);
    }}

    .hero h1 {{
        margin: 0;
        font-size: 36px;
        letter-spacing: -0.5px;
    }}

    .hero p {{
        color: #cbd5e1;
        font-size: 16px;
        margin-top: 10px;
        max-width: 900px;
    }}

    .summary-box {{
        margin-top: 20px;
        background: rgba(255,255,255,0.08);
        border-radius: 18px;
        padding: 18px;
        line-height: 1.7;
        color: #f8fafc;
    }}

    .cards {{
        display: grid;
        grid-template-columns: repeat(6, 1fr);
        gap: 16px;
        margin-bottom: 24px;
    }}

    .card {{
        background: #111827;
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 20px;
        padding: 20px;
        box-shadow: 0 12px 30px rgba(0,0,0,0.22);
    }}

    .card .label {{
        color: #94a3b8;
        font-size: 13px;
        margin-bottom: 10px;
    }}

    .card .value {{
        font-size: 32px;
        font-weight: 800;
        color: white;
    }}

    .card.safe-card .value {{
        color: #22c55e;
    }}

    .card.unsafe-card .value {{
        color: #ef4444;
    }}

    .card.warning-card .value {{
        color: #f59e0b;
    }}

    .card.score-card .value {{
        color: #38bdf8;
    }}

    .analytics {{
        display: grid;
        grid-template-columns: 1fr 1.4fr;
        gap: 18px;
        margin-bottom: 24px;
    }}

    .panel {{
        background: #111827;
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 20px;
        padding: 22px;
        box-shadow: 0 12px 30px rgba(0,0,0,0.22);
    }}

    .panel h2 {{
        margin-top: 0;
        margin-bottom: 18px;
        font-size: 20px;
    }}

    .compliance-circle {{
        width: 190px;
        height: 190px;
        margin: 10px auto 18px;
        border-radius: 50%;
        background:
            conic-gradient(#22c55e 0deg {summary["compliance_rate"] * 3.6}deg, #334155 {summary["compliance_rate"] * 3.6}deg 360deg);
        display: flex;
        align-items: center;
        justify-content: center;
        position: relative;
    }}

    .compliance-circle::after {{
        content: "";
        width: 130px;
        height: 130px;
        background: #111827;
        border-radius: 50%;
        position: absolute;
    }}

    .circle-text {{
        position: relative;
        z-index: 2;
        text-align: center;
    }}

    .circle-text strong {{
        display: block;
        font-size: 34px;
        color: white;
    }}

    .circle-text span {{
        color: #94a3b8;
        font-size: 13px;
    }}

    .risk-row {{
        margin-bottom: 16px;
    }}

    .risk-label {{
        display: flex;
        justify-content: space-between;
        margin-bottom: 7px;
        color: #cbd5e1;
        font-size: 14px;
    }}

    .bar {{
        height: 13px;
        background: #334155;
        border-radius: 999px;
        overflow: hidden;
    }}

    .bar-fill {{
        height: 100%;
        border-radius: 999px;
    }}

    .low-fill {{
        background: #22c55e;
        width: {summary["low_percent"]}%;
    }}

    .medium-fill {{
        background: #f59e0b;
        width: {summary["medium_percent"]}%;
    }}

    .high-fill {{
        background: #ef4444;
        width: {summary["high_percent"]}%;
    }}

    .unknown-fill {{
        background: #64748b;
        width: {summary["unknown_percent"]}%;
    }}

    .toolbar {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 12px;
        margin-bottom: 14px;
    }}

    .toolbar h2 {{
        margin: 0;
        font-size: 22px;
    }}

    input {{
        background: #020617;
        border: 1px solid #334155;
        color: white;
        padding: 12px 14px;
        border-radius: 12px;
        width: 300px;
        outline: none;
    }}

    input::placeholder {{
        color: #64748b;
    }}

    table {{
        width: 100%;
        border-collapse: collapse;
        background: #111827;
        border-radius: 20px;
        overflow: hidden;
        box-shadow: 0 12px 30px rgba(0,0,0,0.22);
        border: 1px solid rgba(255,255,255,0.08);
    }}

    th, td {{
        padding: 13px;
        border-bottom: 1px solid rgba(255,255,255,0.08);
        text-align: left;
        font-size: 13px;
        vertical-align: middle;
    }}

    th {{
        background: #020617;
        color: #f8fafc;
        position: sticky;
        top: 0;
        z-index: 1;
    }}

    tr:hover {{
        background: rgba(255,255,255,0.04);
    }}

    .thumb {{
        width: 90px;
        height: 60px;
        object-fit: cover;
        border-radius: 10px;
        border: 1px solid rgba(255,255,255,0.14);
    }}

    .image-name {{
        font-weight: 600;
        color: #f8fafc;
    }}

    .badge {{
        display: inline-block;
        padding: 7px 11px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 800;
        color: white;
        white-space: nowrap;
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

    .low {{
        background: #16a34a;
    }}

    .medium {{
        background: #f59e0b;
    }}

    .high {{
        background: #dc2626;
    }}

    .unknown {{
        background: #64748b;
    }}

    .score-wrap span {{
        font-weight: 700;
    }}

    .score-bar {{
        margin-top: 6px;
        height: 8px;
        background: #334155;
        border-radius: 999px;
        overflow: hidden;
        width: 90px;
    }}

    .score-fill {{
        height: 100%;
        background: #38bdf8;
        border-radius: 999px;
    }}

    .action {{
        display: inline-block;
        padding: 7px 10px;
        border-radius: 10px;
        font-size: 12px;
        font-weight: 800;
        white-space: nowrap;
    }}

    .no-action {{
        background: rgba(34,197,94,0.15);
        color: #86efac;
    }}

    .wear-helmet, .wear-vest, .manual-review, .review {{
        background: rgba(245,158,11,0.16);
        color: #fcd34d;
    }}

    .stop-work {{
        background: rgba(239,68,68,0.16);
        color: #fca5a5;
    }}

    .center {{
        text-align: center;
        font-size: 18px;
    }}

    .footer {{
        text-align: center;
        color: #94a3b8;
        margin-top: 24px;
        font-size: 13px;
    }}

    @media (max-width: 1100px) {{
        .cards {{
            grid-template-columns: repeat(2, 1fr);
        }}

        .analytics {{
            grid-template-columns: 1fr;
        }}

        input {{
            width: 100%;
        }}

        .toolbar {{
            flex-direction: column;
            align-items: stretch;
        }}
    }}
</style>
</head>

<body>
<div class="page">

    <section class="hero">
        <h1>🦺 NEXTA Smart PPE Safety Dashboard</h1>
        <p>
            AI-powered construction site inspection dashboard using YOLOv8.
            This report summarizes PPE compliance, safety risks, detected violations, and recommended actions.
        </p>

        <div class="summary-box">
            <strong>Executive Summary:</strong>
            The system processed <strong>{summary["total_images"]}</strong> images.
            <strong>{summary["safe_images"]}</strong> were safe, while
            <strong>{summary["unsafe_images"]}</strong> required attention.
            The overall compliance rate is <strong>{summary["compliance_rate"]}%</strong>,
            with an average safety score of <strong>{summary["average_score"]}%</strong>.
        </div>
    </section>

    <section class="cards">
        <div class="card">
            <div class="label">Total Images</div>
            <div class="value">{summary["total_images"]}</div>
        </div>
        <div class="card safe-card">
            <div class="label">Safe Images</div>
            <div class="value">{summary["safe_images"]}</div>
        </div>
        <div class="card unsafe-card">
            <div class="label">Unsafe / Check</div>
            <div class="value">{summary["unsafe_images"]}</div>
        </div>
        <div class="card warning-card">
            <div class="label">Total Violations</div>
            <div class="value">{summary["total_violations"]}</div>
        </div>
        <div class="card score-card">
            <div class="label">Compliance Rate</div>
            <div class="value">{summary["compliance_rate"]}%</div>
        </div>
        <div class="card score-card">
            <div class="label">Average Score</div>
            <div class="value">{summary["average_score"]}%</div>
        </div>
    </section>

    <section class="analytics">
        <div class="panel">
            <h2>Compliance Overview</h2>
            <div class="compliance-circle">
                <div class="circle-text">
                    <strong>{summary["compliance_rate"]}%</strong>
                    <span>Compliance</span>
                </div>
            </div>
        </div>

        <div class="panel">
            <h2>Risk Distribution</h2>

            <div class="risk-row">
                <div class="risk-label"><span>LOW Risk</span><strong>{summary["low_count"]}</strong></div>
                <div class="bar"><div class="bar-fill low-fill"></div></div>
            </div>

            <div class="risk-row">
                <div class="risk-label"><span>MEDIUM Risk</span><strong>{summary["medium_count"]}</strong></div>
                <div class="bar"><div class="bar-fill medium-fill"></div></div>
            </div>

            <div class="risk-row">
                <div class="risk-label"><span>HIGH Risk</span><strong>{summary["high_count"]}</strong></div>
                <div class="bar"><div class="bar-fill high-fill"></div></div>
            </div>

            <div class="risk-row">
                <div class="risk-label"><span>UNKNOWN / Check</span><strong>{summary["unknown_count"]}</strong></div>
                <div class="bar"><div class="bar-fill unknown-fill"></div></div>
            </div>
        </div>
    </section>

    <section class="panel">
        <div class="toolbar">
            <h2>Inspection Results</h2>
            <input type="text" id="searchInput" placeholder="Search by image, status, risk, or violation...">
        </div>

        <table id="inspectionTable">
            <thead>
                <tr>
                    <th>Preview</th>
                    <th>Image</th>
                    <th>Status</th>
                    <th>Risk</th>
                    <th>Score</th>
                    <th>Violations</th>
                    <th>Action</th>
                    <th>Recommendation</th>
                    <th>Worker</th>
                    <th>Helmet</th>
                    <th>Vest</th>
                    <th>Time</th>
                </tr>
            </thead>
            <tbody>
                {rows}
            </tbody>
        </table>
    </section>

    <div class="footer">
        Generated automatically by NEXTA Smart PPE Safety Inspector Prototype · YOLOv8 · Python · OpenCV
    </div>

</div>

<script>
    const searchInput = document.getElementById("searchInput");
    const tableRows = document.querySelectorAll("#inspectionTable tbody tr");

    searchInput.addEventListener("keyup", function() {{
        const value = searchInput.value.toLowerCase();

        tableRows.forEach(row => {{
            const text = row.innerText.toLowerCase();
            row.style.display = text.includes(value) ? "" : "none";
        }});
    }});
</script>

</body>
</html>
"""

    with open(HTML_REPORT, "w", encoding="utf-8") as file:
        file.write(html)


print("\n================================================")
print("        NEXTA SMART PPE SAFETY INSPECTOR")
print("================================================")
print("Loading YOLOv8 model...")
print("✓ Model loaded successfully")
print(f"✓ Model path: {MODEL_PATH}")
print(f"✓ Input folder: {IMAGE_FOLDER}")
print("================================================\n")


for image_name in os.listdir(IMAGE_FOLDER):
    if not image_name.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
        continue

    image_path = os.path.join(IMAGE_FOLDER, image_name)
    output_name = f"result_{image_name}"
    output_path = os.path.join(OUTPUT_FOLDER, output_name)

    print(f"Processing: {image_name}")

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
    action_label = get_action_label(violations, status)

    result.save(filename=output_path)
    draw_status(image_path, output_path, status, risk, violations, safety_score)

    if status != "SAFE":
        flagged_path = os.path.join(FLAGGED_FOLDER, f"flagged_{image_name}")
        shutil.copy(output_path, flagged_path)

    print("-----------------------------------------------")
    print(f"Status       : {status}")
    print(f"Risk Level   : {risk}")
    print(f"Safety Score : {safety_score}%")
    print(f"Violations   : {', '.join(violations) if violations else 'None'}")
    print(f"Action       : {action_label}")
    print("-----------------------------------------------\n")

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
        "Action Label": action_label,
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
average_score = round(sum(item["Safety Score"] for item in logs) / total_images, 1) if total_images > 0 else 0

low_count = sum(1 for item in logs if item["Risk Level"] == "LOW")
medium_count = sum(1 for item in logs if item["Risk Level"] == "MEDIUM")
high_count = sum(1 for item in logs if item["Risk Level"] == "HIGH")
unknown_count = sum(1 for item in logs if item["Risk Level"] == "UNKNOWN")

low_percent = round((low_count / total_images) * 100, 1) if total_images > 0 else 0
medium_percent = round((medium_count / total_images) * 100, 1) if total_images > 0 else 0
high_percent = round((high_count / total_images) * 100, 1) if total_images > 0 else 0
unknown_percent = round((unknown_count / total_images) * 100, 1) if total_images > 0 else 0

summary = {
    "total_images": total_images,
    "safe_images": safe_images,
    "unsafe_images": unsafe_images,
    "total_violations": total_violations,
    "compliance_rate": compliance_rate,
    "average_score": average_score,
    "low_count": low_count,
    "medium_count": medium_count,
    "high_count": high_count,
    "unknown_count": unknown_count,
    "low_percent": low_percent,
    "medium_percent": medium_percent,
    "high_percent": high_percent,
    "unknown_percent": unknown_percent,
}

df = pd.DataFrame(logs)
df.to_csv(LOG_FILE, index=False, encoding="utf-8-sig")

create_html_report(logs, summary)

print("\n================ FINAL SUMMARY ================")
print(f"Images Processed      : {total_images}")
print(f"Safe Images           : {safe_images}")
print(f"Unsafe / Check Images : {unsafe_images}")
print(f"Total Violations      : {total_violations}")
print(f"Compliance Rate       : {compliance_rate}%")
print(f"Average Safety Score  : {average_score}%")
print("------------------------------------------------")
print(f"Annotated Images      : {OUTPUT_FOLDER}")
print(f"Flagged Images        : {FLAGGED_FOLDER}")
print(f"CSV Log               : {LOG_FILE}")
print(f"HTML Dashboard        : {HTML_REPORT}")
print("================================================")
print("Inspection completed successfully ✓")

webbrowser.open(HTML_REPORT)