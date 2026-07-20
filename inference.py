import os
import urllib.request

import torch
import cv2
import numpy as np
from ultralytics import YOLO

# ---------------- LOAD MODEL ----------------
# Locally we use the best.pt sitting next to this file. On a cloud deploy the
# weights aren't in the git repo (they're gitignored), so we download them once
# from MODEL_URL — e.g. a GitHub Release asset. Set MODEL_URL in the Streamlit
# app's Secrets. The download is skipped whenever best.pt already exists.
MODEL_PATH = os.path.join(os.path.dirname(__file__), "best.pt")
MODEL_URL = os.environ.get("MODEL_URL", "")


def _ensure_weights():
    if os.path.exists(MODEL_PATH):
        return
    if not MODEL_URL:
        raise FileNotFoundError(
            "best.pt not found and MODEL_URL is not set. Set MODEL_URL "
            "(e.g. a GitHub Release asset URL) so the weights can be downloaded."
        )
    tmp = MODEL_PATH + ".part"
    urllib.request.urlretrieve(MODEL_URL, tmp)
    os.replace(tmp, MODEL_PATH)


_ensure_weights()
model = YOLO(MODEL_PATH)

disease_info = {
    "Anthracnose": {
        "description": "A fungal disease caused by Colletotrichum species that creates dark, sunken lesions on guava leaves and fruits.",
        "cause": "High humidity, rainfall, and fungal spores surviving on infected plant debris.",
        "impact": "Reduces photosynthesis, causes leaf drop, and leads to fruit rot, lowering yield and market value.",
        "treatment": "Spray Carbendazim (1g/L) or Copper Oxychloride every 10–14 days during outbreaks.",
        "organic": "Apply neem oil or Trichoderma-based biofungicide.",
        "prevention": "Prune infected branches, avoid overhead irrigation, and ensure good air circulation.",
        "future_safety": "Plant resistant varieties such as Allahabad Safeda and apply pre-monsoon fungicide schedules."
    },
    "Nutrient_Deficiency": {
        "description": "A physiological disorder where the plant lacks essential nutrients such as nitrogen, iron, or zinc.",
        "cause": "Poor soil fertility, imbalanced fertilizer use, or improper pH.",
        "impact": "Leads to yellowing (chlorosis), weak growth, and reduced fruit size.",
        "treatment": "Apply balanced NPK fertilizer and foliar sprays of iron and zinc.",
        "organic": "Use vermicompost, seaweed extract, or compost tea.",
        "prevention": "Conduct soil testing and follow nutrient management plans.",
        "future_safety": "Install drip fertigation systems for precise nutrient delivery."
    },
    "Wilt": {
        "description": "A soil-borne fungal disease that blocks water movement inside the plant.",
        "cause": "Pathogens like Fusarium in poorly drained soil.",
        "impact": "Sudden wilting, yellowing, and plant death.",
        "treatment": "Soil drenching with carbendazim or fungicides.",
        "organic": "Apply neem cake and Trichoderma in soil.",
        "prevention": "Improve drainage and avoid replanting in infected soil.",
        "future_safety": "Use resistant rootstocks."
    },
    "Insect_Attack": {
        "description": "Damage caused by insects such as fruit flies, aphids, or mealybugs.",
        "cause": "Warm weather and poor orchard hygiene.",
        "impact": "Leaf curling, fruit drop, and secondary infections.",
        "treatment": "Use neem oil or Dimethoate as needed.",
        "organic": "Introduce ladybirds and lacewings.",
        "prevention": "Install pheromone traps and remove infected fruits.",
        "future_safety": "Use integrated pest management (IPM)."
    },
    "Healthy": {
        "description": "The leaf shows normal color, texture, and structure.",
        "cause": "Proper nutrition and environment.",
        "impact": "Good growth and high yield potential.",
        "treatment": "No treatment required.",
        "organic": "Maintain compost and mulch.",
        "prevention": "Continue good farming practices.",
        "future_safety": "Keep digital growth and health records."
    }
}

model.eval()

CONF_THRESH = 0.45

# ---------------- LABEL MAPPING ----------------
LABEL_MAP = {
    "anth": "Anthracnose",
    "anthracnose": "Anthracnose",
    "nut_def": "Nutrient_Deficiency",
    "nutrient_deficiency": "Nutrient_Deficiency",
    "inse_att": "Insect_Attack",
    "ins_att": "Insect_Attack",
    "insect_attack": "Insect_Attack",
    "wilt": "Wilt",
    "healthy": "Healthy",
}


def canonicalize_label(raw_label: str) -> str:
    normalized = str(raw_label).strip().lower()
    if normalized in LABEL_MAP:
        return LABEL_MAP[normalized]

    no_sep = normalized.replace("_", "").replace(" ", "")
    for key, value in LABEL_MAP.items():
        if key.replace("_", "") == no_sep:
            return value

    for disease_key in disease_info:
        if disease_key.lower() == normalized:
            return disease_key

    return str(raw_label)

# ---------------- SEVERITY UTILS ----------------
def classify(sev):
    if sev < 5:
        return "Healthy"
    elif sev < 20:
        return "Mild"
    elif sev < 50:
        return "Moderate"
    else:
        return "Severe"
    
def severity_color(sev):
    if sev < 5:
        return (0, 255, 0)        # Green
    elif sev < 20:
        return (0, 255, 255)      # Yellow
    elif sev < 50:
        return (0, 165, 255)      # Orange
    else:
        return (0, 0, 255)        # Red

# ---------------- YOLO + SEVERITY ----------------
def run_yolo(img):
    raw = img.copy()
    output = img.copy()
    h, w = img.shape[:2]

    boxes = []
    detected_diseases = set()
    disease_mask = np.zeros((h, w), dtype=np.uint8)

    # ---------- YOLO OBB DETECTION ----------
    results = model(img)[0]

    if results.obb is not None:
        polys = results.obb.xyxyxyxy.cpu().numpy()
        classes = results.obb.cls.cpu().numpy()
        confs = results.obb.conf.cpu().numpy()

        for poly, cls, conf in zip(polys, classes, confs):
            if conf < CONF_THRESH:
                continue

            pts = poly.reshape(4, 2).astype(np.int32)
            label = canonicalize_label(results.names[int(cls)])

            detected_diseases.add(label)

            area = cv2.contourArea(pts)
            boxes.append({
                "label": label,
                "conf": float(conf),
                "points": pts,
                "area": area
            })

            cv2.fillPoly(disease_mask, [pts], 255)
            cv2.polylines(output, [pts], True, (0, 255, 0), 2)

    # ---------- PER-BOX LABELS ----------
    for b in boxes:
        cx = int(b["points"][:, 0].mean())
        cy = int(b["points"][:, 1].mean())

        text = f"{b['label']} {b['conf']:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)

        cv2.rectangle(
            output,
            (cx - tw // 2 - 4, cy - th - 6),
            (cx + tw // 2 + 4, cy),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            output,
            text,
            (cx - tw // 2, cy - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

    # ---------- LEAF SEGMENTATION ----------
    hsv = cv2.cvtColor(raw, cv2.COLOR_BGR2HSV)
    leaf_mask = cv2.inRange(hsv, (25, 30, 30), (90, 255, 255))

    cnts, _ = cv2.findContours(leaf_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        leaf_mask[:] = 0
        cv2.drawContours(leaf_mask, [c], -1, 255, -1)

    leaf_area = cv2.countNonZero(leaf_mask)
    disease_area = cv2.countNonZero(cv2.bitwise_and(disease_mask, leaf_mask))

    # ---------- GLOBAL SEVERITY ----------
    severity = (disease_area / leaf_area) * 100 if leaf_area > 0 else 0.0
    level = classify(severity)
    color = severity_color(severity)

    # # ---------- FOOTER ----------
    # font = cv2.FONT_HERSHEY_SIMPLEX
    # font_scale = 0.8
    # thickness = 2
    # footer_height = 35

    # text_left = "Damage: "
    # text_right = f"{severity:.2f}% ({level})"

    # (lw, lh), _ = cv2.getTextSize(text_left, font, font_scale, thickness)
    # (rw, _), _ = cv2.getTextSize(text_right, font, font_scale, thickness)

    # x = (w - (lw + rw)) // 2
    # y = h - (footer_height // 2) + (lh // 2)

    # cv2.rectangle(output, (0, h - footer_height), (w, h), (0, 0, 0), -1)
    # cv2.putText(output, text_left, (x, y), font, font_scale, (255, 255, 255), thickness)
    # cv2.putText(output, text_right, (x + lw, y), font, font_scale, color, thickness)

    if not detected_diseases:
        detected_diseases = {"Healthy"}

    return output, severity, level, list(detected_diseases), results, boxes

def run_gradcam(img, boxes, target_layer=10):
    device = next(model.model.parameters()).device
    was_training = model.model.training
    model.model.eval()

    img_resized = cv2.resize(img, (640, 640))
    inp = torch.from_numpy(img_resized).permute(2, 0, 1).float().unsqueeze(0).to(device).clone().detach()
    inp.requires_grad_(True)

    activations = []
    gradients = []

    def fwd_hook(m, i, o):
        activations.append(o)

    def bwd_hook(m, gi, go):
        gradients.append(go[0])

    layer = model.model.model[target_layer]
    h1 = layer.register_forward_hook(fwd_hook)
    h2 = layer.register_full_backward_hook(bwd_hook)

    try:
        with torch.enable_grad():
            features = model.model.model[:target_layer + 1](inp.clone())

        loss = features.sum()

        model.model.zero_grad()
        loss.backward()

        A = activations[0]
        G = gradients[0]

        weights = G.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * A).sum(dim=1))[0]

        cam = cam.detach().cpu().numpy()
        cam = cv2.resize(cam, (img.shape[1], img.shape[0]))

        cam = np.maximum(cam, 0)

        # normalize
        cam = cam / (cam.max() + 1e-8)

        # preserve multiple hotspots
        cam = np.power(cam, 0.85)

        # enhance contrast across many regions
        cam = cv2.normalize(cam, None, 0, 1, cv2.NORM_MINMAX)

        cam = np.uint8(cam * 255)

        # lighter blur so separate red zones remain visible
        cam = cv2.GaussianBlur(cam, (11, 11), 0)

        overlay = img.copy()

        # =====================================================
        # PASS 1 : APPLY HEATMAPS
        # =====================================================
        for b in boxes:
            pts = b["points"]

            mask = np.zeros(cam.shape, dtype=np.uint8)
            cv2.fillPoly(mask, [pts], 255)

            heat = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
            heat = cv2.bitwise_and(heat, heat, mask=mask)

            overlay = cv2.addWeighted(overlay, 0.87, heat, 0.30, 0)

        # =====================================================
        # PASS 2 : DRAW GREEN BOXES
        # =====================================================
        for b in boxes:
            pts = b["points"]
            cv2.polylines(overlay, [pts], True, (0, 255, 0), 2)

        # =====================================================
        # PASS 3 : DRAW LABELS LAST (SHARP)
        # =====================================================
        for b in boxes:
            pts = b["points"]
            label = b["label"]
            conf = b["conf"]

            cx = int(pts[:, 0].mean())
            cy = int(pts[:, 1].mean())

            short = label.replace("Nutrient_Deficiency", "Nutrient")
            short = short.replace("Insect_Attack", "Insect")

            text = f"{short} {conf:.2f}"

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.55
            thickness = 2

            (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)

            x1 = cx - tw // 2 - 4
            y1 = cy - th - 6
            x2 = cx + tw // 2 + 4
            y2 = cy

            h_img, w_img = overlay.shape[:2]

            x1 = max(0, x1)
            y1 = max(th + 4, y1)
            x2 = min(w_img, x2)

            cv2.rectangle(
                overlay,
                (x1, y1),
                (x2, y2),
                (0, 0, 0),
                -1
            )

            cv2.putText(
                overlay,
                text,
                (x1 + 4, y2 - 3),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA
            )

        return overlay

    finally:
        h1.remove()
        h2.remove()
        model.model.train(was_training)










