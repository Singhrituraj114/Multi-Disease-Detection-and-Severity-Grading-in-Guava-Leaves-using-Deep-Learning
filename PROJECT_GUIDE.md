# PROJECT GUIDE — Multi-Disease Detection and Severity Grading in Guava Leaves using Deep Learning

> This file explains everything about our project in simple words — what we built, how it works, and what questions your guide may ask during presentation.

---

## TABLE OF CONTENTS

0. [DEEP DIVE — How YOLOv8 Works Inside (Architecture)](#0-deep-dive--how-yolov8-works-inside-architecture)
   - 0.1 [What is a Pixel? What does the model see?](#01-what-is-a-pixel-what-does-the-model-see)
   - 0.2 [Convolution (Conv2D) — The Core Operation](#02-convolution-conv2d--the-core-operation)
   - 0.3 [Feature Maps — What the Model Learns](#03-feature-maps--what-the-model-learns)
   - 0.4 [Pooling — Making It Smaller and Faster](#04-pooling--making-it-smaller-and-faster)
   - 0.5 [Batch Normalization and Activation (ReLU)](#05-batch-normalization-and-activation-relu)
   - 0.6 [YOLOv8 Full Architecture — Backbone, Neck, Head](#06-yolov8-full-architecture--backbone-neck-head)
   - 0.7 [Backbone — Feature Extraction](#07-backbone--feature-extraction)
   - 0.8 [Neck (FPN + PAN) — Feature Fusion](#08-neck-fpn--pan--feature-fusion)
   - 0.9 [Head — Making Predictions](#09-head--making-predictions)
   - 0.10 [OBB — Oriented Bounding Box Output](#010-obb--oriented-bounding-box-output)
   - 0.11 [mAP@0.5 and mAP@0.5-0.95 Explained](#011-map05-and-map05-095-explained)
   - 0.12 [Training Flow — How the Model Learned](#012-training-flow--how-the-model-learned)
1. [What This Project Does](#1-what-this-project-does)
2. [Why Guava? Why AI?](#2-why-guava-why-ai)
3. [How the Project is Structured](#3-how-the-project-is-structured)
4. [Each Part Explained](#4-each-part-explained)
   - 4.1 [Dataset and Labelling](#41-dataset-and-labelling)
   - 4.2 [The AI Model (best.pt)](#42-the-ai-model-bestpt)
   - 4.3 [inference.py — Where the AI Runs](#43-inferencepy--where-the-ai-runs)
   - 4.4 [app.py — The Website](#44-apppy--the-website)
   - 4.5 [PDF Report](#45-pdf-report)
   - 4.6 [Grad-CAM — Showing Why the AI Decided](#46-grad-cam--showing-why-the-ai-decided)
5. [Step-by-Step: What Happens When You Upload a Photo](#5-step-by-step-what-happens-when-you-upload-a-photo)
6. [Disease Classes](#6-disease-classes)
7. [How Severity is Measured](#7-how-severity-is-measured)
8. [Tools and Libraries We Used](#8-tools-and-libraries-we-used)
9. [How to Run the Project](#9-how-to-run-the-project)
10. [Questions Your Guide May Ask — With Simple Answers](#10-questions-your-guide-may-ask--with-simple-answers)
    - 10.1 [Basic Questions](#101-basic-questions)
    - 10.2 [AI Model Questions](#102-ai-model-questions)
    - 10.3 [Code / Working Questions](#103-code--working-questions)
    - 10.4 [Dataset and Training Questions](#104-dataset-and-training-questions)
    - 10.5 [Real-World Use Questions](#105-real-world-use-questions)
    - 10.6 [Harder Questions](#106-harder-questions)

---

## 0. DEEP DIVE — How YOLOv8 Works Inside (Architecture)

> This section explains step by step exactly how our AI model works — from a raw image going in, to a disease box coming out. Read this carefully before your presentation.

---

### 0.1 What is a Pixel? What does the model see?

Before the AI does anything, it needs to understand the image as numbers.

Every image is made of tiny squares called **pixels**. Each pixel has 3 numbers:
- **R** = how much Red (0–255)
- **G** = how much Green (0–255)
- **B** = how much Blue (0–255)

So a 640×640 image = **640 × 640 × 3 = 1,228,800 numbers** fed into the model.

```
A 640x640 image looks like this to the model:

[ [[R,G,B], [R,G,B], [R,G,B], ... ],   ← row 1, 640 pixels
  [[R,G,B], [R,G,B], [R,G,B], ... ],   ← row 2
  ...                                   ← 640 rows total
]

Shape of input tensor = (1, 3, 640, 640)
   1     → one image (batch size)
   3     → 3 color channels (R, G, B)
  640    → height in pixels
  640    → width in pixels
```

---

### 0.2 Convolution (Conv2D) — The Core Operation

Convolution is the most important operation in a CNN. Think of it like a **sliding magnifying glass** that scans the image and picks up patterns.

**How it works:**

```
IMAGE PATCH (5x5 pixels):        FILTER / KERNEL (3x3):
┌─────────────────────┐          ┌───────────┐
│  10  20  30  40  50 │          │  1   0  -1│
│  60  70  80  90 100 │          │  2   0  -2│
│ 110 120 130 140 150 │    ──►   │  1   0  -1│
│ 160 170 180 190 200 │          └───────────┘
│ 210 220 230 240 250 │
└─────────────────────┘

The filter slides over every 3x3 patch of the image.
At each position it does:
  → Multiply each filter value with the pixel it covers
  → Add all products together
  → Write that ONE number as the output

This one number summarises: "Did this filter's pattern exist here?"
```

**What filters detect:**
- Early filters → simple edges (horizontal, vertical, diagonal lines)
- Middle filters → shapes, corners, blobs
- Deep filters → complex textures like disease spots, veins, colour patches

**Key terms:**
- **Kernel / Filter** → the small sliding window (e.g., 3×3 or 1×1)
- **Stride** → how many pixels the filter moves each step (stride=2 means skip every other pixel → output is half the size)
- **Padding** → adding zeros around the image border so the output stays the same size as the input

```
Conv2D STEP-BY-STEP:

Input image  → [1, 3, 640, 640]

Apply 32 filters of size 3x3 with stride 2:
  → Each filter scans the whole image
  → Each filter produces one output grid (feature map)
  → 32 filters → 32 feature maps

Output → [1, 32, 320, 320]
  (size halved because stride=2, but 32 different feature views)
```

---

### 0.3 Feature Maps — What the Model Learns

Each convolution layer produces **Feature Maps** — grids of numbers that represent what the model noticed at each location.

```
One feature map = one filter's view of the image.

Layer 1 feature maps (low level):       Layer 5 feature maps (high level):
┌───────────────────┐                   ┌───────────────────┐
│  edges detected   │                   │  disease patches  │
│  (blurry, simple) │                   │  detected here    │
│                   │                   │  (complex shapes) │
└───────────────────┘                   └───────────────────┘

As we go deeper, feature maps get:
  → SMALLER in spatial size (width × height shrink)
  → MORE in number (more channels = more patterns detected)
  → RICHER in meaning (simple → complex features)
```

---

### 0.4 Pooling — Making It Smaller and Faster

After convolution, we often apply **Pooling** to reduce the size of feature maps without losing the most important information.

**Max Pooling (most common):**

```
2x2 Max Pooling with stride 2:

Input (4x4):         Output (2x2):
┌──┬──┬──┬──┐        ┌────┬────┐
│ 1│ 3│ 2│ 4│        │    │    │
├──┼──┼──┼──┤   →    │  3 │  4 │  ← maximum of each 2x2 block
│ 5│ 6│ 1│ 2│        │    │    │
├──┼──┼──┼──┤        ├────┼────┤
│ 7│ 2│ 9│ 1│        │    │    │
├──┼──┼──┼──┤   →    │  7 │  9 │  ← maximum of each 2x2 block
│ 3│ 4│ 0│ 8│        │    │    │
└──┴──┴──┴──┘        └────┴────┘

Rule: Keep the LARGEST number in each 2x2 window.
Why? The largest number = strongest feature response = most important signal.
```

**Why pooling matters:**
- Reduces computation (smaller grid = fewer calculations)
- Makes the model tolerant to small shifts (if the disease spot moves slightly, the max value stays the same)
- YOLOv8 uses stride-2 convolutions instead of classic pooling, but the effect is the same

---

### 0.5 Batch Normalization and Activation (ReLU)

After every Conv2D, two more operations happen:

#### Batch Normalization (BatchNorm)
- The numbers coming out of a convolution can be very large or very small.
- BatchNorm re-scales all the numbers to have mean ≈ 0 and range around 1.
- This makes training stable and faster — without it the model takes too long to learn.

```
Before BatchNorm:  [0.001, 5000, -3000, 0.02, 800]   ← all over the place
After  BatchNorm:  [-0.5,   1.2,  -1.1,  -0.4, 0.9]  ← nicely scaled
```

#### ReLU (Rectified Linear Unit) — Activation Function
- After BatchNorm, every negative number is set to zero.
- Positive numbers stay as they are.

```
ReLU:
  Input:  [-3, -1, 0, 2, 5, -0.5]
  Output: [ 0,  0, 0, 2, 5,  0  ]

Why? Negative values mean "this filter did NOT detect its pattern here."
We don't need negatives. Setting them to 0 keeps only real detections.
```

> **In YOLOv8 the standard block is called C2f = Conv2D + BatchNorm + SiLU activation**
> SiLU is slightly smoother than ReLU but works the same way in concept.

---

### 0.6 YOLOv8 Full Architecture — Backbone, Neck, Head

YOLOv8 is split into three parts:

```
INPUT IMAGE (640×640×3)
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│                    BACKBONE                                   │
│   (extracts features — like building a rich picture summary)  │
│                                                               │
│  Stem Conv → C2f × N → C2f × N → C2f × N → C2f × N → SPPF   │
│  640→320   → 320→160 → 160→80 → 80→40  → 40→20               │
│                                                               │
│  Produces 3 feature maps at different scales:                 │
│    P3 = 80×80  (small diseases — fine details)                │
│    P4 = 40×40  (medium diseases)                              │
│    P5 = 20×20  (large diseases — big picture)                 │
└───────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│                     NECK (FPN + PAN)                          │
│   (merges features from all scales together — fusion)         │
│                                                               │
│   P5 upsampled → merged with P4  →  merged output            │
│   P4 upsampled → merged with P3  →  merged output            │
│   merged P3    → downsampled → merged with P4 → merged output │
│   merged P4    → downsampled → merged with P5 → merged output │
└───────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│                     HEAD                                      │
│   (makes final predictions — disease label + box coordinates) │
│                                                               │
│   For each of 3 scales (P3, P4, P5):                          │
│     → Predict OBB coordinates (8 corner points of rotated box)│
│     → Predict class probabilities (Anthracnose? Wilt? etc.)   │
│     → Predict confidence score (how sure?)                    │
└───────────────────────────────────────────────────────────────┘
        │
        ▼
  NMS (Non-Max Suppression) → remove duplicate boxes
        │
        ▼
  FINAL OUTPUT: boxes + labels + confidence scores
```

---

### 0.7 Backbone — Feature Extraction

The backbone is the main "seeing" part of the model. It reads the raw image and extracts features at multiple levels of detail.

```
RAW IMAGE
640 × 640 × 3
     │
     ▼
[Stem — Conv2D 3→32, stride 2]
     Shrinks: 640→320
     Learns: basic edges and colour patches
     │
     ▼
[C2f Block — 32→64, stride 2]
     Shrinks: 320→160
     Learns: simple shapes, spots
     │
     ▼
[C2f Block — 64→128, stride 2]
     Shrinks: 160→80  ← this is called P3 (small-object scale)
     Learns: leaf textures, early disease marks
     │
     ▼
[C2f Block — 128→256, stride 2]
     Shrinks: 80→40   ← this is called P4 (medium-object scale)
     Learns: disease patch shapes
     │
     ▼
[C2f Block — 256→512, stride 2]
     Shrinks: 40→20   ← this is called P5 (large-object scale)
     Learns: full disease regions, overall leaf health
     │
     ▼
[SPPF — Spatial Pyramid Pooling Fast]
     Applies max pooling at 3 different window sizes (5×5, 9×9, 13×13)
     Merges all results → captures context at multiple zoom levels
     Still 20×20 but much richer information
```

**What is C2f?**
C2f stands for "Cross Stage Partial with 2 convolutions and feature reuse." It works like this:

```
C2f block:
  Input
    │
    ├─→ Conv (half channels) ─→ [Bottleneck × N] ─→ output_A
    │
    └─→ Conv (half channels) ─────────────────────→ output_B
    
  output_A + output_B → Concatenate → Conv → Final output

Why? Splitting and merging forces the model to learn both
detailed features AND broad features in parallel.
```

---

### 0.8 Neck (FPN + PAN) — Feature Fusion

After the backbone we have 3 feature maps at different sizes (P3=80×80, P4=40×40, P5=20×20). The problem is:

- P3 has **fine detail** but only sees small regions → good for small spots
- P5 has **big-picture understanding** but blurry on small details → good for large regions

The **Neck** combines them so every scale benefits from every level of understanding.

**FPN → top-down path (big to small, adding detail):**

```
P5 (20×20, deep knowledge)
     │
     ▼ Upsample 2× → 40×40
     + P4 (40×40, medium knowledge)
     ▼
 Merged P4 (40×40) — now knows both big-picture AND medium details
     │
     ▼ Upsample 2× → 80×80
     + P3 (80×80, fine detail)
     ▼
 Merged P3 (80×80) — now has ALL levels of knowledge
```

**PAN → bottom-up path (small to big, adding location precision):**

```
Merged P3 (80×80, rich detail)
     │
     ▼ Downsample 2× → 40×40
     + Merged P4
     ▼
 Final P4 — sharp + contextual
     │
     ▼ Downsample 2× → 20×20
     + P5
     ▼
 Final P5 — sharp + contextual
```

**After Neck we have 3 refined feature maps:** P3, P4, P5 — each now contains both fine and coarse information. This is why YOLOv8 can detect both tiny disease spots AND large disease regions in the same image.

---

### 0.9 Head — Making Predictions

The Head reads the three feature maps (P3, P4, P5) and makes the final answer.

```
For each feature map position (every grid cell), the head predicts:

┌──────────────────────────────────────────────────────────┐
│  CLASS SCORES: probability for each disease              │
│    [Anthracnose=0.87, Wilt=0.03, Insect=0.02, ...]       │
│                                                          │
│  CONFIDENCE: how sure the model is (0 to 1)              │
│    e.g., 0.91                                            │
│                                                          │
│  BOX COORDINATES (for OBB):                              │
│    8 values = 4 corner points (x1,y1), (x2,y2),         │
│               (x3,y3), (x4,y4) → rotated box             │
└──────────────────────────────────────────────────────────┘

P3 (80×80 grid) → detects small disease spots
P4 (40×40 grid) → detects medium disease patches
P5 (20×20 grid) → detects large infected regions

Total candidate boxes = 80×80 + 40×40 + 20×20 = 8800 boxes
Most have very low confidence → filtered out by threshold (0.45)
Remaining boxes → Non-Max Suppression → final clean boxes
```

**Non-Max Suppression (NMS):**
Multiple grid cells near the same disease spot will all predict a box. NMS keeps only the best one and removes overlapping ones.

```
Before NMS:       After NMS:
[box A 0.91]      [box A 0.91]  ← keep best
[box B 0.88]      removed       ← overlaps A too much (IoU > 0.45)
[box C 0.85]      removed       ← overlaps A too much
[box D 0.72]      [box D 0.72]  ← keep (different location)
```

---

### 0.10 OBB — Oriented Bounding Box Output

Normal YOLO predicts: `(center_x, center_y, width, height)` — always a straight rectangle.

YOLOv8 OBB predicts: 4 corner points — `(x1,y1), (x2,y2), (x3,y3), (x4,y4)` — a rotated rectangle.

```
Normal Bounding Box (AABB):        Oriented Bounding Box (OBB):

 ┌──────────────────────┐                ╱────────╲
 │  ╱╲  disease  ╱╲    │               ╱  disease ╲
 │ ╱  ╲  patch  ╱  ╲   │              ╱  perfectly ╲
 │╱    ╲       ╱    ╲  │             ╲   fitting    ╱
 └──────────────────────┘              ╲           ╱
                                        ╲─────────╱

Problem: box includes a lot of         Box tightly wraps the disease.
healthy leaf (wasted area)             Less healthy leaf included.
Severity calculation is wrong.         Severity is accurate.
```

Disease spots on leaves are rarely perfectly horizontal. OBB allows the box to tilt and rotate to match the shape of the disease, giving a much tighter fit.

---

### 0.11 mAP@0.5 and mAP@0.5-0.95 Explained

These are the two main numbers used to say "how accurate is our model?"

Before understanding mAP, we need to know three simpler terms:

**Precision:**
```
Out of everything the model said was a disease, how many actually were?

Precision = True Positives / (True Positives + False Positives)

Example: Model detected 10 boxes. 8 were real diseases. 2 were wrong.
Precision = 8/10 = 0.80 = 80%
```

**Recall:**
```
Out of all the actual diseases in the photos, how many did the model find?

Recall = True Positives / (True Positives + False Negatives)

Example: There were 12 real disease spots total. Model found 8.
Recall = 8/12 = 0.67 = 67%
```

**AP (Average Precision) — for one class:**
```
AP is the area under the Precision-Recall curve.

  Precision
  1.0 │╲
      │  ╲
  0.8 │   ╲___
      │       ╲___
  0.6 │           ╲___
      │                ╲
  0.0 └────────────────── Recall
      0.0  0.2  0.4  0.6  0.8  1.0

  Area under this curve = AP for that class.
  AP = 1.0 means perfect (found every disease, no false alarms).
```

**mAP (mean Average Precision) = average AP across all classes.**
```
mAP = (AP_Anthracnose + AP_Wilt + AP_Deficiency + AP_Insect + AP_Healthy) / 5
```

---

**mAP@0.5 — What does the @0.5 mean?**

A detection is only called "correct" if the predicted box overlaps the real box with IoU ≥ 0.5 (50% overlap).

```
  Predicted box:   ┌──────────┐
                   │          │
  Real box:    ┌───│──────┐   │
               │   │      │   │
               │   └──────┼───┘
               │          │
               └──────────┘

  Overlap area = the middle part where both boxes cover.
  IoU = Overlap / (Combined area of both boxes)

  If IoU ≥ 0.5 → this detection is CORRECT ✓
  If IoU < 0.5 → this detection is WRONG ✗
```

So **mAP@0.5** = our model's accuracy when we only need 50% box overlap.
This is the main metric shown in our results.

---

**mAP@0.5-0.95 — Stricter version:**

Instead of testing at just IoU=0.5, we test at 10 different levels:
IoU = 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95

Then we average the mAP at all 10 levels.

```
mAP@0.50 = 0.91  ← easy, 50% overlap needed
mAP@0.55 = 0.89
mAP@0.60 = 0.85
mAP@0.65 = 0.81
mAP@0.70 = 0.74
mAP@0.75 = 0.65  ← stricter
mAP@0.80 = 0.55
mAP@0.85 = 0.42
mAP@0.90 = 0.31
mAP@0.95 = 0.18  ← very strict, nearly perfect boxes needed

mAP@0.5-0.95 = average of all above = 0.63
```

**In simple words:**
- mAP@0.5 = "Did the model roughly find the disease? (box just needs to cover half)"
- mAP@0.5-0.95 = "Did the model find it precisely? (box needs to be very tight too)"

mAP@0.5-0.95 is harder to score high on because it also tests box accuracy, not just detection.

---

### 0.12 Training Flow — How the Model Learned

```
TRAINING LOOP (runs for N epochs, e.g., 100 rounds):

┌─────────────────────────────────────────────────────────────┐
│ STEP 1: Feed a batch of images (e.g., 16 at once)           │
│         → Model makes predictions (boxes + labels)          │
│                                                             │
│ STEP 2: Compare predictions to real labels                  │
│         → Calculate LOSS (how wrong the predictions are)    │
│         Loss = Box loss + Class loss + Confidence loss       │
│                                                             │
│ STEP 3: Backpropagation                                     │
│         → Send the error BACKWARDS through all layers       │
│         → Compute gradient for every weight (how much did   │
│           each weight contribute to the error?)             │
│                                                             │
│ STEP 4: Optimizer (Adam/SGD) updates weights                │
│         new_weight = old_weight - (learning_rate × gradient)│
│         → Weights shift slightly in the direction that      │
│           reduces the error                                 │
│                                                             │
│ STEP 5: Repeat with next batch                              │
│         → After all batches = 1 epoch done                  │
│                                                             │
│ STEP 6: After each epoch, test on validation images         │
│         → Calculate mAP@0.5 on validation set               │
│         → If best so far → save as best.pt                  │
│                                                             │
│ STEP 7: Repeat all steps for N epochs                       │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
   best.pt ← the saved model with highest validation mAP
```

**What is a Learning Rate?**
A number (e.g., 0.01) that controls how big each weight update step is.
- Too large → model jumps around and never settles
- Too small → model learns very slowly
- YOLOv8 uses a **learning rate scheduler** — starts higher, gradually reduces it

**What is a Loss Function?**
It measures how wrong the model is. Three parts for YOLO:
- **Box Loss** → how far off are the predicted box corners from the real corners?
- **Class Loss** → did it predict the correct disease label?
- **Confidence Loss** → was its certainty score appropriate?

Training = slowly minimizing all three losses at the same time.

---

## 1. What This Project Does

Think of it like a **doctor for guava plants**.

A farmer takes a photo of a guava leaf and uploads it to our website. Our AI looks at the photo and tells:
- **Which disease is present** (e.g., Anthracnose, Wilt, etc.)
- **How bad the disease is** (shown as a percentage — we call it Pathogen Load)
- **Where exactly on the leaf the disease is** (it draws a box around it)
- **Why the AI made that decision** (shown as a colored heatmap — red means highly infected area)
- **What treatment to use** (shown on screen and also in a downloadable PDF report)

The farmer does not need any expert. Just a phone and internet connection.

---

## 2. Why Guava? Why AI?

### Why Guava?
- Guava is a very common fruit crop grown in India, Brazil, and many tropical countries.
- It gets infected by diseases like Anthracnose and Wilt quite often.
- If diseases are not caught early, the entire harvest can be lost.
- Currently, farmers need to call an expert to diagnose diseases — that is slow and expensive.

### Why AI?
- The old way was to write rules like "if the leaf is yellow, it has nutrient deficiency" — but this fails when lighting changes or the disease looks slightly different.
- AI (deep learning) learns from thousands of example photos and figures out the patterns by itself.
- Once trained, it can check a photo in under a second.

---

## 3. How the Project is Structured

```
User opens the website and uploads a photo
               |
               v
        app.py  (the website)
               |
               v
       inference.py  (the AI does its work here)
       |                          |
       v                          v
  YOLOv8 model             Grad-CAM heatmap
  finds diseases            shows why AI decided
  + measures severity
               |
               v
   Results shown on website:
   - Marked image
   - Disease name + severity %
   - Treatment info
   - PDF download button
```

All of this happens in just 2 main Python files:
- **`inference.py`** — the AI brain
- **`app.py`** — the website and display

---

## 4. Each Part Explained

### 4.1 Dataset and Labelling

- We collected photos of guava leaves — both healthy and diseased.
- Each photo was **manually labelled** — a person drew boxes around every disease spot and wrote the disease name.
- These labelled boxes are called **Oriented Bounding Boxes (OBB)** — a rectangle that can be rotated to fit the disease patch tightly.
- The labelling tool used was **Roboflow** — a website that makes labelling images easy.
- The dataset was exported in a format that YOLOv8 understands.

### 4.2 The AI Model (`best.pt`)

- `best.pt` is our trained AI model — a file that stores all the "knowledge" the AI learned from labelled photos.
- It is based on **YOLOv8 OBB** — a very popular and fast object detection AI.
- "Object detection" means the AI finds objects in a photo and draws boxes around them.
- After training, the model that performed best on test photos was saved as `best.pt`.

### 4.3 `inference.py` — Where the AI Runs

This file has three jobs:

#### Job 1 — Disease Knowledge Bank (`disease_info`)
A Python dictionary (like a table) that stores information about each disease:
- What it is, what causes it, what damage it does, how to treat it, organic options, prevention tips, and future advice.

#### Job 2 — `run_yolo()` function — Finding Disease and Measuring Severity

Step by step:

```
1. Take the uploaded photo as a pixel grid (image array)
2. Pass it to the YOLOv8 model
3. Model gives back:
   - Boxes (rotated polygons) around disease spots
   - Disease label for each box (e.g., "Anthracnose")
   - Confidence score (how sure it is — e.g., 0.87 means 87% sure)
4. Remove any box where confidence is below 45% (too uncertain)
5. For each good box:
   - Draw the polygon on the image
   - Paint those pixels white in a "disease map"
6. Find the leaf area separately:
   - Convert image to HSV color format
   - Green pixels = leaf → make a "leaf map"
   - Keep only the biggest green area (removes background noise)
7. Calculate Severity:
   Severity % = (Pixels that are diseased AND on the leaf) / (Total leaf pixels) x 100
8. Decide severity level:
   - Below 5%   = Healthy
   - 5 to 20%   = Mild
   - 20 to 50%  = Moderate
   - Above 50%  = Severe
```

#### Job 3 — `run_gradcam()` function — Showing the AI's Thinking

Creates a colored heatmap over disease areas. Explained in Section 4.6.

### 4.4 `app.py` — The Website

Built using **Streamlit** — a Python tool that lets you build websites without knowing HTML or JavaScript.

What the website has:
- **Left sidebar**: Choose mode (Standard or Grad-CAM), upload photos
- **Main area**: Analyzed image on the left, disease info and PDF button on the right
- **Dark green theme** with an animated scanning line (for style)
- **When no photo is uploaded**: Shows a background nature image with "NEURAL CORE IDLE" text

### 4.5 PDF Report

When you click "Download Report", a PDF is created automatically using a Python library called **fpdf2**.

The PDF contains:
1. The analyzed leaf photo
2. Severity percentage, severity level, and date/time
3. For each detected disease: description, cause, damage, treatment, organic option, prevention, and future advice

We use the **DejaVu** font so all characters display properly in the PDF.

### 4.6 Grad-CAM — Showing Why the AI Decided

**Why we need it:**
AI models give answers but don't explain why. A farmer won't trust an AI that says "this leaf is diseased" without showing any reason. Grad-CAM creates a visual explanation.

**What it looks like:**
A heatmap is drawn over the leaf:
- **Red/warm colours** = the AI focused heavily here
- **Blue/cool colours** = the AI ignored this area

**How it works (simple version):**
1. While the AI analyzes the image, we secretly record what the middle layers of the AI are producing. These are called **activations** — basically numbers that represent what the AI sees at that layer.
2. We run the result backwards through the AI to see which middle-layer activations caused the final answer. These backwards values are called **gradients**.
3. High activation + high gradient = very important area for the decision.
4. We colour these areas (red = important, blue = not important) and blend it over the original image.
5. We only show the heatmap inside the detected disease boxes so it doesn't distract from the rest of the leaf.

**In one line:** Grad-CAM shows the farmer exactly which part of the leaf made the AI say "this is diseased."

---

## 5. Step-by-Step: What Happens When You Upload a Photo

```
STEP 1: You upload a .jpg or .png photo in the browser

STEP 2: app.py reads the file and converts it to a pixel array

STEP 3: inference.py passes it to YOLOv8
         → Finds disease regions, draws boxes, collects disease names

STEP 4: Severity is calculated
         → What % of the leaf is covered by disease

STEP 5: (If Grad-CAM mode is selected)
         → Heatmap is generated and blended on the image

STEP 6: app.py shows results:
         → Left side: the marked leaf image
         → Right side: severity score, disease info cards, PDF download

STEP 7: You click Download Report
         → PDF is created and saved to your device
```

---

## 6. Disease Classes

| Disease | Type | What Causes It | How It Looks on Leaf |
|---|---|---|---|
| **Anthracnose** | Fungal | Fungus called *Colletotrichum* | Dark brown sunken spots |
| **Nutrient Deficiency** | Lack of nutrition | Missing nitrogen, iron, or zinc in soil | Leaf turns yellow or pale |
| **Wilt** | Fungal/soil | Fungus called *Fusarium* in soil | Leaf droops, edges turn brown |
| **Insect Attack** | Pest damage | Aphids, fruit flies, mealybugs | Holes, curling, sticky patches |
| **Healthy** | — | Good soil, good care | Fully green, normal look |

---

## 7. How Severity is Measured

We measure what percentage of the leaf is covered by disease.

**Simple formula:**

```
Severity (%) = (Number of diseased leaf pixels / Total leaf pixels) x 100
```

**Severity levels:**

| Level | Severity % | Colour | What to Do |
|---|---|---|---|
| Healthy | Less than 5% | Green | No action needed |
| Mild | 5% to 20% | Yellow | Watch it, optional organic spray |
| Moderate | 20% to 50% | Orange | Apply fungicide or treatment now |
| Severe | More than 50% | Red | Act immediately, heavy treatment |

**How the leaf is separated from the background:**
We convert the image to HSV color format. Pixels with green colour (hue between 25 and 90) are the leaf. We then keep only the biggest green shape to ignore small green objects in the background.

---

## 8. Tools and Libraries We Used

| Tool | What It Does | Why We Chose It |
|---|---|---|
| **YOLOv8 OBB** | Detects diseases and draws boxes | Fast, accurate, supports rotated boxes |
| **PyTorch** | Runs the deep learning model | YOLOv8 is built on PyTorch |
| **OpenCV** | Reads/edits images, draws shapes | The standard image tools library in Python |
| **Streamlit** | Builds the website | Pure Python, very quick to build, no HTML needed |
| **Grad-CAM library** | Creates the explanation heatmap | Works directly with our PyTorch model |
| **fpdf2** | Creates PDF reports | Pure Python, supports custom fonts |
| **NumPy** | Handles pixel number arrays | Core math tool for Python |
| **Pillow (PIL)** | Saves images for the PDF | Needed to put photos inside fpdf2 PDFs |

---

## 9. How to Run the Project

```bash
# Step 1: Activate the virtual environment (Windows)
.\venv\Scripts\Activate.ps1

# Step 2: Install all required libraries (only needed once)
pip install -r requirements.txt

# Step 3: Start the app
streamlit run app.py
```

The website opens at: `http://localhost:8501`

- Upload one or more guava leaf images (.jpg / .png)
- Choose mode: Standard (just detection) or Enhanced XAI (with Grad-CAM heatmap)
- View results and download the PDF

---

## 10. Questions Your Guide May Ask — With Simple Answers

### 10.1 Basic Questions

**Q: What is the main goal of your project?**
To automatically find diseases in guava leaves from a photo and tell how severe they are — without needing any expert. The farmer uploads a photo and gets an instant diagnosis with treatment advice.

**Q: Which diseases does your system detect?**
Five types: Anthracnose, Nutrient Deficiency, Wilt, Insect Attack, and Healthy.

**Q: What is Pathogen Load?**
It is the percentage of the leaf that is covered by disease. For example, if 30% of the leaf has disease spots, the pathogen load is 30%.

**Q: What is the real-world use of this project?**
A farmer with a smartphone can take a photo of a sick leaf, upload it to the website, and get an instant result — which disease it is, how bad it is, and what to do. This saves time and money, and helps them act before the entire crop is lost.

**Q: What is the difference between image classification and object detection?**
- Classification just says: "This leaf has Anthracnose." (one answer for the whole image)
- Detection says: "There is Anthracnose at this exact location in the image." (finds and marks the position)

We use detection because we need to know where the disease is to measure how much of the leaf it covers.

---

### 10.2 AI Model Questions

**Q: What is YOLO?**
YOLO means "You Only Look Once." It is an AI model that detects objects in photos. Unlike older methods that scan an image piece by piece, YOLO looks at the whole image in one go. This makes it very fast.

**Q: What is YOLOv8 OBB? How is it different from normal YOLO?**
Normal YOLO draws straight horizontal rectangles. OBB (Oriented Bounding Box) draws **tilted rectangles** that can rotate to match the angle of the object. Disease spots on leaves are often at an angle, so tilted boxes fit them much better.

**Q: What is Grad-CAM and why did you use it?**
Grad-CAM makes the AI explain its decision visually. It highlights which parts of the leaf the AI was focusing on when it said "this is diseased." We use it so farmers and researchers can trust the result — they can see the AI is looking at the right spot.

**Q: What is a confidence score and why do we use a threshold of 0.45?**
A confidence score is how sure the AI is about a detection — from 0 (not sure at all) to 1 (100% sure). We only keep detections above 0.45 (45% confident). Below that, the detection is probably a mistake. 0.45 is a balance — not too strict (which would miss real diseases) and not too loose (which would give false alarms).

**Q: What is the difference between classification, detection, and segmentation?**
- **Classification**: "This is Anthracnose." (whole image, one answer)
- **Detection**: "Anthracnose is here (bounding box) with 87% confidence." (locates it)
- **Segmentation**: "These exact pixels are Anthracnose." (pixel-by-pixel)

Our project does detection with rotated boxes.

---

### 10.3 Code / Working Questions

**Q: How does the severity calculation work?**
1. YOLO finds disease regions — we fill them as white pixels in a disease map.
2. We find the leaf by checking which pixels are green (using HSV colors) — leaf map.
3. We count pixels that are both in the disease map AND in the leaf map.
4. Severity = (those pixels / total leaf pixels) × 100.

**Q: Why do you use HSV color format instead of RGB to find the leaf?**
In normal RGB, the same green color can look brighter or darker depending on lighting. In HSV format, the color (called Hue) stays in the same number range (25 to 90 for green) no matter how bright or dark the lighting is. So it is easier and more reliable to find the green leaf using HSV.

**Q: What does `cv2.fillPoly` do?**
It fills the inside of a polygon shape with a color. We use it to fill the disease region (detected by YOLO as a rotated box) with white in our disease map. This lets us count how many pixels are inside the disease region.

**Q: How does the PDF report get generated?**
After analysis, the `generate_detailed_pdf()` function creates a PDF using the `fpdf2` library. It adds the leaf image, severity percent, severity level, timestamp, and for every detected disease: description, cause, damage, treatment, organic option, prevention, and future advice. The PDF is then made available as a download button on the website.

**Q: What does `model.eval()` do?**
It tells the AI model to switch to testing mode. In this mode the model does not learn or update itself, and certain things like Dropout (which randomly turns off parts of the network during training) are turned off. This makes predictions stable.

**Q: Why switch to training mode during Grad-CAM?**
Grad-CAM needs to run calculations backwards through the model (to find gradients). This sometimes does not work in eval mode. So we briefly switch to training mode only for Grad-CAM, then switch back. We use a `try/finally` block to make sure the switch-back always happens even if there is an error.

---

### 10.4 Dataset and Training Questions

**Q: How did you collect and label the dataset?**
We gathered photos of guava leaves showing different diseases and healthy conditions. Using **Roboflow**, we drew rotated bounding boxes around each disease spot in every image and gave them labels. The final dataset was exported in YOLOv8 OBB format (text files with 8 coordinates per box).

**Q: What is data augmentation?**
It means creating extra training images by making small changes to the original ones — like flipping the image, rotating it, making it brighter or darker, zooming in, etc. This gives the model more variety to learn from and stops it from just memorizing the original photos. YOLOv8 applies augmentation automatically.

**Q: What metrics were used to check model quality?**
- **Precision**: Out of all boxes the model drew, what % were correctly placed on real diseases?
- **Recall**: Out of all real disease boxes in the test photos, what % did the model find?
- **mAP@50**: The main accuracy number. Checks both precision and recall together. The "50" means the predicted box must overlap the real box by at least 50% to count as correct.

**Q: What is IoU?**
IoU stands for Intersection over Union. It measures how much the predicted box overlaps the actual labelled box. 
```
IoU = Overlap area / Total area covered by both boxes
```
IoU = 1.0 means perfect match. IoU = 0 means no overlap. We need at least 0.5 IoU for a detection to count as correct.

**Q: What is overfitting and how did you prevent it?**
Overfitting is when the model does very well on training photos but fails on new ones — like a student who memorizes answers without understanding. We prevented it by:
- Using data augmentation (more variety in training)
- Saving `best.pt` — the model with the best score on photos it had never seen during training
- Using a confidence threshold to filter uncertain predictions

---

### 10.5 Real-World Use Questions

**Q: Can the system detect multiple diseases in one leaf?**
Yes. YOLO can draw multiple boxes of different types in one photo. All detected diseases are collected, shown as separate info cards on screen, and included in the PDF report.

**Q: Can this work on a mobile phone?**
Currently it is a website that runs on a computer. For mobile use, the model could be converted to a lighter format (ONNX or TFLite) and built into a mobile app. Detection takes less than half a second so it would work well on modern phones.

**Q: What if someone uploads a photo that is not a guava leaf?**
The model only knows guava leaf diseases. If a non-leaf image is uploaded, the model will likely find nothing (low confidence, filtered out). The system will default to "Healthy." We could add a step to check if the image is a leaf before running detection.

**Q: How would a farmer in a rural area use this?**
If there is internet, they open the website on their phone, upload a leaf photo, and see the result in seconds. The PDF can be downloaded and shared with agriculture officers. No technical knowledge needed — just take a photo and upload.

**Q: What are the limitations of your project?**
- Works only for guava leaves, not other crops
- Needs a clear, well-lit photo for best accuracy
- The Grad-CAM shows one heatmap for all classes combined, not separately per disease
- No live video/camera support yet
- Model accuracy improves with more training images — we are limited by dataset size

---

### 10.6 Harder Questions

**Q: What is the difference between Grad-CAM and LIME?**
Both explain why an AI made a decision, but they work differently.
- **Grad-CAM**: Goes inside the AI and uses the internal math (gradients). Very fast. Gives a smooth heatmap.
- **LIME**: Does not go inside the AI. It hides parts of the image and checks how the answer changes. Slower but works with any AI type.

We used Grad-CAM because it is faster and works directly with our PyTorch model.

**Q: Why is OBB better than normal bounding boxes for disease spots?**
Normal boxes are always drawn straight (horizontal). Disease spots on leaves are often diagonal or oddly shaped. A tilted OBB box can align with the disease shape, fitting it more tightly. This means less healthy leaf is included in the box, making the severity calculation more accurate.

**Q: What are PyTorch hooks and how do you use them for Grad-CAM?**
Hooks are like listeners attached to a layer in the AI. They run automatically when data passes through.
- **Forward hook**: Captures the layer's output when data moves forward through the model.
- **Backward hook**: Captures the gradients when the error is sent backwards through the model.

We attach these hooks to a middle layer of YOLO so Grad-CAM can read the activations and gradients it needs.

**Q: Why do we apply ReLU in Grad-CAM?**
ReLU removes all negative numbers (sets them to zero). In Grad-CAM, negative values mean those areas worked against the disease prediction. We only want to show areas that supported the decision (positive values). ReLU removes the negatives, leaving a clean heatmap of only the areas that pointed towards the disease.

**Q: How would you improve this project in the future?**
- Collect more labelled images from different farms, seasons, and lighting
- Extend it to other crops like mango, tomato, or wheat
- Use pixel-level marking (segmentation) instead of boxes for more exact area measurement
- Create separate heatmaps for each disease class
- Build a proper mobile app
- Add a history tracker so farmers can monitor leaf health over time

**Q: What is the difference between `best.pt` and `last.pt`?**
During training, the model is tested on unseen photos after every round (epoch).
- `best.pt` = saved from the round with the highest accuracy on test photos. Most reliable.
- `last.pt` = saved from the final training round, which may have started overfitting.

We always use `best.pt` for real predictions.

---

*This guide was written in plain, simple language for easy reading and confident presentation.*
