"# Cancer_Detection_with_xai" 

# Image Preprocessing Short Notes

## 1. Model Input Shape

Deep learning models expect:

```txt id="6jxyc9"
(batch_size, height, width, channels)
```

Example:

```txt id="44c7wn"
(1, 224, 224, 3)
```

Meaning:

* `1` → number of images (batch size)
* `224,224` → image size
* `3` → RGB channels

---

# 2. `np.expand_dims(img_array, axis=0)`

Adds a new dimension at the beginning.

```python id="py2ycw"
img_array = np.expand_dims(img_array, axis=0)
```

Before:

```txt id="qz9q0k"
(224,224,3)
```

After:

```txt id="jmu8fp"
(1,224,224,3)
```

`axis=0` means:

```txt id="it1u8s"
Add dimension at start
```

Used to create batch dimension.

---

# 3. `astype("float32")`

Converts datatype.

```python id="km5e0v"
img_array.astype("float32")
```

Images are usually:

```txt id="vls0pi"
uint8
```

Meaning:

* Unsigned integer
* Pixel range: `0–255`

`float32` is used because:

* Faster
* Less memory
* GPU optimized

---

# 4. `/255.0`

Normalizes pixel values.

```python id="hf87k4"
img_array / 255.0
```

Before:

```txt id="nch0lx"
0 → 255
```

After:

```txt id="g7nt3u"
0.0 → 1.0
```

Benefits:

* Faster convergence
* Stable gradients
* Better predictions

---

# Final Process

```python id="egbjlwm"
img_array = np.expand_dims(img_array, axis=0)

img_array = img_array.astype("float32") / 255.0
```

Final Output:

```txt id="2jlwm8"
Shape  : (1,224,224,3)
Type   : float32
Range  : 0.0 → 1.0
```


Here’s an upgraded version of the prompt with:

* example input
* function input parameters
* dynamic variables
* reusable architecture
* structured generation instructions
* better controllability for the LLM

This version is MUCH better for real-world AI code generation systems.

---

# MASTER PROMPT — DYNAMIC AI DOCUMENT GENERATOR

You are a senior Python engineer and expert document automation developer.

Your task is to generate COMPLETE production-ready Python code that creates professional human-readable documents/reports dynamically.

The generated code must use function-based architecture so users can pass dynamic inputs to generate different reports.

---

# PRIMARY OBJECTIVE

Generate Python code that:

1. Accepts dynamic input parameters
2. Creates professional DOCX/PDF reports
3. Uses modern formatting
4. Produces enterprise-grade human-readable output
5. Uses reusable modular functions
6. Is fully executable

---

# REQUIRED LIBRARIES

Use these libraries:

```python id="tvnq2j"
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

import io
import os
import datetime
import textwrap
import logging
```

Optional libraries if required:

```python id="ab32fr"
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
```

Optional PDF support:

```python id="ib96t4"
from reportlab.platypus import *
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
```

---

# CODE ARCHITECTURE REQUIREMENTS

The generated code MUST contain:

## 1. Main Report Function

Example:

```python id="qwl7si"
def generate_report(
    title,
    patient_name,
    report_type,
    summary,
    findings,
    confidence_score,
    table_data=None,
    image_paths=None,
    output_path="output.docx"
):
```

The function should dynamically generate the report based on the provided inputs.

---

# REQUIRED FEATURES

The generated document should support:

* Dynamic titles
* Dynamic sections
* Dynamic paragraphs
* Dynamic tables
* Dynamic images
* Dynamic metadata
* Dynamic branding
* Dynamic colors/styles
* Optional charts
* Optional footer
* Optional disclaimer

---

# DOCUMENT DESIGN REQUIREMENTS

The generated report should include:

## Header

* Branding
* Report title
* Date/time
* Metadata

## Body

* Executive summary
* Structured sections
* Tables
* Images
* Charts (optional)

## Footer

* Disclaimer
* Generated timestamp
* Branding

---

# STYLING REQUIREMENTS

Use professional formatting:

* Proper font hierarchy
* Clean spacing
* Modern alignment
* Table styling
* Color themes
* Section spacing
* Premium enterprise-style layouts

Preferred aesthetics:

* Medical reports
* Financial reports
* AI analytics reports
* Executive summaries

---

# FUNCTIONAL REQUIREMENTS

The code MUST:

1. Be production-ready
2. Include imports
3. Include helper functions
4. Include comments
5. Include error handling
6. Save output automatically
7. Support reusable templates
8. Use clean architecture
9. Avoid hardcoded values
10. Use dynamic variables

---

# EXAMPLE USER INPUT

Example runtime usage:

```python id="k8bj68"
generate_report(
    title="AI Diagnostic Report",
    patient_name="John Doe",
    report_type="Thyroid Cancer Detection",
    summary="The model predicts benign classification.",
    findings=[
        "No malignant patterns detected",
        "Confidence score is high",
        "GradCAM indicates low-risk regions"
    ],
    confidence_score=96.45,
    table_data=[
        ["Metric", "Value"],
        ["Prediction", "Benign"],
        ["Confidence", "96.45%"]
    ],
    image_paths=[
        "scan.png",
        "gradcam.png"
    ],
    output_path="diagnostic_report.docx"
)
```

---

# EXPECTED OUTPUT

The generated code should create:

* A professionally formatted DOCX report
* Human-readable layout
* Styled tables
* Proper headings
* Embedded images
* Metadata
* Footer/disclaimer

The final report should look enterprise-grade.

---

# IMPORTANT ENGINEERING RULES

DO:

* Generate COMPLETE executable Python code
* Use modular reusable functions
* Use helper functions
* Use proper naming conventions
* Use clean formatting
* Add comments where useful

DO NOT:

* Generate pseudo-code
* Skip imports
* Leave TODO placeholders
* Return incomplete snippets
* Use markdown formatting in final code
* Use unsafe shell execution

---

# SECURITY RULES

Avoid:

* subprocess shell execution
* unsafe eval/exec
* arbitrary file deletion
* dangerous OS commands

---

# OUTPUT FORMAT RULES

Return ONLY valid Python code.

No explanations.

No markdown.

No commentary.

---

# USER REQUIREMENTS

{USER_REQUIREMENTS}

---

# DOCUMENT CONTEXT

{DOCUMENT_CONTEXT}

---

# FINAL TASK

Generate the complete production-ready Python implementation now.
