```
 _____ __  _______ ____      _     ____ _____ ___  
| ____|\ \/ /_   _|  _ \    / \   / ___|_   _/ _ \ 
|  _|   \  /  | | | |_) |  / _ \ | |     | || | | |
| |___  /  \  | | |  _ <  / ___ \| |___  | || |_| |
|_____|/_/\_\ |_| |_| \_\/_/   \_\\____| |_| \___/ 
```

**Turn stacks of PDFs into structured data. On your servers. In milliseconds.**

Legal teams get 500-page records dumps. Paralegals spend hours indexing them by hand. The big cloud tools from AWS, Azure, and GCP get it wrong constantly. Extracto consistently outperforms them. HIPAA compliant. Runs on your infrastructure. Your data never leaves your network.

**Live demo:** [tryextracto.com](https://www.tryextracto.com)


## What it does

Drop a mixed stack of PDFs. Extracto figures out what each form is, pulls every field, and gives you back clean structured data.

```
$ extracto auto ./case_martinez/ --output json

  ✓ CMS-1500   Martinez, Elena R · M54.2, G89.29 · $450.00      99%   340ms
  ✓ EOB        Anthem Health · Claim #A9924 · $698.00            100%  280ms
  ✓ PHQ-9      Rodriguez, Jamie · Score 18/27 · Moderately Severe 92%  190ms
  ✓ HIPAA      Martinez, Elena R · Auth valid 01/2025–01/2026    95%   150ms
  ✓ FROI       Chen, David W · L. Knee S83.511A · Workers' comp  87%  310ms

  5 documents · 5 types · 47 fields extracted · 1.27s total
```

**Split & Extract** — Upload one PDF with hundreds of mixed forms. Extracto splits it, classifies each one, extracts everything.

**Index Medical Records** — Upload a records bundle. Get it indexed by provider and date of service with clinical data extracted from each encounter.

**Query Everything** — All extracted data flows into a searchable SQLite database.

```python
from extracto.storage.db import ExtractoDB

db = ExtractoDB("extracto.db")
db.search_by_diagnosis("M54.2")
db.search_by_cpt("97110")
db.search_by_provider("Mitchell")
```


## Supported forms

| Type | What it extracts | Accuracy |
|---|---|---|
| CMS-1500 | Patient info, ICD-10 codes, service lines, charges, totals | 100% |
| EOB | Payer, claim/check numbers, financial tables, reason codes | 100% |
| PHQ-9 | 9-item depression scores, total, severity level | 100% |
| HIPAA Auth | Patient info, date ranges, excluded categories | 100% |
| FROI / DWC-1 | Employee/employer, injury details, body parts with laterality | 100% |
| Medical Intake | Demographics, allergies, symptoms, claim type | 99.9% |
| Insurance Claim | Work-related, auto accident, coverage details | 100% |
| Any document | Key-value pairs, tables, dates, entities, ICD-10 codes | Extracted |

Accuracy figures are for digital PDFs. Scanned documents: 60-86% depending on form type.


## Why not a cloud service?

| | Extracto | AWS Textract | Google Doc AI | Azure Doc Intel |
|---|---|---|---|---|
| CMS-1500 extraction | Specialized | Generic OCR | Generic OCR | Generic OCR |
| ICD-10 / CPT parsing | Built in | No | No | No |
| Multi-form splitting | Automatic | No | Manual | No |
| Records indexing | Built in | No | No | No |
| PHI leaves your network | Never | Always | Always | Always |
| BAA required | No | Yes | Yes | Yes |
| Per-page cost | Free | $0.01-0.06 | $0.01-0.10 | $0.01-0.05 |
| Works offline | Yes | No | No | No |


## Get started

```bash
git clone https://github.com/StuckInTheNet/Extracto.git
cd Extracto
pip install -e .

extracto auto inbox/ --out results/            # auto-classify and extract
extracto index records_bundle.pdf --out index/  # index medical records by provider
extracto extract form.pdf --overlays            # extract with visual overlays
extracto eval dataset/manifest.json             # evaluate accuracy
```

Or use the Python API:

```python
from extracto.pipeline.auto import auto_extract_single

result = auto_extract_single("claim.pdf")
print(result["classified_type"])              # "cms1500"
print(result["extraction"])                   # {"patient_name": "Martinez, Elena R", ...}
print(result["classification_confidence"])    # 0.99
```


## Web demo

Try it at [tryextracto.com](https://www.tryextracto.com) or run locally:

```bash
python -m extracto.web.app --dev
# http://localhost:8080
```

Drag-and-drop upload, split & extract pipelines, medical records indexing, data explorer with SQL queries, side-by-side PDF viewer.


## How it works

```
PDF Input
├─ Detection (6-tier fallback)
│  ├─ PDF-native vector drawings          12ms/page
│  ├─ YOLOv8 (trained on 2,400 images)    scanned docs
│  ├─ Mark detection (circled answers)     real-world forms
│  ├─ Position-based ink density           scanned checkboxes
│  ├─ AcroForm widgets                     fillable PDFs
│  └─ OpenCV + LR classifiers             final fallback
│
├─ Classification
│  └─ Multi-pattern anchor matching        8+ form types
│
├─ Extraction
│  ├─ 7 dedicated form extractors          spatial reasoning
│  └─ Generic extractor                    any document
│
├─ Post-processing (optional)
│  └─ LLM correction for low-confidence fields
│
├─ Storage
│  └─ SQLite with full-text search         zero config
│
└─ Output
   ├─ Structured JSON
   ├─ Searchable database
   ├─ Records index (provider x DOS)
   └─ Web UI with PDF viewer
```


## HIPAA compliance

Extracto runs on your infrastructure. Your servers, your network.

- No patient data touches cloud APIs or third-party processors
- No BAA required — there is no third-party processor
- PHI minimized in output — index references page numbers, not patient identifiers
- All test data in this repo is synthetic


## License

GPL-3.0. See [LICENSE](LICENSE) for details.
