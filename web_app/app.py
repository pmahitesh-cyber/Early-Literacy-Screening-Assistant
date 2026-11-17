"""
FastAPI Server for Teacher-Facing Literacy Screening Assistant
Serves interactive educator dashboard and REST APIs for audio scoring & triage.
"""

import os
import sys
import shutil
import tempfile
import json
import pandas as pd
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.inference import LiteracyScorer

app = FastAPI(
    title="Early Literacy Screening Assistant",
    description="Automated Literacy Screener for Primary Classrooms (DrivenData Goodnight Moon challenge)",
    version="2.0.0"
)

# Directories
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "templates")
DATA_DIR = os.path.join(BASE_DIR, "data")
AUDIO_DIR = os.path.join(DATA_DIR, "audio")
PRESETS_DIR = os.path.join(DATA_DIR, "presets")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
if os.path.exists(AUDIO_DIR):
    app.mount("/audio", StaticFiles(directory=AUDIO_DIR), name="audio")
if os.path.exists(PRESETS_DIR):
    app.mount("/presets", StaticFiles(directory=PRESETS_DIR), name="presets")

templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Initialize inference scorer
scorer = LiteracyScorer(model_path=os.path.join(BASE_DIR, "models", "literacy_screener_models.joblib"))


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render teacher-facing classroom screening dashboard."""
    cohort_path = os.path.join(REPORTS_DIR, "screening_summary.csv")
    total_screened = 0
    on_track_count = 0
    borderline_count = 0
    flagged_count = 0
    recent_students = []
    
    if os.path.exists(cohort_path):
        df_cohort = pd.read_csv(cohort_path)
        total_screened = len(df_cohort)
        counts = df_cohort['screening_status'].value_counts().to_dict()
        on_track_count = counts.get("Proficient", 0)
        borderline_count = counts.get("Monitor / Borderline", 0)
        flagged_count = counts.get("Flagged for Teacher Review", 0)
        recent_students = df_cohort.head(20).to_dict(orient="records")
        
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "total_screened": total_screened,
            "on_track_count": on_track_count,
            "borderline_count": borderline_count,
            "flagged_count": flagged_count,
            "recent_students": recent_students
        }
    )


@app.post("/api/score")
async def score_audio_clip(
    file: UploadFile = File(...),
    student_id: str = Form("student_001"),
    grade: str = Form("1"),
    task: str = Form("blending"),
    expected_text: Optional[str] = Form("sun")
):
    """
    Score uploaded or recorded audio clip and return diagnostic triage report.
    """
    try:
        suffix = os.path.splitext(file.filename)[1] if file.filename else ".wav"
        if not suffix:
            suffix = ".wav"
            
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            temp_path = tmp.name
            
        result = scorer.score_audio(
            audio_source=temp_path,
            task=task,
            grade=grade,
            student_id=student_id,
            expected_text=expected_text or "sun"
        )
        
        try:
            os.remove(temp_path)
        except Exception:
            pass
            
        return JSONResponse(content=result)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/presets")
async def get_test_presets():
    """Return all 8 real-time case test presets."""
    presets = [
        {
            "id": "case_blending_fluent",
            "name": "Case 1: Phoneme Blending (/s/-/u/-/n/) — Fluent Reader",
            "task": "blending",
            "expected_text": "sun",
            "grade": "1",
            "student_id": "student_012_fluent",
            "audio_url": "/presets/case_blending_fluent.wav",
            "expected_outcome": "Proficient / On-Track",
            "badge": "success",
            "description": "Child hears /s/-/u/-/n/ and blends smoothly into 'sun' within 1.8s. Short pause, stable vocal formant transitions."
        },
        {
            "id": "case_blending_struggling",
            "name": "Case 2: Phoneme Blending (/s/-/u/-/n/) — Struggling Reader",
            "task": "blending",
            "expected_text": "sun",
            "grade": "1",
            "student_id": "student_015_struggling",
            "audio_url": "/presets/case_blending_struggling.wav",
            "expected_outcome": "Flagged for Teacher Review",
            "badge": "danger",
            "description": "Child exhibits 40% hesitation silence, prolonged stall, and mispronounced blending. Immediate Tier-2 alert."
        },
        {
            "id": "case_deletion_struggling",
            "name": "Case 3: Phoneme Deletion ('cupcake' without 'cup') — Struggling",
            "task": "deletion",
            "expected_text": "cake",
            "grade": "1",
            "student_id": "student_009_flagged",
            "audio_url": "/presets/case_deletion_struggling.wav",
            "expected_outcome": "Flagged for Teacher Review",
            "badge": "danger",
            "description": "Child struggles to isolate base morpheme, repeating initial segment before stalling out. High hesitation."
        },
        {
            "id": "case_deletion_fluent",
            "name": "Case 4: Phoneme Deletion ('cupcake' without 'cup') — Fluent",
            "task": "deletion",
            "expected_text": "cake",
            "grade": "1",
            "student_id": "student_003_fluent",
            "audio_url": "/presets/case_deletion_fluent.wav",
            "expected_outcome": "Proficient / On-Track",
            "badge": "success",
            "description": "Immediate accurate elision ('cake') within 1.7s. Clean articulation and prompt onset."
        },
        {
            "id": "case_nonword_borderline",
            "name": "Case 5: Nonword Repetition ('plip') — Borderline Reader",
            "task": "nonword_repetition",
            "expected_text": "plip",
            "grade": "KG",
            "student_id": "student_021_monitor",
            "audio_url": "/presets/case_nonword_borderline.wav",
            "expected_outcome": "Borderline / Needs Monitoring",
            "badge": "warning",
            "description": "Kindergarten student hesitates on initial consonant cluster (/pl-/) before pronouncing. Borderline monitoring."
        },
        {
            "id": "case_sentence_fluent",
            "name": "Case 6: Sentence Repetition — Proficient Reader",
            "task": "sentence_repetition",
            "expected_text": "The dog ran fast",
            "grade": "2",
            "student_id": "student_007_proficient",
            "audio_url": "/presets/case_sentence_fluent.wav",
            "expected_outcome": "Proficient / On-Track",
            "badge": "success",
            "description": "Grade 2 child fluently echoes full sentence with clear prosodic rhythm and natural cadence."
        },
        {
            "id": "case_sentence_struggling",
            "name": "Case 7: Sentence Repetition — Reading Difficulty",
            "task": "sentence_repetition",
            "expected_text": "The dog ran fast",
            "grade": "2",
            "student_id": "student_018_dyslexia_risk",
            "audio_url": "/presets/case_sentence_struggling.wav",
            "expected_outcome": "Flagged for Teacher Review",
            "badge": "danger",
            "description": "Auditory working memory bottleneck: child pauses for 1.2s, stammers, and truncates the predicate."
        }
    ]
    return JSONResponse(content=presets)


@app.get("/api/metrics")
async def get_benchmark_metrics():
    """Return model evaluation metrics on held-out speakers."""
    metrics_path = os.path.join(REPORTS_DIR, "evaluation_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            data = json.load(f)
        return JSONResponse(content=data)
    else:
        raise HTTPException(status_code=404, detail="Metrics file not found. Run train.py first.")


@app.get("/api/sample_clips")
async def get_sample_clips():
    """Return list of sample clips for UI testing."""
    meta_path = os.path.join(DATA_DIR, "train_metadata.csv")
    labels_path = os.path.join(DATA_DIR, "train_labels.csv")
    
    if os.path.exists(meta_path) and os.path.exists(labels_path):
        df_m = pd.read_csv(meta_path)
        df_l = pd.read_csv(labels_path)
        df = pd.merge(df_m, df_l, on="filename")
        
        fluent_samples = df[df['score'] == 1].head(4).to_dict(orient="records")
        struggling_samples = df[df['score'] == 0].head(4).to_dict(orient="records")
        
        return JSONResponse(content={
            "fluent_samples": fluent_samples,
            "struggling_samples": struggling_samples
        })
    return JSONResponse(content={"fluent_samples": [], "struggling_samples": []})


@app.get("/api/cohort")
async def get_cohort_data():
    """Return full classroom screening cohort dataset."""
    cohort_path = os.path.join(REPORTS_DIR, "screening_summary.csv")
    if os.path.exists(cohort_path):
        df_cohort = pd.read_csv(cohort_path)
        return JSONResponse(content=df_cohort.to_dict(orient="records"))
    return JSONResponse(content=[])


@app.get("/api/export_cohort")
async def export_cohort():
    """Download screening summary CSV report."""
    cohort_path = os.path.join(REPORTS_DIR, "screening_summary.csv")
    if os.path.exists(cohort_path):
        return FileResponse(cohort_path, media_type="text/csv", filename="early_literacy_screening_report.csv")
    raise HTTPException(status_code=404, detail="Cohort report not found.")


if __name__ == "__main__":
    import uvicorn
    print("Starting Early Literacy Screening Assistant Server on http://127.0.0.1:8000 ...")
    uvicorn.run("web_app.app:app", host="127.0.0.1", port=8000, reload=False)
