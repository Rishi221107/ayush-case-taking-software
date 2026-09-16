import os
from dotenv import load_dotenv
import re
import uuid
import shutil
import tempfile
import json
import requests
from datetime import datetime
from typing import List, Optional
from PIL import Image
from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from triage_engine import ClinicalDifferentialEngine

# 1. Initialize App & Engine
app = FastAPI(title="AYUSH Clinical Voice & Triage Platform")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pass dataset path directly here (looks for dataset.csv in root folder)
DATASET_FILE = "AyurGenixAI_Dataset.csv"
triage_engine = ClinicalDifferentialEngine(dataset_path=DATASET_FILE)

# 2. Configuration & Keys
load_dotenv()
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "")
SARVAM_STT_URL = "https://api.sarvam.ai/speech-to-text"

# 3. Models
class TriageSession(BaseModel):
    positive_symptoms: List[str]
    negative_symptoms: List[str]

# Dictionary mapping factor columns to English & Hindi spoken questions
FACTOR_PROMPTS = {
    "fever": {"en": "Do you have an elevated body temperature or chills?", "hi": "क्या आपको बुखार या ठंड लग रही है?"},
    "cough": {"en": "Are you experiencing a cough or throat irritation?", "hi": "क्या आपको खांसी या गले में खराश की समस्या है?"},
    "joint_pain": {"en": "Do you have body ache or severe joint pain?", "hi": "क्या आपके जोड़ों या बदन में तेज़ दर्द हो रहा है?"},
    "abdominal_pain": {"en": "Are you experiencing pain in your stomach or abdomen?", "hi": "क्या आपके पेट में दर्द महसूस हो रहा है?"},
    "burning_sensation": {"en": "Is there a burning sensation in your chest or stomach?", "hi": "क्या सीने या पेट में जलन महसूस हो रही है?"},
    "dyspnea": {"en": "Do you feel shortness of breath or difficulty breathing?", "hi": "क्या सांस लेने में तकलीफ या सांस फूलने की समस्या है?"},
    "fatigue": {"en": "Are you experiencing extreme exhaustion or weakness?", "hi": "क्या आपको बहुत अधिक थकान या कमज़ोरी लग रही है?"},
    "headache": {"en": "Do you have an intense or recurring headache?", "hi": "क्या आपके सिर में तेज़ दर्द बना हुआ है?"},
    "nausea": {"en": "Do you feel like vomiting or have general nausea?", "hi": "क्या आपको उल्टी आने जैसा या जी मिचलाने का अहसास है?"},
    "loss_of_appetite": {"en": "Have you noticed a sudden loss of appetite?", "hi": "क्या आपकी भूख कम हो गई है या खाना खाने का मन नहीं करता?"},
    "chest_pain": {"en": "Are you feeling pressure or tightness in your chest?", "hi": "क्या आपके सीने में भारीपन या दर्द है?"}
}

# ----------------- ENDPOINTS ----------------- #

@app.get("/")
def root():
    return {"status": "AYUSH Platform Engine Running", "dataset_loaded": DATASET_FILE}

# Route 1: Sarvam Speech Transcription & Red Flag Interception
@app.post("/api/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_audio:
        shutil.copyfileobj(file.file, temp_audio)
        temp_audio_path = temp_audio.name

    try:
        headers = {
            "api-subscription-key": SARVAM_API_KEY.strip(),
            "Authorization": f"Bearer {SARVAM_API_KEY.strip()}"
        }
        payload = {"model": "saaras:v3", "language_code": "unknown"}

        with open(temp_audio_path, "rb") as audio_file:
            files = {"file": (file.filename, audio_file, file.content_type or "audio/webm")}
            response = requests.post(SARVAM_STT_URL, headers=headers, data=payload, files=files, timeout=20)

        if response.status_code != 200:
            return {"status": "error", "message": f"Sarvam API error: {response.text}"}

        res_json = response.json()
        transcript = res_json.get("transcript", "").strip()
        lower_text = transcript.lower()

        # Emergency check
        emergency_keywords = ["chest pain", "heart attack", "can't breathe", "saans", "khoon", "blood"]
        is_emergency = any(kw in lower_text for kw in emergency_keywords)

        # Extract initial symptom matches against dataset factors
        detected_symptoms = []
        for factor in triage_engine.top_factors:
            if factor.replace("_", " ") in lower_text or factor in lower_text:
                detected_symptoms.append(factor)

        # Common vernacular alias mappings
        if not detected_symptoms:
            if "bukhar" in lower_text or "fever" in lower_text: detected_symptoms.append("fever")
            elif "khansi" in lower_text or "cough" in lower_text: detected_symptoms.append("cough")
            elif "dard" in lower_text or "pain" in lower_text: detected_symptoms.append("joint pain")
            elif "hand" in lower_text or "haath" in lower_text: detected_symptoms.append("joint pain")
            elif "pet" in lower_text or "stomach" in lower_text: detected_symptoms.append("abdominal pain")

        # Evaluate against the 446-disease dataset
        next_step = triage_engine.evaluate_state(detected_symptoms, [])

        return {
            "status": "success",
            "transcript": transcript,
            "is_emergency": is_emergency,
            "detected_symptoms": detected_symptoms,
            "triage_state": next_step
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}
    finally:
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

# Route 2: Dynamic Question Engine based on Dataset Entropy
@app.post("/api/triage/next-step")
async def next_triage_step(session: TriageSession):
    eval_result = triage_engine.evaluate_state(session.positive_symptoms, session.negative_symptoms)

    if eval_result["status"] == "continue":
        factor = eval_result["next_question_factor"]
        clean_factor = factor.lower().replace(" ", "_")

        q_data = FACTOR_PROMPTS.get(clean_factor, {
            "en": f"Are you experiencing {factor.replace('_', ' ')}?",
            "hi": f"क्या आपको {factor.replace('_', ' ')} की समस्या है?"
        })

        return {
            "status": "question",
            "factor_probe": clean_factor,
            "question_en": q_data["en"],
            "question_hi": q_data["hi"],
            "remaining_diseases": eval_result["remaining_candidates_count"]
        }
    else:
        fhir_bundle = build_fhir_bundle(session.positive_symptoms, eval_result["probable_diseases"])
        return {
            "status": "complete",
            "probable_diseases": eval_result["probable_diseases"],
            "fhir_bundle": fhir_bundle
        }

# Route 3: OCR & Document Extraction
@app.post("/api/scan-document")
async def scan_document(file: UploadFile = File(...)):
    try:
        image = Image.open(file.file)
        fname = file.filename.lower()

        # Dynamic entity parser simulation
        if "blood" in fname or "cbc" in fname or "lab" in fname:
            raw_text = "Diagnostic Lab: Fasting Glucose: 154 mg/dL (High). HbA1c: 7.9% (High). Normal renal parameters."
            extracted_diagnoses = ["Suspected Diabetes Mellitus"]
            abnormal_flags = ["Elevated Fasting Glucose (154 mg/dL)", "HbA1c > 6.5%"]
        else:
            raw_text = "Clinical Consultation Note: Advised Tab Paracetamol 650mg TDS, Tab Pantoprazole 40mg OD. Review in 3 days."
            extracted_diagnoses = ["Acute Febrile Illness"]
            abnormal_flags = ["Pyrexia of unknown origin"]

        return {
            "status": "success",
            "raw_text": raw_text,
            "extracted_diagnoses": extracted_diagnoses,
            "abnormal_findings": abnormal_flags,
            "document_type": "Clinical Record"
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

# Route 4: ABHA Record Gateway
@app.get("/api/abha/{abha_id}")
async def get_abha_record(abha_id: str):
    return {
        "status": "success",
        "data": {
            "abha_id": f"{abha_id}@abdm",
            "name": "Ramesh Chandra",
            "age": 48,
            "chronic_conditions": ["Type 2 Diabetes Mellitus", "Hypertension"],
            "current_medications": ["Metformin 500mg BD", "Amlodipine 5mg OD"],
            "recent_lab_records": [
                {"date": "2026-05-12", "test": "HbA1c", "result": "7.8%", "flag": "High"},
                {"date": "2026-05-12", "test": "Serum Creatinine", "result": "1.0 mg/dL", "flag": "Normal"}
            ]
        }
    }

# Helper: ABDM FHIR R4 Bundle Builder
def build_fhir_bundle(symptoms: list, probable_diagnoses: list) -> dict:
    bundle_id = str(uuid.uuid4())
    encounter_id = str(uuid.uuid4())

    entries = [
        {
            "resource": {
                "resourceType": "Encounter",
                "id": encounter_id,
                "status": "in-progress",
                "class": {"system": "http://terminology.hl7.org/CodeSystem/v3-ActCode", "code": "AMB", "display": "Ambulatory/OPD"},
                "subject": {"reference": "Patient/A-104", "display": "Ramesh Chandra"},
                "period": {"start": datetime.utcnow().isoformat() + "Z"}
            }
        }
    ]

    for diag in probable_diagnoses:
        entries.append({
            "resource": {
                "resourceType": "Condition",
                "id": str(uuid.uuid4()),
                "clinicalStatus": {"coding": [{"system": "http://terminology.hl7.org/CodeSystem/condition-clinical", "code": "provisional"}]},
                "code": {"text": diag},
                "subject": {"reference": "Patient/A-104"},
                "encounter": {"reference": f"Encounter/{encounter_id}"},
                "note": [{"text": f"Elicited via dynamic intake from factors: {', '.join(symptoms)}"}]
            }
        })

    return {
        "resourceType": "Bundle",
        "id": bundle_id,
        "type": "collection",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "entry": entries
    }


class ChatMessage(BaseModel):
    role: str # "user" or "assistant"
    content: str

class ConversationalIntakeRequest(BaseModel):
    history: List[ChatMessage]
    language: str = "en" # "en" or "hi"

SYSTEM_PROMPT_SOCRATES = """
You are an expert AI clinical intake assistant at an AYUSH Hospital OPD.
Your goal is to elicit a thorough medical history using the clinical SOCRATES framework:
- Site (exact location of symptom)
- Onset (when did it start, sudden or gradual)
- Character (sharp, dull, throbbing, burning)
- Radiation (does it spread anywhere)
- Associations (other accompanying symptoms like numbness, swelling, fever)
- Time course (constant, comes and goes)
- Exacerbating/relieving factors
- Severity (scale of 1-10)

Rules:
1. Speak warmly and empathetically like a doctor.
2. Ask ONE or TWO focused follow-up questions at a time. Never overwhelm the patient with a checklist.
3. If the patient says something vague like "pain in hand", clarify the exact location (wrist, fingers, palm, elbow), onset duration, and nature of the pain.
4. If language is 'hi', reply in clear, conversational Hindi/Hinglish. If 'en', reply in English.
5. ALWAYS respond in this exact JSON format:
{
  "next_question": "Your conversational question here",
  "extracted_symptoms": ["list", "of", "confirmed", "symptoms"],
  "socrates_slots": {
    "site": "...",
    "onset": "...",
    "character": "...",
    "severity": "..."
  },
  "is_intake_complete": false
}
"""

@app.post("/api/intake/chat")
async def conversational_intake(req: ConversationalIntakeRequest):
    # Format message history
    messages = [{"role": "system", "content": SYSTEM_PROMPT_SOCRATES}]
    for msg in req.history:
        messages.append({"role": msg.role, "content": msg.content})

    # OPTION A: Calling an active local SLM via Ollama (e.g., Qwen2.5-3B or Llama-3.2-3B)
    # If running Ollama locally:
    try:
        ollama_res = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": "qwen2.5:3b", # or llama3.2:3b
                "messages": messages,
                "format": "json",
                "stream": False
            },
            timeout=10
        )
        if ollama_res.status_code == 200:
            content = json.loads(ollama_res.json()["message"]["content"])
            return {"status": "success", "data": content}
    except Exception:
        pass

    # OPTION B: Fallback / Mock SLM logic if local SLM is not running yet
    # Generates dynamic SOCRATES follow-up based on the last message
    last_user_msg = req.history[-1].content.lower() if req.history else ""
    
    if "hand" in last_user_msg and ("pain" in last_user_msg or "dard" in last_user_msg):
        if req.language == "hi":
            question = "दर्द हाथ में किस जगह हो रहा है—कलाई (wrist), उंगलियों, या कोहनी में? और यह कब से है?"
        else:
            question = "Where specifically is the pain in your hand—in the wrist, fingers, or elbow? And since when did it start?"
        slots = {"site": "hand (unspecified)", "onset": "unknown", "character": "pain"}
    elif "fever" in last_user_msg or "bukhar" in last_user_msg:
        if req.language == "hi":
            question = "बुखार कितने दिनों से है? क्या साथ में ठंड या कंपकंपी भी महसूस हो रही है?"
        else:
            question = "How many days have you had this fever? Is it accompanied by chills or body shivering?"
        slots = {"symptom": "fever", "onset": "unknown", "character": "febrile"}
    else:
        if req.language == "hi":
            question = "यह समस्या कितने समय से है, और क्या आपको इसके साथ कोई अन्य लक्षण महसूस हो रहा है?"
        else:
            question = "How long have you been experiencing this, and are there any other accompanying symptoms?"
        slots = {"symptom": last_user_msg}

    return {
        "status": "success",
        "data": {
            "next_question": question,
            "extracted_symptoms": [last_user_msg],
            "socrates_slots": slots,
            "is_intake_complete": False
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)