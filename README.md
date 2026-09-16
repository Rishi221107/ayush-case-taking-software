# AYUSH Clinical Intake Platform

An AI-powered, multimodal clinical intake and decision-support system designed for Ayurvedic hospitals and OPDs (e.g., AIIA). The platform combines vernacular voice intake, mathematical symptom entropy reduction, document digitization, and ABDM/FHIR standards to shorten patient intake from 15–20 minutes to under 3 minutes.

---

## Key Features

- **Vernacular Speech Intake (Sarvam AI)**: High-accuracy Speech-to-Text supporting Hindi, English, and regional accents with automatic clinical code-mixing support.
- **Dynamic Differential Triage Engine**: Mathematical information-gain algorithm parsing the `AyurGenixAI_Dataset` (446 conditions) to ask high-entropy follow-up questions.
- **Conversational SOCRATES Probing**: Context-aware clinical history-taking (Site, Onset, Character, Radiation, Associations, Timing, Exacerbating factors, Severity) powered by edge-quantized SLMs (Ollama / Qwen2.5-3B).
- **Document OCR & Digitization**: Extracts diagnostic findings, abnormal lab values, and past prescriptions from uploaded images or reports.
- **ABDM & FHIR R4 Native**: Generates standardized HL7 FHIR R4 clinical bundles (Encounter, Condition, Observation) ready for direct integration into Doctor OPD desks and ABHA systems.
- **DPDP Act Compliant**: Designed for local edge/on-premise deployment to ensure sensitive patient data remains within hospital intranet boundaries.

---

## Tech Stack

- **Backend**: Python 3.10+, FastAPI, Uvicorn, Pydantic
- **Data & Triage Engine**: Pandas, NumPy (`AyurGenixAI_Dataset.csv`)
- **Speech-to-Text**: Sarvam AI API (`saaras:v3`)
- **Clinical Probing SLM**: Ollama (`qwen2.5:3b` / `llama3.2:3b`)
- **Document Processing**: Pillow (PIL), OCR Engine
- **Frontend**: HTML5, Tailwind CSS, Vanilla JavaScript (Web Audio API)

---

## Project Structure

```text
├── AyurGenixAI_Dataset.csv   # Ayurvedic disease & symptom matrix (446 conditions)
├── main.py                   # FastAPI application entry point & API endpoints
├── triage_engine.py          # Differential triage & entropy evaluation engine
├── kiosk.html                # Patient-facing OPD voice intake kiosk interface
├── .env.example              # Template for local environment variables
└── README.md                 # Project documentation
