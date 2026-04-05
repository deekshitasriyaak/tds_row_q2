import base64
import json
import os
import tempfile
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from groq import Groq

app = FastAPI()

client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))


async def transcribe_audio(audio_base64: str) -> str:
    audio_bytes = base64.b64decode(audio_base64)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        tmp_path = f.name

    try:
        with open(tmp_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                file=audio_file,
                model="whisper-large-v3",
                language="ko",
                response_format="text"
            )
        return transcription
    finally:
        os.unlink(tmp_path)


def parse_dataset_from_transcript(transcript: str) -> dict:
    system_prompt = """You are a data analyst. The user will provide a transcript (possibly in Korean) describing a dataset.
Extract the dataset statistics and return ONLY a valid JSON object with these exact keys:
{
  "rows": <integer, total number of rows>,
  "columns": [<list of all column name strings>],
  "mean": {<numeric_col>: <float>, ...},
  "std": {<numeric_col>: <float>, ...},
  "variance": {<numeric_col>: <float>, ...},
  "min": {<numeric_col>: <float>, ...},
  "max": {<numeric_col>: <float>, ...},
  "median": {<numeric_col>: <float>, ...},
  "mode": {<numeric_col>: <float>, ...},
  "range": {<numeric_col>: <float>, ...},
  "allowed_values": {<col>: [<list>] for categorical cols, null for numeric cols},
  "value_range": {<col>: [<min>, <max>] for numeric cols, null for categorical cols},
  "correlation": [[<float>, ...], ...]
}

Rules:
- mean/std/variance/min/max/median/mode/range: only numeric columns
- allowed_values: list of valid values for categorical columns, null for numeric
- value_range: [min, max] for numeric columns, null for categorical
- correlation: 2D matrix of Pearson correlation coefficients between numeric columns (same order as they appear in columns list)
- range = max - min for each numeric column
- Return ONLY raw JSON, no markdown, no code blocks, no explanation"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Transcript: {transcript}"}
        ],
        temperature=0,
        response_format={"type": "json_object"}
    )

    text = response.choices[0].message.content
    return json.loads(text)


@app.post("/")
async def analyze_audio(request: Request):
    try:
        body = await request.json()
        audio_id = body.get("audio_id", "unknown")
        audio_base64 = body.get("audio_base64", "")

        print(f"[{audio_id}] Transcribing...")
        transcript = await transcribe_audio(audio_base64)
        print(f"[{audio_id}] Transcript: {transcript[:300]}")

        print(f"[{audio_id}] Parsing stats...")
        stats = parse_dataset_from_transcript(transcript)
        print(f"[{audio_id}] Done. rows={stats.get('rows')}, cols={stats.get('columns')}")

        return JSONResponse(content=stats)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/health")
async def health():
    return {"status": "ok"}
