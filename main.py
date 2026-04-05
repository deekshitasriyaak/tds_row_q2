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
    system_prompt = """You are a data analyst. The user gives a transcript (possibly Korean) describing a dataset.

Return ONLY a valid JSON object. No markdown, no code blocks.

Schema:
{
  "rows": <int>,
  "columns": [<all column names in order>],
  "mean":     {<numeric_col>: <float>},
  "std":      {<numeric_col>: <float>},
  "variance": {<numeric_col>: <float>},
  "min":      {<numeric_col>: <float>},
  "max":      {<numeric_col>: <float>},
  "median":   {<numeric_col>: <float>},
  "mode":     {<numeric_col>: <float>},
  "range":    {<numeric_col>: <float>},
  "allowed_values": {<categorical_col>: [<values>]},
  "value_range":    {<numeric_col>: [<min>, <max>]},
  "correlation": [[<float>,...],...]
}

CRITICAL RULES:
- mean/std/variance/min/max/median/mode/range/value_range: ONLY numeric columns. Never include categorical columns here.
- allowed_values: ONLY categorical columns (non-numeric, e.g. gender, grade, type). If NO categorical columns exist, return "allowed_values": {}.
- Do NOT put numeric columns in allowed_values. Do NOT put categorical columns in mean/std/etc.
- value_range: [min, max] pair for each numeric column only.
- correlation: 2D matrix (list of lists) of Pearson correlations among numeric columns, in same order as they appear in "columns".
- range = max - min per numeric column.
- variance = std^2."""

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
    result = json.loads(text)

    # --- Post-processing safety net ---
    numeric_cols = list(result.get("mean", {}).keys())
    all_cols = result.get("columns", [])
    categorical_cols = [c for c in all_cols if c not in numeric_cols]

    # allowed_values: only categorical cols
    allowed_values = result.get("allowed_values", {})
    allowed_values = {k: v for k, v in allowed_values.items() if k in categorical_cols}
    result["allowed_values"] = allowed_values

    # value_range: only numeric cols
    value_range = result.get("value_range", {})
    value_range = {k: v for k, v in value_range.items() if k in numeric_cols}
    result["value_range"] = value_range

    # correlation must be [] when no numeric columns exist
    if len(numeric_cols) == 0:
        result["correlation"] = []
        for key in ("mean", "std", "variance", "min", "max", "median", "mode", "range"):
            result[key] = {}

    return result


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
        print(f"[{audio_id}] allowed_values={stats.get('allowed_values')}")
        print(f"[{audio_id}] value_range keys={list(stats.get('value_range', {}).keys())}")

        return JSONResponse(content=stats)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/health")
async def health():
    return {"status": "ok"}
