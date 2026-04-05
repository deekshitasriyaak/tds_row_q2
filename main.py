import base64
import json
import os
import tempfile
import re
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import httpx

app = FastAPI()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")


async def transcribe_audio(audio_base64: str) -> str:
    """Transcribe audio using OpenAI Whisper API."""
    audio_bytes = base64.b64decode(audio_base64)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        tmp_path = f.name

    try:
        import openai
        client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY)
        with open(tmp_path, "rb") as audio_file:
            transcript = await client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language="ko"
            )
        return transcript.text
    finally:
        os.unlink(tmp_path)


async def parse_dataset_from_transcript(transcript: str) -> dict:
    """Use GPT to parse the transcript into structured dataset statistics."""
    import openai
    client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY)

    system_prompt = """You are a data analyst. The user will provide a Korean transcript describing a dataset.
Extract the dataset statistics and return ONLY a valid JSON object with these exact keys:
{
  "rows": <integer>,
  "columns": [<list of column name strings>],
  "mean": {<col>: <float>, ...},
  "std": {<col>: <float>, ...},
  "variance": {<col>: <float>, ...},
  "min": {<col>: <float>, ...},
  "max": {<col>: <float>, ...},
  "median": {<col>: <float>, ...},
  "mode": {<col>: <float or list>, ...},
  "range": {<col>: <float>, ...},
  "allowed_values": {<col>: [<list of allowed values>] or null, ...},
  "value_range": {<col>: [<min, max>] or null, ...},
  "correlation": [<list of correlation row arrays, numeric>]
}

Rules:
- Only include numeric columns in mean/std/variance/min/max/median/mode/range/correlation.
- For categorical columns, include them in allowed_values (list of valid values) and skip from numeric stats.
- value_range is [min, max] for numeric columns, null for categorical.
- allowed_values is a list for categorical columns, null for numeric.
- correlation is a 2D list (matrix) of correlation coefficients between numeric columns.
- Return ONLY valid JSON, no markdown, no explanation."""

    response = await client.chat.completions.create(
        model="gpt-4o",
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

        # Step 1: Transcribe
        transcript = await transcribe_audio(audio_base64)
        print(f"[{audio_id}] Transcript: {transcript[:200]}")

        # Step 2: Parse stats from transcript
        stats = await parse_dataset_from_transcript(transcript)
        print(f"[{audio_id}] Stats keys: {list(stats.keys())}")

        return JSONResponse(content=stats)

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


@app.get("/health")
async def health():
    return {"status": "ok"}
