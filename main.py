import os
import sys
import json
import subprocess

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI(title="Code Interpreter API")

# Enable CORS for browser-based testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CodeRequest(BaseModel):
    code: str


class ErrorAnalysis(BaseModel):
    error_lines: list[int]


class CodeResponse(BaseModel):
    error: list[int]
    result: str


def execute_python_code(code: str) -> dict:
    """Execute Python code in a separate process and capture its output."""

    runner = """
import sys
import traceback

source = sys.stdin.read()

try:
    exec(compile(source, "<student_code>", "exec"), {})
except BaseException:
    traceback.print_exc()
    sys.exit(1)
"""

    try:
        completed = subprocess.run(
            [sys.executable, "-I", "-c", runner],
            input=code,
            capture_output=True,
            text=True,
            timeout=5,
            cwd="/tmp",
            env={
                "PATH": os.environ.get("PATH", ""),
                "PYTHONDONTWRITEBYTECODE": "1",
            },
        )

        if completed.returncode == 0:
            return {
                "success": True,
                "output": completed.stdout,
            }

        return {
            "success": False,
            "output": completed.stderr,
        }

    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "output": "Execution timed out after 5 seconds.",
        }


def analyze_error_with_ai(code: str, traceback_text: str) -> list[int]:
    """Use AI Pipe to identify the failing student-code line numbers."""

    token = os.environ.get("AIPIPE_TOKEN")

    if not token:
        raise HTTPException(
            status_code=500,
            detail="AIPIPE_TOKEN environment variable is not configured.",
        )

    client = OpenAI(
        api_key=token,
        base_url="https://aipipe.org/openai/v1",
    )

    prompt = f"""
Analyze the Python execution error.

Identify the line number or line numbers in the submitted code
that caused the error. Count lines starting from 1.

Use the traceback to determine the location. Do not invent line
numbers or return line numbers belonging to this analysis service.

Return JSON in this exact format:
{{"error_lines": [3]}}

If no specific submitted-code line can be identified, return:
{{"error_lines": []}}

SUBMITTED CODE:
{code}

TRACEBACK:
{traceback_text}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a precise Python traceback analyzer. "
                        "Return only valid JSON with an error_lines array."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content

        if not content:
            raise ValueError("AI returned an empty response.")

        analysis = ErrorAnalysis.model_validate_json(content)

        # Keep unique, valid line numbers within the submitted code.
        line_count = len(code.splitlines())

        return sorted({
            line
            for line in analysis.error_lines
            if 1 <= line <= line_count
        })

    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"AI error analysis failed: {type(exc).__name__}",
        ) from exc


@app.post("/code-interpreter", response_model=CodeResponse)
def code_interpreter(request: CodeRequest):
    execution = execute_python_code(request.code)

    if execution["success"]:
        return CodeResponse(
            error=[],
            result=execution["output"],
        )

    error_lines = analyze_error_with_ai(
        request.code,
        execution["output"],
    )

    return CodeResponse(
        error=error_lines,
        result=execution["output"],
    )


@app.get("/")
def home():
    return {"message": "Code Interpreter API is running"}