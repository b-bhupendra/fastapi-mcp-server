"""
Objective Extractor Module.
Specialized Dual-Ollama consensus engine that analyzes:
'What was this repository/project specifically trying to achieve?'
Extracts intended goals, current WIP state, blockers, and next steps.
"""

import os
import json
import urllib.request
import urllib.error
from typing import Dict, List, Any, Optional

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MODELS = ["qwen2.5:7b", "llama3.2:latest"]
TEMPERATURES = [0.0, 0.5, 0.9]

def call_ollama_generate(model: str, prompt: str, temperature: float) -> str:
    url = f"{OLLAMA_URL}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": 250
        }
    }
    try:
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=45) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("response", "").strip()
    except Exception:
        return ""

def call_ollama_embed(text: str) -> List[float]:
    url = f"{OLLAMA_URL}/api/embeddings"
    payload = {
        "model": "nomic-embed-text",
        "prompt": text[:2000]
    }
    try:
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("embedding", [0.0] * 768)
    except Exception:
        return [0.0] * 768

def extract_project_objective(ctx: Dict[str, Any], progress_cb=None) -> Dict[str, Any]:
    """Runs multi-temperature Dual-Ollama consensus specifically focused on project intention & goals."""
    prompt = f"""You are a senior software architect analyzing a developer's repository.
Analyze the files, manifests, and documentation to determine:
1. INTENDED OBJECTIVE: What problem is this project trying to solve? What was the developer trying to achieve?
2. CURRENT STATUS: Is this a POC, production app, library, or experiment?
3. NEXT STEPS: What would be the next 2 logical steps to complete or advance this project?

Project Name: {ctx['project_name']}
Language: {ctx['primary_language']}
Files: {', '.join(ctx.get('top_files', []))}
Tech Stack: {', '.join(ctx.get('tech_stack', []))}
README / Manifest Content:
{ctx.get('readme_preview', '')[:1200]}

Format your response exactly as:
OBJECTIVE: <2 concise sentences describing what this project is trying to achieve>
STATUS: <POC / In Development / Production / Learning Experiment>
NEXT_STEPS: <2 bullet points of recommended next steps>
"""

    responses = []
    total_passes = len(MODELS) * len(TEMPERATURES)
    current_pass = 0

    for model in MODELS:
        for temp in TEMPERATURES:
            current_pass += 1
            if progress_cb:
                progress_cb(current_pass, total_passes, f"Objective Analysis: {model} (T={temp})")
            out = call_ollama_generate(model, prompt, temp)
            if out and "OBJECTIVE:" in out:
                responses.append(out)

    if not responses:
        # Fallback heuristic
        return {
            "intended_objective": f"The project '{ctx['project_name']}' is designed to implement {ctx['primary_language']} functionality based on {', '.join(ctx.get('top_files', [])[:4])}.",
            "project_status": "In Development",
            "next_steps": ["Add comprehensive unit tests", "Finalize core entry point modules"],
            "raw_consensus": ""
        }

    # Best consensus selection based on token richness and structured adherence
    best_candidate = responses[0]
    best_score = -1
    for cand in responses:
        score = len(cand)
        if "OBJECTIVE:" in cand:
            score += 50
        if "STATUS:" in cand:
            score += 30
        if "NEXT_STEPS:" in cand:
            score += 30
        if score > best_score:
            best_score = score
            best_candidate = cand

    # Parse sections
    objective_text = ""
    status_text = "In Development"
    next_steps = []

    lines = best_candidate.splitlines()
    parsing_next_steps = False

    for line in lines:
        line_s = line.strip()
        if line_s.startswith("OBJECTIVE:"):
            objective_text = line_s.replace("OBJECTIVE:", "").strip()
            parsing_next_steps = False
        elif line_s.startswith("STATUS:"):
            status_text = line_s.replace("STATUS:", "").strip()
            parsing_next_steps = False
        elif line_s.startswith("NEXT_STEPS:"):
            parsing_next_steps = True
            step = line_s.replace("NEXT_STEPS:", "").strip()
            if step:
                next_steps.append(step)
        elif parsing_next_steps and (line_s.startswith("-") or line_s.startswith("*") or (len(line_s) > 2 and line_s[0].isdigit() and line_s[1] in [".", ")"])):
            clean_step = line_s.lstrip("-*0123456789. )").strip()
            if clean_step:
                next_steps.append(clean_step)

    if not objective_text:
        objective_text = best_candidate[:250].replace("\n", " ").strip()
    if not next_steps:
        next_steps = ["Verify dependencies and build", "Complete primary entry point"]

    return {
        "intended_objective": objective_text,
        "project_status": status_text,
        "next_steps": next_steps[:3],
        "raw_consensus": best_candidate
    }
