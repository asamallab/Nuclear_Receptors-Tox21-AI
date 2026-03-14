"""Prompt-based molecular activity classification using a locally hosted
LLM (GGUF) via llama-cpp-python"""

import os
import sys
import time
import pickle
import json
import math
import pandas as pd
from llama_cpp import Llama

# ARGUMENT
if len(sys.argv) != 4:
    print("Usage: python3 model.py <model_path> <prompt_file> <output_dir>")
    sys.exit(1)

MODEL_PATH  = sys.argv[1]
PROMPT_FILE = sys.argv[2]
OUTPUT_DIR  = sys.argv[3]
THRESHOLD = 0.5

N_THREADS = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count()))
os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(PROMPT_FILE, "rb") as f:
    all_prompts = pickle.load(f)

# MODEL INFERENCE
llm = Llama(
    model_path=MODEL_PATH,
    n_threads=N_THREADS,
    n_ctx=2048,
    n_batch=2048,
    logits_all=True,
    verbose=False
)

results = []
start_total = time.time()

for i, (chem_id, prompt) in enumerate(all_prompts.items(), 1):

    llm.reset()
    clean_prompt = prompt.replace("<|begin_of_text|>", "", 1)

    output = llm.create_completion(
        clean_prompt,
        max_tokens=1,
        temperature=0.0,
        logprobs=50,
        echo=False
    )

    try:
        top_logprobs = output["choices"][0]["logprobs"]["top_logprobs"][0]
    except (KeyError, IndexError, TypeError):
        top_logprobs = {}

    lp_act_space     = top_logprobs.get(" active",   -100.0)
    lp_act_nospace   = top_logprobs.get("active",    -100.0)
    lp_inact_space   = top_logprobs.get(" inactive", -100.0)
    lp_inact_nospace = top_logprobs.get("inactive",  -100.0)

    prob_act_sum   = math.exp(lp_act_space) + math.exp(lp_act_nospace)
    prob_inact_sum = math.exp(lp_inact_space) + math.exp(lp_inact_nospace)
    total = prob_act_sum + prob_inact_sum

    if total == 0:
        prob_active = 0.0
        prob_inactive = 1.0
    else:
        prob_active   = prob_act_sum / total
        prob_inactive = prob_inact_sum / total

    pred_label = 1 if prob_active >= THRESHOLD else 0

    results.append({
        "dsstox_substance_id": chem_id,
        "prob_active": prob_active,
        "prob_inactive": prob_inactive,
        "pred_label": pred_label
    })

pred_df = pd.DataFrame(results)

pred_df.to_csv(os.path.join(OUTPUT_DIR, "predictions.csv"), index=False)
pred_df.to_json(os.path.join(OUTPUT_DIR, "predictions.json"), orient="records", indent=2)

metadata = {
    "model_path": MODEL_PATH,
    "threads": N_THREADS,
    "threshold": THRESHOLD,
    "num_prompts": len(all_prompts),
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
}

with open(os.path.join(OUTPUT_DIR, "run_metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)

print("Completed successfully")