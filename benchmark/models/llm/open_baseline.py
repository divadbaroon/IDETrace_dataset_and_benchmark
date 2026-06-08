"""
Open-source LLM baseline evaluation (Ollama).

Usage:
    python benchmark/models/llm/ollama_baseline.py --model llama3.1:8b
    python benchmark/models/llm/ollama_baseline.py --model qwen3.5:9b
    python benchmark/models/llm/ollama_baseline.py --model deepseek-r1:8b
"""

import argparse
import os
import time
import requests
from prompts import SYSTEM_PROMPT, run_all_tasks, save_and_print


def call_ollama(ollama_url, prompt, model, max_retries=3):
    api_url = ollama_url.replace('/v1', '').rstrip('/') + '/api/chat'
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "think": False,
        "options": {"temperature": 0.0, "num_predict": 50},
    }
    for attempt in range(max_retries):
        try:
            resp = requests.post(api_url, json=payload, timeout=120)
            resp.raise_for_status()
            content = resp.json().get('message', {}).get('content', '')
            return content.strip() if content else ''
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"    Failed after {max_retries} retries: {e}")
                return None


def main():
    parser = argparse.ArgumentParser(description='LLM Baseline (Ollama)')
    parser.add_argument('--model', default='llama3.1:8b')
    parser.add_argument('--root-dir', default='.')
    parser.add_argument('--sample-size', type=int, default=1000)
    parser.add_argument('--ollama-url', default='http://localhost:11434/v1')
    parser.add_argument('--output', default='benchmark/results/llm_open_results.json')
    parser.add_argument('--tasks', default='next_state,error_imminence,query_imminence,query_type')
    args = parser.parse_args()

    print("=" * 60)
    print(f"  LLM BASELINE (Ollama: {args.model})")
    print("=" * 60)

    call_fn = lambda prompt: call_ollama(args.ollama_url, prompt, args.model)
    results, test_deployments = run_all_tasks(call_fn, args)
    save_and_print(results, test_deployments, 'ollama', args.model,
                   os.path.join(args.root_dir, args.output))


if __name__ == '__main__':
    main()