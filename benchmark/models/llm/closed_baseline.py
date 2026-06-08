"""
Closed-source LLM baseline evaluation (OpenAI).

Usage:
    python benchmark/models/llm/openai_baseline.py --model gpt-4o-mini
    python benchmark/models/llm/openai_baseline.py --model gpt-4o
    python benchmark/models/llm/openai_baseline.py --model gpt-5.5
"""

import argparse
import os
import time
from prompts import SYSTEM_PROMPT, run_all_tasks, save_and_print


def call_openai(client, prompt, model, max_retries=3):
    for attempt in range(max_retries):
        try:
            kwargs = {
                "model": model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            }
            if model.startswith("gpt-5"):
                kwargs["max_completion_tokens"] = 500
            else:
                kwargs["temperature"] = 0.0
                kwargs["max_tokens"] = 20

            response = client.chat.completions.create(**kwargs)
            return response.choices[0].message.content.strip()
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"    Failed after {max_retries} retries: {e}")
                return None


def main():
    parser = argparse.ArgumentParser(description='LLM Baseline (OpenAI)')
    parser.add_argument('--model', default='gpt-4o-mini')
    parser.add_argument('--root-dir', default='.')
    parser.add_argument('--sample-size', type=int, default=1000)
    parser.add_argument('--output', default='benchmark/results/llm_closed_results.json')
    parser.add_argument('--tasks', default='next_state,error_imminence,query_imminence,query_type')
    args = parser.parse_args()

    from dotenv import load_dotenv
    load_dotenv()
    from openai import OpenAI
    client = OpenAI()

    print("=" * 60)
    print(f"  LLM BASELINE (OpenAI: {args.model})")
    print("=" * 60)

    call_fn = lambda prompt: call_openai(client, prompt, args.model)
    results, test_deployments = run_all_tasks(call_fn, args)
    save_and_print(results, test_deployments, 'openai', args.model,
                   os.path.join(args.root_dir, args.output))


if __name__ == '__main__':
    main()