

import json
import re
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import requests
import os

@dataclass
class LogItem:
    id: str
    system_prompt: str
    user_prompt: str
    response: str
    contexts: List[str] = None

class GroqRAGASCalculator:

    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    def _call_groq_api(self, prompt: str, max_retries: int = 3) -> str:
        for attempt in range(max_retries):
            try:
                payload = {
                    "model": self.model,
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "temperature": 0.1,
                    "max_tokens": 1024
                }

                response = requests.post(self.base_url, headers=self.headers, json=payload)

                if response.status_code == 200:
                    result = response.json()
                    return result["choices"][0]["message"]["content"].strip()
                elif response.status_code == 429:
                    wait_time = 2 ** attempt
                    print(f"Rate limited. Waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"API Error: {response.status_code} - {response.text}")
                    if attempt == max_retries - 1:
                        return "0.0"
                    time.sleep(1)

            except Exception as e:
                print(f"Exception in API call: {str(e)}")
                if attempt == max_retries - 1:
                    return "0.0"
                time.sleep(1)

        return "0.0"

    def calculate_faithfulness(self, resp: str, ctx: List[str]) -> float:
        if not ctx or not resp:
            return 0.0

        valid_ctx = [c.strip() for c in ctx if c and c.strip()]
        if not valid_ctx:
            return 0.0

        combined_ctx = "\n\n".join(valid_ctx)

        prompt = f"""
You are evaluating the faithfulness of an AI response to given contexts. Faithfulness measures how well the response is supported by the provided contexts.

CONTEXTS:
{combined_ctx}

RESPONSE:
{resp}

TASK: Rate the faithfulness of the response on a scale of 0.0 to 1.0, where:
- 1.0 = Response is fully supported by the contexts, no hallucinations
- 0.8 = Response is mostly supported, minor unsupported details
- 0.6 = Response is partially supported, some unsupported claims
- 0.4 = Response has significant unsupported content
- 0.2 = Response is minimally supported by contexts
- 0.0 = Response is not supported by contexts or contradicts them

Consider:
1. Are the facts in the response present in the contexts?
2. Are there any claims not supported by the contexts?
3. Does the response contradict any information in the contexts?

Respond with ONLY a decimal number between 0.0 and 1.0 (e.g., 0.75)
"""

        result = self._call_groq_api(prompt)
        try:
            score = float(result)
            return max(0.0, min(score, 1.0))
        except ValueError:
            nums = re.findall(r'\d+\.?\d*', result)
            if nums:
                try:
                    score = float(nums[0])
                    if score > 1.0:
                        score = score / 10
                    return max(0.0, min(score, 1.0))
                except ValueError:
                    pass
            return 0.0

    def calculate_answer_relevancy(self, resp: str, user_q: str) -> float:
        if not resp or not user_q:
            return 0.0

        prompt = f"""
You are evaluating how relevant an AI response is to a user's question. Answer relevancy measures how well the response addresses what the user asked.

USER QUESTION:
{user_q}

AI RESPONSE:
{resp}

TASK: Rate the answer relevancy on a scale of 0.0 to 1.0, where:
- 1.0 = Response directly and completely answers the question
- 0.8 = Response answers the question well with minor irrelevant details
- 0.6 = Response partially answers the question but misses some aspects
- 0.4 = Response addresses the question but with significant irrelevant content
- 0.2 = Response barely addresses the question
- 0.0 = Response doesn't address the question at all

Consider:
1. Does the response answer what was specifically asked?
2. Is the response focused on the user's question?
3. Are there irrelevant tangents or off-topic content?
4. Does the response provide the type of information the user was seeking?

Respond with ONLY a decimal number between 0.0 and 1.0 (e.g., 0.85)
"""

        result = self._call_groq_api(prompt)
        try:
            score = float(result)
            return max(0.0, min(score, 1.0))
        except ValueError:
            nums = re.findall(r'\d+\.?\d*', result)
            if nums:
                try:
                    score = float(nums[0])
                    if score > 1.0:
                        score = score / 10
                    return max(0.0, min(score, 1.0))
                except ValueError:
                    pass
            return 0.0

    def calculate_context_precision(self, ctx: List[str], user_q: str) -> float:
        if not ctx or not user_q:
            return 0.0

        valid_ctx = [c.strip() for c in ctx if c and c.strip()]
        if not valid_ctx:
            return 0.0

        if len(valid_ctx) > 1:
            scores = []
            for i, context in enumerate(valid_ctx):
                prompt = f"""
You are evaluating the precision of retrieved context for a user question. Context precision measures how relevant a piece of context is to answering the user's question.

USER QUESTION:
{user_q}

CONTEXT:
{context}

TASK: Rate the context precision on a scale of 0.0 to 1.0, where:
- 1.0 = Context is highly relevant and directly helps answer the question
- 0.8 = Context is relevant and mostly helpful for the question
- 0.6 = Context is somewhat relevant but partially helpful
- 0.4 = Context has some relevance but limited usefulness
- 0.2 = Context is minimally relevant to the question
- 0.0 = Context is irrelevant to the question

Consider:
1. Does the context contain information needed to answer the question?
2. How much of the context is relevant vs irrelevant to the question?
3. Would this context help someone answer the user's question?

Respond with ONLY a decimal number between 0.0 and 1.0 (e.g., 0.90)
"""

                result = self._call_groq_api(prompt)
                try:
                    score = float(result)
                    scores.append(max(0.0, min(score, 1.0)))
                except ValueError:
                    nums = re.findall(r'\d+\.?\d*', result)
                    if nums:
                        try:
                            score = float(nums[0])
                            if score > 1.0:
                                score = score / 10
                            scores.append(max(0.0, min(score, 1.0)))
                        except ValueError:
                            scores.append(0.0)
                    else:
                        scores.append(0.0)

                time.sleep(0.1)

            return sum(scores) / len(scores) if scores else 0.0

        else:
            context = valid_ctx[0]
            prompt = f"""
You are evaluating the precision of retrieved context for a user question. Context precision measures how relevant the context is to answering the user's question.

USER QUESTION:
{user_q}

CONTEXT:
{context}

TASK: Rate the context precision on a scale of 0.0 to 1.0, where:
- 1.0 = Context is highly relevant and directly helps answer the question
- 0.8 = Context is relevant and mostly helpful for the question
- 0.6 = Context is somewhat relevant but partially helpful
- 0.4 = Context has some relevance but limited usefulness
- 0.2 = Context is minimally relevant to the question
- 0.0 = Context is irrelevant to the question

Respond with ONLY a decimal number between 0.0 and 1.0 (e.g., 0.90)
"""

            result = self._call_groq_api(prompt)
            try:
                score = float(result)
                return max(0.0, min(score, 1.0))
            except ValueError:
                nums = re.findall(r'\d+\.?\d*', result)
                if nums:
                    try:
                        score = float(nums[0])
                        if score > 1.0:
                            score = score / 10
                        return max(0.0, min(score, 1.0))
                    except ValueError:
                        pass
                return 0.0

def parse_log_file(json_path: str) -> List[LogItem]:
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    items = []

    if isinstance(data, list):
        for dataset in data:
            if 'items' in dataset:
                for item in dataset['items']:
                    log_item = extract_log_item(item)
                    items.append(log_item)
    elif isinstance(data, dict) and 'items' in data:
        for item in data['items']:
            log_item = extract_log_item(item)
            items.append(log_item)
    else:
        raise ValueError("Unexpected JSON structure")

    return items

def extract_log_item(item: Dict) -> LogItem:
    sys_prompt = ""
    user_q = ""
    ctx = []

    if 'input' in item and isinstance(item['input'], list):
        for input_item in item['input']:
            if input_item.get('role') == 'system':
                sys_prompt = input_item.get('context', '')
            elif input_item.get('role') == 'user':
                user_q = input_item.get('context', '')

    resp = ""
    if 'expectedOutput' in item and isinstance(item['expectedOutput'], list):
        for output_item in item['expectedOutput']:
            if output_item.get('role') == 'assistant':
                resp = output_item.get('content', '')
                break

    if 'contexts' in item and item['contexts']:
        ctx = item['contexts']
    elif sys_prompt:
        ctx = extract_contexts_from_system_prompt(sys_prompt)

    if not ctx and sys_prompt:
        ctx = [sys_prompt]

    return LogItem(
        id=item.get('id', ''),
        system_prompt=sys_prompt,
        user_prompt=user_q,
        response=resp,
        contexts=ctx if ctx else []
    )

def extract_contexts_from_system_prompt(sys_prompt: str) -> List[str]:
    if not sys_prompt:
        return []

    ctx = []

    patterns = [
        r'Context:\s*(.*?)(?=\n\n|\nUser:|\nQuestion:|$)',
        r'Based on the following:\s*(.*?)(?=\n\n|\nUser:|\nQuestion:|$)',
        r'Given:\s*(.*?)(?=\n\n|\nUser:|\nQuestion:|$)',
        r'Information:\s*(.*?)(?=\n\n|\nUser:|\nQuestion:|$)',
        r'Reference:\s*(.*?)(?=\n\n|\nUser:|\nQuestion:|$)',
    ]

    for pattern in patterns:
        matches = re.findall(pattern, sys_prompt, re.DOTALL | re.IGNORECASE)
        ctx.extend([match.strip() for match in matches if match.strip()])

    if not ctx:
        chunks = re.split(r'\n\n+', sys_prompt)
        ctx = [chunk.strip() for chunk in chunks if chunk.strip() and len(chunk.strip()) > 50]

    return ctx[:5]

def compute_ragas_metrics_with_groq(json_path: str, groq_key: str, output_path: str = None) -> List[Dict[str, Any]]:
    items = parse_log_file(json_path)

    calc = GroqRAGASCalculator(groq_key, model="llama-3.3-70b-versatile")

    results = []
    total = len(items)

    print(f"Processing {total} items with Groq API...")
    print(f"Using model: {calc.model}")

    for i, item in enumerate(items, 1):
        print(f"\nProcessing item {i}/{total}: {item.id}")
        print(f"  Contexts: {len(item.contexts) if item.contexts else 0}")
        print(f"  User prompt length: {len(item.user_prompt) if item.user_prompt else 0}")
        print(f"  Response length: {len(item.response) if item.response else 0}")

        print("  Calculating faithfulness...")
        faith = calc.calculate_faithfulness(item.response, item.contexts)

        print("  Calculating answer relevancy...")
        rel = calc.calculate_answer_relevancy(item.response, item.user_prompt)

        print("  Calculating context precision...")
        prec = calc.calculate_context_precision(item.contexts, item.user_prompt)

        print(f"  Metrics: F={faith:.2f}, R={rel:.2f}, P={prec:.2f}")

        result = {
            "id": item.id,
            "faithfulness": round(faith, 2),
            "answer_relevancy": round(rel, 2),
            "context_precision": round(prec, 2)
        }

        results.append(result)

        time.sleep(0.5)

    if output_path:
        with open(output_path, 'w', encoding='utf-8') as file:
            json.dump(results, file, indent=2)

    return results

if __name__ == "__main__":
    GROQ_API_KEY = "YOUR GROQ_API_KEY"  # Replace with your actual Groq API key

    if not GROQ_API_KEY:
        print("Please set your Groq API key!")
        print("You can set it as an environment variable: export GROQ_API_KEY='your_key_here'")
        print("Or you can get one from: https://console.groq.com/")
        exit(1)

    input_file = "log.json"
    output_file = "ragas_output_groq.json"

    try:
        results = compute_ragas_metrics_with_groq(input_file, GROQ_API_KEY, output_file)

        print(f"\n{'='*60}")
        print("RAGAS Metrics Computation Completed!")
        print(f"{'='*60}")
        print(f"Processed {len(results)} log items")
        print(f"Results saved to: {output_file}")

        print("\nSample results:")
        for result in results[:3]:
            print(json.dumps(result, indent=2))

        if len(results) > 3:
            print(f"\n... and {len(results) - 3} more items")

        if results:
            avg_faith = sum(r['faithfulness'] for r in results) / len(results)
            avg_rel = sum(r['answer_relevancy'] for r in results) / len(results)
            avg_prec = sum(r['context_precision'] for r in results) / len(results)

            print(f"\n{'='*40}")
            print("SUMMARY STATISTICS")
            print(f"{'='*40}")
            print(f"Average Faithfulness:     {avg_faith:.3f}")
            print(f"Average Answer Relevancy: {avg_rel:.3f}")
            print(f"Average Context Precision: {avg_prec:.3f}")
            print(f"Overall RAGAS Score:      {(avg_faith + avg_rel + avg_prec) / 3:.3f}")

            print(f"\n{'='*40}")
            print("SCORE DISTRIBUTION")
            print(f"{'='*40}")

            for metric in ['faithfulness', 'answer_relevancy', 'context_precision']:
                scores = [r[metric] for r in results]
                high = len([s for s in scores if s >= 0.7])
                medium = len([s for s in scores if 0.4 <= s < 0.7])
                low = len([s for s in scores if s < 0.4])

                print(f"{metric.replace('_', ' ').title()}:")
                print(f"  High (≥0.7): {high} ({high/len(scores)*100:.1f}%)")
                print(f"  Medium (0.4-0.7): {medium} ({medium/len(scores)*100:.1f}%)")
                print(f"  Low (<0.4): {low} ({low/len(scores)*100:.1f}%)")

    except FileNotFoundError:
        print(f"Error: Could not find input file '{input_file}'")
        print("Please ensure the JSON log file exists and the path is correct.")
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()

def process_with_rate_limiting(items: List[LogItem], groq_key: str, rpm: int = 30) -> List[Dict[str, Any]]:
    calc = GroqRAGASCalculator(groq_key, model="llama-3.3-70b-versatile")
    results = []

    delay = 60 / rpm

    for i, item in enumerate(items):
        print(f"Processing item {i+1}/{len(items)}: {item.id}")

        faith = calc.calculate_faithfulness(item.response, item.contexts)
        time.sleep(delay)

        rel = calc.calculate_answer_relevancy(item.response, item.user_prompt)
        time.sleep(delay)

        prec = calc.calculate_context_precision(item.contexts, item.user_prompt)
        time.sleep(delay)

        result = {
            "id": item.id,
            "faithfulness": round(faith, 2),
            "answer_relevancy": round(rel, 2),
            "context_precision": round(prec, 2)
        }

        results.append(result)

        print(f"  Completed: F={faith:.2f}, R={rel:.2f}, P={prec:.2f}")

    return results

ALT_MODELS = [
    "llama-3.1-8b-instant",
    "llama3-70b-8192",
    "llama3-8b-8192",
    "gemma2-9b-it"
]

def test_model_availability(key: str, model: str) -> bool:
    calc = GroqRAGASCalculator(key, model=model)
    test_prompt = "Rate this on a scale of 0.0 to 1.0: This is a test. Answer with just: 0.5"

    result = calc._call_groq_api(test_prompt)
    return result != "0.0"

def find_working_model(key: str) -> str:
    models = ["llama-3.3-70b-versatile"] + ALT_MODELS

    for model in models:
        print(f"Testing model: {model}")
        if test_model_availability(key, model):
            print(f"✓ Model {model} is working")
            return model
        else:
            print(f"✗ Model {model} failed")

    raise Exception("No working models found. Please check your API key and model availability.")

