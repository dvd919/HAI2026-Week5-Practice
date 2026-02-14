import os
import json
import pandas as pd
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import Optional
from movie_tool import get_tools, query_movie_db
from chart_tool import get_chart_tool, validate_chart

# ── API key: same pattern as app.py ──
try:
    import streamlit as st
    api_key = st.secrets["OPENAI_API_KEY"]
except Exception:
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.environ["OPENAI_API_KEY"]

client = OpenAI(api_key=api_key)

# ── Reasoning model (redefined locally to avoid importing agent_panel) ──
class Reasoning(BaseModel):
    reason: str = Field(description="Your reasoning about what you know so far and what to do next")
    use_tool: bool = Field(description="True if you need to run code or create a chart, False if you can give the final answer")
    answer: Optional[str] = Field(default=None, description="Your final answer in one short paragraph. Only provide when use_tool is False.")

# ── Load data ──
df = pd.read_csv("movies.csv")

# ── Helpers ──
passed = 0
failed = 0

def report(name, success, detail=""):
    global passed, failed
    tag = "PASS" if success else "FAIL"
    if success:
        passed += 1
    else:
        failed += 1
    print(f"[{tag}] {name}")
    if detail:
        print(f"       {detail}")

# ── Test 1: Structured Output (thinking phase) ──
print("=" * 60)
print("Test 1: Structured Output (parse with Reasoning model)")
print("=" * 60)
try:
    messages = [
        {"role": "system", "content": "You are a data analyst with access to a tool that executes Python code on a movie database."},
        {"role": "user", "content": "What is the average IMDB rating in the dataset?"},
    ]
    response = client.chat.completions.parse(
        model="gpt-4o-mini",
        messages=messages,
        response_format=Reasoning,
    )
    reasoning = response.choices[0].message.parsed
    print(f"  reason:   {reasoning.reason}")
    print(f"  use_tool: {reasoning.use_tool}")
    print(f"  answer:   {reasoning.answer}")
    report("Structured Output", reasoning is not None and isinstance(reasoning.reason, str))
except Exception as e:
    report("Structured Output", False, str(e))

# ── Test 2: Tool Calling (acting phase) ──
print()
print("=" * 60)
print("Test 2: Tool Calling (create with tools)")
print("=" * 60)
tool_call = None
try:
    tools = get_tools(df)
    tools.append(get_chart_tool())
    messages = [
        {"role": "system", "content": "You are a data analyst with access to a tool that executes Python code on a movie database."},
        {"role": "user", "content": "What is the average IMDB rating in the dataset?"},
        {"role": "assistant", "content": "I need to query the database to compute the average IMDB rating."},
    ]
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=tools,
        parallel_tool_calls=False,
    )
    msg = response.choices[0].message
    if msg.tool_calls:
        for tc in msg.tool_calls:
            print(f"  tool:  {tc.function.name}")
            print(f"  args:  {tc.function.arguments}")
            if tc.function.name == "QueryMovieDB":
                tool_call = tc
        report("Tool Calling", True, f"{len(msg.tool_calls)} tool call(s) returned")
    else:
        report("Tool Calling", False, "No tool calls returned")
except Exception as e:
    report("Tool Calling", False, str(e))

# ── Test 3: Tool Execution (end-to-end) ──
print()
print("=" * 60)
print("Test 3: Tool Execution (run QueryMovieDB end-to-end)")
print("=" * 60)
try:
    if tool_call is None:
        report("Tool Execution", False, "Skipped — no QueryMovieDB call from Test 2")
    else:
        args = json.loads(tool_call.function.arguments)
        result = query_movie_db(args["code"], df)
        print(f"  code:   {args['code']}")
        print(f"  result: {result.strip()}")
        report("Tool Execution", bool(result.strip()) and "Error" not in result)
except Exception as e:
    report("Tool Execution", False, str(e))

# ── Summary ──
print()
print("=" * 60)
total = passed + failed
print(f"Results: {passed}/{total} passed, {failed}/{total} failed")
print("=" * 60)
