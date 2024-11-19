import os
import braintrust
from dataclasses import dataclass
from openai import AsyncOpenAI

braintrust.login(api_key=os.environ["BRAINTRUST_API_KEY"])
client = braintrust.wrap_openai(AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"]))

import duckdb

# DuckDB has an easy wrapper for loading datasets from Hugging Face.
con = duckdb.connect(":memory:")
full_result = con.query("""
    SELECT * FROM 'hf://datasets/stanfordnlp/coqa/data/validation-00000-of-00001.parquet'
        LIMIT 40
""").fetchall()

single_result = full_result[10]

# print("Source:")
# print(single_result[0])

# print("Passage:")
# print(single_result[1])

# print("\nQuestion:")
# print(single_result[2][0])

# print("\nAnswer:")
# print(single_result[3]["input_text"][0])



@dataclass
class QuestionAnswer:
    passage: str
    question: str
    expected_answer: str
    generated_answer: str


qa_pairs = [
    QuestionAnswer(
        passage=r[1],
        question=question,
        generated_answer=r[3]["input_text"][i],
        expected_answer=r[3]["input_text"][i],
    )
    for r in full_result
    for (i, question) in enumerate(r[2])
]

print(len(qa_pairs))