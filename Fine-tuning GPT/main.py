import os
from openai import OpenAI

client = OpenAI()


client.files.create(
  file=open("train.jsonl", "rb"),
  purpose="fine-tune"
)

# print(client.files.list(limit=1))

client.fine_tuning.jobs.create(
  training_file="file-nf9ikjLiJRWkzx40j2g5o5Zs",
  model="gpt-3.5-turbo" #change to gpt-4-0613 if you have access
)

print(client.fine_tuning.jobs.list(limit=2))
# List 10 fine-tuning jobs
print(client.fine_tuning.jobs.retrieve("ftjob-Injev1HiX8xKUiP0RhU14j8P"))

completion = client.chat.completions.create(
  model="ft:gpt-3.5-turbo-0125:personal::AVATmxkI",
  messages=[
    {"role": "system", "content": "You are teaching assistant for Machine Learning. You should help to user to answer on his question."},
    {"role": "user", "content": "What is k-means clustering?"}
  ]
)
print(completion.choices[0].message)
