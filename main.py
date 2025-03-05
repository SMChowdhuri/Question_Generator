
import os
import json
import pandas as pd
import traceback

!export GOOGLE_API_KEY="AIzaSyCTygHY6vGWlaRkG2-jIKi9_nRXkIqvBTM"

!pip install -q langchain-google-genai
!pip install --upgrade -q langchain-google-genai
!pip show langchain-google-genai
!pip install -q google-generativeai

import google.generativeai as genai
genai.configure(api_key="AIzaSyCTygHY6vGWlaRkG2-jIKi9_nRXkIqvBTM")
for model in genai.list_models():
    print(model.name)

from langchain_google_genai import ChatGoogleGenerativeAI
os.environ["GOOGLE_API_KEY"] = "AIzaSyCTygHY6vGWlaRkG2-jIKi9_nRXkIqvBTM"
# Create an instance of the LLM, using the 'gemini-pro' model with a specified creativity level
llm = ChatGoogleGenerativeAI(model='gemini-pro', temperature=0.9)

llm

response = llm.invoke('Write a paragraph about life on Mars in year 2100.')
print(response.content)

!pip install langchain_community

from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
from langchain.callbacks import get_openai_callback
# import PyPDF2

RESPONSE_JSON = {
    "1": {
        "mcq": "multiple choice question",
        "options": {
            "a": "choice here",
            "b": "choice here",
            "c": "choice here",
            "d": "choice here",
        },

        "correct": "correct answer",
    },
    "2": {
        "mcq": "multiple choice question",
        "options": {
            "a": "choice here",
            "b": "choice here",
            "c": "choice here",
            "d": "choice here",
        },
        "correct": "correct answer",
    },
    "3": {
        "mcq": "multiple choice question",
        "options": {
            "a": "choice here",
            "b": "choice here",
            "c": "choice here",
            "d": "choice here",
        },
        "correct": "correct answer",
    },
}

TEMPLATE="""
Text:{text}
You are an expert MCQ maker. Given the above text, it is your job to \
create a quiz  of {number} multiple choice questions for {subject} students in {tone} tone.
Make sure the questions are not repeated and check all the questions to be conforming the text as well.
Make sure to format your response like  RESPONSE_JSON below  and use it as a guide. \
Ensure to make {number} MCQs
{response_json}
"""

quiz_generation_prompt = PromptTemplate(
    input_variables=["text", "number", "subject", "tone", "response_json"],
    template=TEMPLATE
    )

quiz_chain=LLMChain(llm=llm, prompt=quiz_generation_prompt, output_key="quiz", verbose=True)

TEMPLATE2="""
You are an expert english grammarian and writer. Given a Multiple Choice Quiz for {subject} students.\
You need to evaluate the complexity of the question and give a complete analysis of the quiz. Only use at max 50 words for complexity analysis.
if the quiz is not at per with the cognitive and analytical abilities of the students,\
update the quiz questions which needs to be changed and change the tone such that it perfectly fits the student abilities
Quiz_MCQs:
{quiz}
###RESPONSE
Check from an expert English Writer of the above quiz:
"""

quiz_evaluation_prompt=PromptTemplate(input_variables=["subject", "quiz"], template=TEMPLATE)

review_chain=LLMChain(llm=llm, prompt=quiz_evaluation_prompt, output_key="review", verbose=True)

generate_evaluate_chain=SequentialChain(chains=[quiz_chain, review_chain], input_variables=["text", "number", "subject", "tone", "response_json"],
                                        output_variables=["quiz", "review"], verbose=True,)

# generate_evaluate_chain = SequentialChain(
#     chains=[quiz_chain, review_chain],  # Use quiz_chain instead of quiz_evaluation_prompt
#     input_variables=["text", "number", "subject", "tone", "response_json"],
#     output_variables=["quiz", "review"],
#     verbose=True,
# )

# file_path=r"/content/content.text"

# file_path

# with open(file_path, 'r') as file:
    # TEXT = file.read()

!pip install PyPDF2

import PyPDF2

file_path = r"/content/biophotonics_sample_3.2.pdf"

with open(file_path, 'rb') as file:  # Open in binary read mode ('rb')
    pdf_reader = PyPDF2.PdfReader(file)

    TEXT = ""
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        TEXT += page.extract_text()  # Extract text from each page

print(TEXT)  # Check the extracted content

# !pip install PyPDF2 langchain tiktoken google-ai-generativelanguage

# import PyPDF2
# import json
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.chat_models import ChatVertexAI
# from langchain.callbacks import get_openai_callback
# import time

# # Assuming you have defined generate_evaluate_chain and other necessary functions

# file_path = r"/content/Introduction to Biophotonics-1-157.pdf"

# with open(file_path, 'rb') as file:  # Open in binary read mode ('rb')
#     pdf_reader = PyPDF2.PdfReader(file)

#     TEXT = ""
#     for page_num in range(len(pdf_reader.pages)):
#         page = pdf_reader.pages[page_num]
#         TEXT += page.extract_text()  # Extract text from each page

# # Split the text into smaller chunks
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
# text_chunks = text_splitter.split_text(TEXT)

# print(TEXT)

# Serialize the Python dictionary into a JSON-formatted string
json.dumps(RESPONSE_JSON)

NUMBER=6
SUBJECT="Bio-Photonics"
TONE="hard"

# results = []
# for chunk in text_chunks:
#     with get_openai_callback() as cb:
#         response = generate_evaluate_chain(
#             {
#                 "text": chunk,
#                 "number": NUMBER,
#                 "subject": SUBJECT,
#                 "tone": TONE,
#                 "response_json": json.dumps(RESPONSE_JSON)
#             }
#         )
#         results.append(response)
#     time.sleep(1)

#https://python.langchain.com/docs/modules/model_io/llms/token_usage_tracking

#How to setup Token Usage Tracking in LangChain
with get_openai_callback() as cb:
    response=generate_evaluate_chain(
        {
            "text": TEXT,
            "number": NUMBER,
            "subject":SUBJECT,
            "tone": TONE,
            "response_json": json.dumps(RESPONSE_JSON)
        }
        )

print(f"Total Tokens:{cb.total_tokens}")
print(f"Prompt Tokens:{cb.prompt_tokens}")
print(f"Completion Tokens:{cb.completion_tokens}")
print(f"Total Cost:{cb.total_cost}")

response

quiz=response.get("quiz")
print(quiz)

# quiz = json.loads(RESPONSE_JSON)

# # Process the data into a table-like format
# quiz_table_data = []
# for key, value in quiz.items():
#     mcq = value.get("mcq", "No Question Provided")
#     options = value.get("options", {})
#     options_str = " | ".join(
#         [f"{opt}: {opt_val}" for opt, opt_val in options.items()]
#     )
#     correct = value.get("correct", "No Answer Provided")
#     quiz_table_data.append({"MCQ": mcq, "Choices": options_str, "Correct": correct})

# # Output the formatted data
# for item in quiz_table_data:
#     print(item)

# # quiz=json.loads(quiz)
# import json
# quiz=json.loads(quiz)
# quiz_table_data = []
# for key, value in quiz.items():
#     mcq = value["mcq"]
#     options = " | ".join(
#         [
#             f"{option}: {option_value}"
#             for option, option_value in value["options"].items()
#             ]
#         )
#     correct = value["correct"]
#     quiz_table_data.append({"MCQ": mcq, "Choices": options, "Correct": correct})

import json
import re

# Extract JSON data from the quiz string
match = re.search(r'\{(.*)\}', quiz, re.DOTALL)

if match:
    json_data = match.group(1)
    # Decode the extracted JSON data
    quiz = json.loads("{" + json_data + "}")

    quiz_table_data = []
    for key, value in quiz.items():
        mcq = value["mcq"]
        options = " | ".join(
            [
                f"{option}: {option_value}"
                for option, option_value in value["options"].items()
            ]
        )
        correct = value["correct"]
        quiz_table_data.append({"MCQ": mcq, "Choices": options, "Correct": correct})
else:
    print("Could not extract JSON data from quiz string.")

quiz_table_data

quiz=pd.DataFrame(quiz_table_data)

quiz.to_csv("Sample_3.2.csv",index=False)

# Moddel accuracy and Other Fact prediction

from datetime import datetime
datetime.now().strftime('%m_%d_%Y_%H_%M_%S')

import matplotlib.pyplot as plt

complexities = {"Easy": 8, "Medium": 6, "Hard": 6}  # Replace with actual counts from analysis
plt.pie(complexities.values(), labels=complexities.keys(), autopct='%1.1f%%', startangle=140, colors=['#7fc97f', '#beaed4', '#fdc086'])
plt.title("MCQ Difficulty Distribution")
plt.show()

mcq_data = {"Category 1": [5, 1], "Category 2": [6, 2]}  # [Correct, Incorrect]
categories = list(mcq_data.keys())
correct = [data[0] for data in mcq_data.values()]
incorrect = [data[1] for data in mcq_data.values()]

plt.bar(categories, correct, label='Correct', color='green')
plt.bar(categories, incorrect, bottom=correct, label='Incorrect', color='red')
plt.xlabel("Categories")
plt.ylabel("Number of Questions")
plt.title("Quiz Performance by Category")
plt.legend()
plt.show()

import matplotlib.pyplot as plt

tokens = {"Prompt Tokens": cb.prompt_tokens, "Completion Tokens": cb.completion_tokens, "Total Tokens": cb.total_tokens}
plt.bar(tokens.keys(), tokens.values(), color='skyblue')
plt.xlabel("Token Type")
plt.ylabel("Token Count")
plt.title("Token Usage Breakdown")
plt.show()

correct_answers = ['a', 'b', 'c', 'a', 'd', 'a', 'c']  # Replace with actual data
plt.hist(correct_answers, bins=4, color='purple', rwidth=0.8)
plt.xlabel("Correct Options")
plt.ylabel("Frequency")
plt.title("Correct Answer Distribution")
plt.show()

import re
import matplotlib.pyplot as plt

# Define text_chunks (replace with your actual text data)
# Example:
text_chunks = [
    "This is the first chunk of text.",
    "Here's the second chunk, with some more words.",
    "And finally, the third chunk."
]

word_count = [len(re.findall(r'\w+', chunk)) for chunk in text_chunks]
plt.plot(range(len(word_count)), word_count, marker='o', color='orange')
plt.xlabel("Chunk Number")
plt.ylabel("Word Count")
plt.title("Word Count Per Text Chunk")
plt.grid(True)
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Sample complexity data for quiz questions
complexity_analysis = [
    "simple", "medium", "hard", "medium", "simple",
    "simple", "hard", "medium", "hard", "medium",
    "simple", "medium", "medium", "hard", "simple"
]

# Count the occurrences of each complexity level
complexity_counts = Counter(complexity_analysis)

# Extract data for the graph
levels = list(complexity_counts.keys())  # Complexity levels (e.g., simple, medium, hard)
counts = list(complexity_counts.values())  # Counts for each level

# Plotting the complexity analysis graph
plt.figure(figsize=(8, 6))
sns.barplot(x=levels, y=counts, palette="viridis")

# Adding labels and title
plt.title("Quiz Complexity Analysis", fontsize=16)
plt.xlabel("Complexity Level", fontsize=14)
plt.ylabel("Number of Questions", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Show the graph
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()