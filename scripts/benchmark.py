import json
import os
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

def get_openai_response(prompt, options, model="gpt-4.1", temperature=0.0):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Ќе ти биде поставено прашање и ќе ти бидат дадени опции. Ти треба да одговориш со буквата на опцијата која е точниот одговор, како A, B, C, или D."},
            {"role": "user", "content": f"{prompt}\n\nОпции: {options}"}
        ],
        temperature=temperature
    )
    return response.choices[0].message.content
def main():
    with open("eval_datasets/mmmu_Agriculture/mmmu_en_to_mk_nllb_streaming_20250723_125600.jsonl", "r") as f:
        full_data = []
        for line in f:
            data = json.loads(line)
            full_data.append(data)
    print(len(full_data))
    print(full_data[0].keys())

    correct = 0
    incorrect = 0

    for example in tqdm(full_data):
        prompt = example["translated_question_mk"]
        options = example["translated_options_mk"]
        response = get_openai_response(prompt, options)
        print(response)
        print(example["answer"])
        if response == example["answer"]:
            correct += 1
        else:
            incorrect += 1
    print(f"Correct: {correct}, Incorrect: {incorrect}")

if __name__ == "__main__":
    main()
