import json
import os
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm
import base64

load_dotenv()

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def get_openai_response(prompt, options, image_path, model="gpt-4.1", temperature=0.0):

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    base64_image = encode_image(image_path)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "Ќе ти биде поставено прашање и ќе ти бидат дадени опции. Ти треба да одговориш со буквата на опцијата која е точниот одговор, како A, B, C, или D."
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                        }
                    },
                ],
            }
        ],
        max_tokens=300,
    )

    return response.choices[0].message.content

def main():

    categories = os.listdir("evaluation")

    results = {}
    for category in categories:
        results[category] = {}
        for file in os.listdir(f"/root/synglot/evaluation/{category}"):
            if file.endswith(".jsonl"):
                full_data = []
                with open(f"/root/synglot/evaluation/{category}/{file}", "r") as f:
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
                    image_filename = example['image_1'].split("/")[-1]
                    image_path = os.path.join("/root/synglot/evaluation", category, "mmmu_media", image_filename)
                    response = get_openai_response(prompt, options, image_path)
                    print(f"Response: {response}; Answer: {example['answer']}")
                    if response == example["answer"]:
                        correct += 1
                    else:
                        incorrect += 1
                print(f"Correct: {correct}, Incorrect: {incorrect}")
                results[category] = {
                    "correct": correct,
                    "incorrect": incorrect,
                    "accuracy": correct / (correct + incorrect)
                }
        # test run 1 category
        break

    with open("results.json", "w") as f:
        json.dump(results, f)
if __name__ == "__main__":
    main()
