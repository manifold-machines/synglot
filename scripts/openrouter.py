import requests
import json
import base64
from pathlib import Path
import os
from dotenv import load_dotenv
load_dotenv()

system_prompt = """Ќе ти биде поставено прашање врз основа на слика и ќе ти бидат дадени опции за точен одговор. Ти треба да одговориш САМО со буквата на опцијата која е точниот одговор, како A, B, C... во следниот формат:

### A ###

Не треба да даваш никакви други информации, ниту да го објаснуваш одговорот.

"""

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def answer_question(question, options, image_path, model="google/gemma-3-4b-it"):
    base64_image = encode_image_to_base64(image_path)
    data_url = f"data:image/jpeg;base64,{base64_image}"

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
    "Authorization": f"Bearer {os.environ.get('OPENROUTER_API_KEY')}",
    "Content-Type": "application/json"
    }    

    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"{question}\n\nОпции: {options}"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": data_url
                    }
                }
            ]
        }
    ]

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": 200
    }

    response = requests.post(url, headers=headers, json=payload)
    return response.json()#["choices"][0]["message"]["content"]


if __name__ == "__main__":
    question = "Која од опциите е точниот одговор? Детално објасни зошто."
    options = "A. 1\nB. 2\nC. 3\nD. 4"
    image_path = "/root/synglot/mmmu-mk/mmmu_Materials/mmmu_media/image_1_000015.jpg"
    model = "google/gemma-3-4b-it"  # Default model for testing
    answer = answer_question(question, options, image_path, model)
    print(answer)
