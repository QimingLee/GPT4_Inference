import base64
import requests

# OpenAI API Key
api_key = "sk-4vfdkP4cczqNHUZtBbE655309785453f9d55A428EdFc3aBe"

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

# Path to your image
image_path = "/share/home/fengxiaocheng/whzhong/qmli/CODA-LM-main/data/test/images/0001.jpg"

# Getting the base64 string
base64_image = encode_image(image_path)

headers = {
  "Content-Type": "application/json",
  "Authorization": f"Bearer {api_key}"
}

payload = {
  "model": "gpt-4o-2024-05-13",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "What’s in this image?"
        },
        {
          "type": "image_url",
          "image_url": {
            "url": f"data:image/jpeg;base64,{base64_image}",
            "detail": "high"
          }
        }
      ]
    }
  ],
  "max_tokens": 300
}

response = requests.post("https://openkey.cloud/v1", headers=headers, json=payload)

print(response.json())