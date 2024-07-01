import base64
import json
from openai import OpenAI

# Initialize OpenAI client
# client = OpenAI(api_key="sk-4vfdkP4cczqNHUZtBbE655309785453f9d55A428EdFc3aBe", base_url="https://openkey.cloud/v1")
client = OpenAI(api_key="sk-TxD13msrrg1eaWbI04970cAdCdE64e948d7c0190198cF06f", base_url="https://openkey.cloud/v1")

# Function to encode the image to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Input and output file paths
input_file = '/share/home/fengxiaocheng/whzhong/qmli/CODA-LM-main/data/CODA-LM/Test/vqa_anno/region_perception.jsonl'
output_file = '/share/home/fengxiaocheng/whzhong/qmli/gpt4/gpt4_eccv_fewshot_test/region_perception_answer.jsonl'

# Few-shot examples for image-text interaction
few_shot_examples = [
    {
        "prompt": "Please describe the object inside the red rectangle in the image and explain why it affect ego car driving.",
        "image_path": "/share/home/fengxiaocheng/whzhong/qmli/CODA-LM-main/data/test/images_w_boxes/0009_object_4.jpg",
        "answer": "This object is a traffic sign with directional arrows and supplementary plates. The sign shows three arrows indicating lane directions: the left arrow directs traffic to turn left, the central arrow indicates that the lane goes straight ahead, and the right arrow signifies a lane for turning right. The plates below the arrows display speed limits and vehicle classification restrictions. The presence of this sign guides the ego car to choose the correct lane based on its intended route. If the ego car intends to proceed straight, it should align with the central arrow. The speed limit and vehicle classification signs instruct the driver to adhere to the indicated speed limit and lane usage based on the type of vehicle they are operating."
    },
    {
        "prompt": "Please describe the object inside the red rectangle in the image and explain why it affect ego car driving.",
        "image_path": "/share/home/fengxiaocheng/whzhong/qmli/CODA-LM-main/data/test/images_w_boxes/0954_object_1.jpg",
        "answer": "The object is a large commercial vehicle specifically designed for mixing and transporting concrete to construction sites, commonly called a concrete mixer or cement mixer truck. It is characterized by its rotating drum which mixes the concrete materials. Due to its considerable size and the nature of its cargo, this vehicle often has a slower acceleration and deceleration rate compared to standard vehicles. When driving near such a heavy vehicle, the ego car should maintain a safe following distance and be prepared for longer stopping times. The future driving behavior should include increased caution, as maneuverability could be limited and the mixer truck could make wide turns that require additional space."
    },
    {
        "prompt": "Please describe the object inside the red rectangle in the image and explain why it affect ego car driving.",
        "image_path": "/share/home/fengxiaocheng/whzhong/qmli/CODA-LM-main/data/test/images_w_boxes/2270_object_2.jpg",
        "answer": "Two individuals are detected on foot, likely pedestrians, appearing near the curbside of the road. These pedestrians might enter the roadway or cross the street, so the ego vehicle must be prepared to slow down or stop to yield the right of way and avoid a collision, adhering to traffic laws that prioritize pedestrian safety."
    },
    {
        "prompt": "Please describe the object inside the red rectangle in the image and explain why it affect ego car driving.",
        "image_path": "/share/home/fengxiaocheng/whzhong/qmli/CODA-LM-main/data/test/images_w_boxes/3708_object_1.jpg",
        "answer": "A traffic cone is a conical marker, often orange in color with white bands for high visibility, used to guide traffic by delineating driving lanes, marking off-road construction or other areas that may be dangerous or off-limits to vehicles. It can influence the driving behavior by indicating to the driver of the ego car to stay clear of a certain lane or area on the road, guiding them to follow a safe path away from potential hazards or closed-off areas."
    },
    {
        "prompt": "Please describe the object inside the red rectangle in the image and explain why it affect ego car driving.",
        "image_path": "/share/home/fengxiaocheng/whzhong/qmli/CODA-LM-main/data/test/images_w_boxes/4358_object_2.jpg",
        "answer": "A construction vehicle, specifically a wheel loader, is visible. This type of vehicle is commonly used for moving or loading materials such as asphalt, demolition debris, dirt, snow, feed, gravel, logs, raw minerals, recycled material, rock, sand, wood chips, etc. The presence of a construction vehicle suggests that there may be ongoing construction or roadwork ahead. The ego vehicle must proceed with caution, as construction sites can be hazardous, with potential obstacles, workers on or near the roadway, and unexpected movements of construction equipment. The driver should reduce speed and be prepared to obey any temporary traffic signs or directions from construction personnel for safe passage."
    }
]

# Prepare few-shot examples for GPT-4 input format
few_shot_prompts = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": ex["prompt"]
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encode_image(ex['image_path'])}",
                    "detail": "high"
                }
            },
            {
                "type": "text",
                "text": ex["answer"]
            }
        ]
    }
    for ex in few_shot_examples
]

# Open input and output files
with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
    for line in infile:
        # Parse JSON data from current line
        data = json.loads(line)
        
        # Extract information
        question_id = data['question_id']
        image_path = "/share/home/fengxiaocheng/whzhong/qmli/CODA-LM-main/data/" + data['image']
        question_text = data['question']
        base64_image = encode_image(image_path)
        
        # Prepare messages for the GPT-4 API request
        messages = [
            {"role": "system", "content": "You are an autonomous driving expert, specializing in recognizing traffic scenes and making driving decisions."},
            {"role": "user",
             "content": [
                    {
                    "type": "text",
                    "text": question_text
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
        ]
        
        # Incorporate few-shot examples into the messages
        messages.extend(few_shot_prompts)
        
        # Request completion from GPT-4 API using few-shot method
        response = client.chat.completions.create(
            model="gpt-4o-2024-05-13",
            messages=messages,
            stream=False
        )
        
        # Extract model response from API response
        model_response = response.choices[0].message.content
        
        # Print model response (for debugging purposes)
        print(f"Question ID: {question_id}\nQuestion: {question_text}\nAnswer: {model_response}\n")
        
        # Add answer field to data
        data['answer'] = model_response
        
        # Write updated data back to output JSONL file
        outfile.write(json.dumps(data) + '\n')

print("Processing completed. Answers added to each entry and saved to", output_file)
