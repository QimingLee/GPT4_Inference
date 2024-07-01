import base64
import json
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(api_key="sk-4vfdkP4cczqNHUZtBbE655309785453f9d55A428EdFc3aBe", base_url="https://openkey.cloud/v1")

# Function to encode the image to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Input and output file paths
input_file = '/share/home/fengxiaocheng/whzhong/qmli/CODA-LM-main/data/CODA-LM/Test/vqa_anno/driving_suggestion_part2.jsonl'
output_file = '/share/home/fengxiaocheng/whzhong/qmli/gpt4/gpt4_eccv_test/driving_suggestion_answer_part3.jsonl'

# Few-shot examples for image-text interaction
few_shot_examples = [
    {
        "prompt": "There is an image of traffic captured from the perspective of the ego car. Focus on objects influencing the ego car's driving behavior: vehicles (cars, trucks, buses, etc.), vulnerable road users (pedestrians, cyclists, motorcyclists), traffic signs (no parking, warning, directional, etc.), traffic lights (red, green, yellow), traffic cones, barriers, miscellaneous(debris, dustbin, animals, etc.). You must not discuss any objects beyond the seven categories above. Please provide driving suggestions for the ego car based on the current scene.",
        "image_path": "/share/home/fengxiaocheng/whzhong/qmli/CODA-LM-main/data/test/images/0002.jpg",
        "answer": "Maintain a safe following distance from the black SUV ahead and be prepared to stop due to the red traffic light at the intersection. Stay alert for any potential unexpected maneuvers from the white sedan on the right. Do not attempt to change lanes to the right due to construction barriers and traffic cones, and monitor the situation beyond the intersection for any changes that may affect traffic flow."
    },
    {
        "prompt": "There is an image of traffic captured from the perspective of the ego car. Focus on objects influencing the ego car's driving behavior: vehicles (cars, trucks, buses, etc.), vulnerable road users (pedestrians, cyclists, motorcyclists), traffic signs (no parking, warning, directional, etc.), traffic lights (red, green, yellow), traffic cones, barriers, miscellaneous(debris, dustbin, animals, etc.). You must not discuss any objects beyond the seven categories above. Please provide driving suggestions for the ego car based on the current scene.",
        "image_path": "/share/home/fengxiaocheng/whzhong/qmli/CODA-LM-main/data/test/images/0062.jpg",
        "answer": "The ego car should maintain a safe following distance from the vehicle in front to allow time to react to any sudden stops. It should also reduce speed in preparation for the construction zone ahead, be alert for potential instructions from the construction worker, and be ready to adjust the vehicle's position slightly to the left to stay within the guided path created by the traffic cones and away from the barriers. Observing any temporary traffic signs or signals installed due to construction will be critical for safe navigation. Additionally, passing the parked or stopped black sedan cautiously and maintaining a safe distance from the construction workers are also advised. Be alert for any additional signs or changes in the road layout ahead."
    },
    {
        "prompt": "There is an image of traffic captured from the perspective of the ego car. Focus on objects influencing the ego car's driving behavior: vehicles (cars, trucks, buses, etc.), vulnerable road users (pedestrians, cyclists, motorcyclists), traffic signs (no parking, warning, directional, etc.), traffic lights (red, green, yellow), traffic cones, barriers, miscellaneous(debris, dustbin, animals, etc.). You must not discuss any objects beyond the seven categories above. Please provide driving suggestions for the ego car based on the current scene.",
        "image_path": "/share/home/fengxiaocheng/whzhong/qmli/CODA-LM-main/data/test/images/3474.jpg",
        "answer": "Given the positions and potential behaviors of these road users, the ego car should reduce speed and increase following distance due to the construction vehicle ahead. It should also prepare to change lanes to the left, checking for the sedan and motorcyclist's positions and signaling intentions clearly. Overtaking should be done with care. Continuous monitoring of the workers near the roadside is essential to react to any unexpected entries onto the road. As there are barriers on the right, any maneuvers should be planned with consideration of the reduced lane width. Additionally, the ego car should slow down near the parked utility vehicle and the area with traffic cones, watching for any workers or equipment that might enter the roadway. Readiness to react to the pedestrian crossing and the group near the utility vehicle is also crucial for safety. Given the construction truck ahead, the presence of road workers, a motorcyclist close by, and wet road conditions, the ego vehicle should monitor surrounding vehicles closely, particularly the white car and the merging yellow truck, to adjust its positioning in response to their movements. As the construction area approaches, the vehicle should be prepared to navigate around the work zone, possibly changing lanes if safe to do so, while respecting the reduced speed limits typical of such areas. Always be ready to stop if necessary and observe any temporary signage."
    },
    {
        
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
                        # "detail": "high"
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
