import base64
import json
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(api_key="sk-TxD13msrrg1eaWbI04970cAdCdE64e948d7c0190198cF06f", base_url="https://openkey.cloud/v1")

# Function to encode the image to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Input and output file paths
input_file = '/share/home/fengxiaocheng/whzhong/qmli/CODA-LM-main/data/CODA-LM/Test/vqa_anno/general_perception_part2.jsonl'
output_file = '/share/home/fengxiaocheng/whzhong/qmli/gpt4/gpt4_eccv_fewshot_test/general_perception_answer_part3.jsonl'

# Few-shot examples for image-text interaction
few_shot_examples = [
    {
        "prompt": "Please describe the object inside the red rectangle in the image and explain why it affect ego car driving.",
        "image_path": "/share/home/fengxiaocheng/whzhong/qmli/CODA-LM-main/data/test/images/3404.jpg",
        "answer": "The traffic image reveals multiple road users and objects that influence the driving behavior of the ego car. Here's the summary of the observed data:\n\n1. **Vehicles**:\n    - A large yellow construction vehicle is directly in the ego car's path, occupying most of the nearest lane. It appears stationary or moving slowly. This vehicle poses an obstacle that might necessitate a lane change by the ego car to avoid a collision. The ego car has to maintain a safe following distance and be prepared for sudden stops or maneuvers by the construction vehicle.\n    - A white utility vehicle with red stripes and roof-mounted equipment is on the left side of the ego car in a parallel lane. Though not directly in the ego car\u2019s path, this vehicle's proximity and potential lateral movement could influence lane-changing decisions. The ego car needs to stay alert for lane changes or sudden stops by surrounding vehicles.\n    - Various vehicles populate adjacent lanes and oncoming traffic, including cars facing the opposite direction ahead. This necessitates constant monitoring by the ego car for potential lane changes by other drivers that could affect traffic flow.\n\n2. **Traffic Signs**:\n    - A sign on the right past the construction vehicle indicates a road work area ahead. This warns of possible further obstructions or traffic pattern changes, prompting the ego car to anticipate these factors.\n    - A directional sign above the road shows allowable directions, indicating the options for straight travel and a right turn. This informs the ego car of the possible maneuvers at the upcoming intersection or road section.\n\n3. **Traffic Lights**:\n    - Green traffic lights are visible ahead, signaling permission to proceed through the intersection. The ego car can continue forward if the lane is clear. Special caution is advised due to the construction vehicle\u2019s potential movements.\n\n4. **Traffic Cones**:\n    - Bright orange traffic cones to the right of the construction vehicle suggest a demarcated work area or lane closure. This signals the ego car to consider changing lanes to the left.\n\n5. **Barriers**:\n    - Red and white plastic barriers on the right delineate the construction work zone from the traffic area. These barriers indicate the bounds of the construction zone, assisting in lane decisions for the ego car.\n\n6. **Other Objects**:\n    - Various construction signs and materials on the right side signal an active construction site. These indicators necessitate increased vigilance from the ego car due to potential sudden changes or obstacles from the construction activities.\n\nIn summary, the traffic image shows a complex situation with multiple static and moving entities, including a prominent construction vehicle, various traffic signs, green traffic lights, traffic cones, barriers, and other construction-related objects. The ego car must navigate these elements carefully, monitoring its surroundings, anticipating changes, and making informed maneuvers to ensure safe passage."
    },
    {
        "prompt": "Please describe the object inside the red rectangle in the image and explain why it affect ego car driving.",
        "image_path": "/share/home/fengxiaocheng/whzhong/qmli/CODA-LM-main/data/test/images/2099.jpg",
        "answer": "The traffic scene under review reveals a mix of vehicles, construction equipment, and various road user categories, each influencing the ego vehicle's maneuvering decisions. Visible in the lanes around the ego vehicle are a diverse array of vehicles including a green taxi, a black sedan, a white SUV, and commercial vehicles. Their placement, either stationary or in slow-moving traffic on both sides of the ego vehicle, necessitates careful consideration of lane changes or speed adjustments due to their close proximity and respective velocities.\n\nAdditionally, the scene features construction-related heavy machinery, specifically a Volvo truck within the right lane, demarcated by traffic cones. This indicates an active construction zone, implying potential lane closure and a requirement for the ego vehicle to either merge left or decelerate accordingly to navigate safely around the work zone.\n\nDespite the complex vehicular and construction presence, there seems to be an absence of vulnerable road users such as pedestrians, cyclists, or motorcyclists in the immediate vicinity of the ego vehicle. This absence allows for a concentrated focus on vehicular traffic and fixed obstacles without the additional concern for less predictable movements from these road users.\n\nA partially visible traffic sign hangs above the road, its full message obscured by the angle of view. Although the exact directive of the sign remains unclear, its presence suggests it could be indicating crucial traffic instructions, warranting preparedness for unexpected changes in traffic patterns.\n\nThe traffic lights are present but indecipherable in the given image, leaving the ego vehicle's reaction to the traffic signal uncertain until the light status becomes clearer. This uncertainty emphasizes the importance of readiness to either stop, proceed, or proceed with caution depending on the eventual light signal detected.\n\nSurrounding the construction zone, a series of orange traffic cones extend from the ego vehicle's right side, pointing to a work area that may involve lane narrowing or closure. These cones serve to direct the ego vehicle's path, ensuring safe lane adherence or merging.\n\nThough not directly visible, the scene suggests the presence of barriers related to the construction zone, inferred from the positioning of construction vehicles. These barriers, if present, would necessitate altered traffic navigation by the ego car, with an emphasis on maintaining safe speeds.\n\nLastly, construction-related materials and machinery located adjacent to the delineated cone area could pose additional hazards. Their presence near the roadway demands heightened vigilance to preemptively address any potential shifts in their positioning or effects on the road conditions.\n\nTogether, these elements compose a road scene fraught with potential hazards and navigational challenges, from vehicle congestion and construction activities to indiscernible traffic directives, necessitating a multifaceted approach to safely maneuver the ego vehicle through the environment."
    },
    {
        "prompt": "Please describe the object inside the red rectangle in the image and explain why it affect ego car driving.",
        "image_path": "/share/home/fengxiaocheng/whzhong/qmli/CODA-LM-main/data/test/images/4019.jpg",
        "answer": "The traffic image under discussion exhibits various elements of interest divided into distinct categories that impact the driving patterns and decisions of the ego car. Here is a summary of the observed elements:\n\n**Vehicles:**\nA notable observation includes a dark-colored SUV traveling ahead of the ego car within the same lane. This SUV has its brake lights activated, signaling a potential slow-down or halt. The presence and actions of this SUV directly influence the ego car's driving behavior, necessitating a preparedness to decelerate or stop to maintain safe following distances and prevent collisions.\n\n**Traffic Signs:**\nTwo distinct types of traffic signs are identified on the roadside. The first is a round blue sign featuring a red cross, positioned on the right side of the road. This sign explicitly prohibits stopping or parking along this road section, advising the ego car to maintain its course without considering the road shoulder as a stop or park area. Additionally, there is directional signage placed to the left of the road. These signs provide crucial lane usage and navigation directions, aiding the ego car in selecting the appropriate lane and understanding permitted traffic movements in that area.\n\n**Traffic Cones and Barriers:**\nA combination of alternating white and orange traffic cones, alongside striped barriers, have been arranged on the left side of the road. This setup likely denotes the presence of construction work or roadworks, signaling potential hazards or alterations in the standard traffic patterns. Such configurations necessitate heightened caution from the ego car, particularly regarding lane discipline and awareness of possible obstructions. Similarly, red and white traffic cones have been placed on the road, forming a barrier for an upcoming lane closure, accompanied by a yellow construction sign. This arrangement indicates that a section of the road ahead is temporarily inaccessible, requiring the ego car to prepare for a lane merge if its path is impacted.\n\n**Barriers:**\nObservations include white barriers with reflective properties positioned along the median strip, delineating the separation of opposite traffic flows. These barriers serve as a visual reminder for the ego car to remain within its lane, avoiding any maneuvers that would entail crossing into oncoming traffic lanes. Another set of red and white striped traffic barriers and yellow posts have been placed to segregate lanes on the right side, underscoring the necessity for the ego car to adhere to its lane and refrain from unauthorized right turns or lane shifts beyond the barriers.\n\n**Other Objects:**\nGround markings displaying directional arrows have been observed. These markings are instrumental in guiding the ego car to maintain correct lane positioning and prepare for future lane changes or turns as dictated by the route and traffic regulations.\n\nIn conclusion, the collective presence and arrangement of these vehicles, traffic signs, cones, barriers, and other navigational aids exert a significant influence on the driving strategy and decision-making processes of the ego car, ensuring navigation is conducted safely and in compliance with road rules and conditions."
    },
    {
        "prompt": "Please describe the object inside the red rectangle in the image and explain why it affect ego car driving.",
        "image_path": "/share/home/fengxiaocheng/whzhong/qmli/CODA-LM-main/data/test/images/1187.jpg",
        "answer": "In the observed traffic scenario, various road users and objects with corresponding implications for autonomous driving are present:\n\nThere is a large orange truck parked on the right side of the lane in which the ego vehicle is currently traveling. The truck is facing away from the ego vehicle and is stationed near an area where maintenance activities are taking place. This truck is a significant consideration for the ego vehicle because it obstructs part of the roadway, reducing the available width for passing and posing a risk of sudden movement or the entry of workers onto the road.\n\nAdjacent to the parked truck, workers are engaged in maintenance or construction tasks. These individuals present a potential hazard for the ego vehicle as they could step into the roadway unexpectedly, necessitating caution and readiness to react by the autonomous system.\n\nA traffic sign on the right, fixed to a lamp post, displays blue with white arrows pointing both upwards and to the right. This sign alerts drivers, including the autonomous system of the ego vehicle, that the current lane permits both straight travel and right turns. Additionally, it signals that a turn may be obligatory shortly, influencing the choices made by the vehicle's navigation system.\n\nAhead, there is a traffic light suspended above the center of the road, showing a green signal. For the ego vehicle, this green light indicates that it can legally continue on its way. In the absence of this signal, the vehicle would have to rely solely on road signage and markings for guidance on how to behave at intersections.\n\nSurrounding the maintenance area, numerous traffic cones are placed to the right, delineating the boundary of the work zone. These cones serve to guide traffic away from the area and indicate that parts of the road are temporarily inaccessible, typically signifying a narrower path for vehicles to traverse. Consequently, the ego vehicle is prompted to decelerate and possibly alter its course to steer clear of the work zone.\n\nAlongside these cones, white and red striped barriers are installed, marking out the maintenance area further. These barriers enhance the visual cues for the ego vehicle to remain at a secure distance from workers and machinery, which is crucial for avoiding accidents.\n\nAdditionally, several street bins are situated behind the traffic cones on the adjacent sidewalk. Although these bins don't directly interfere with the road, they suggest increased pedestrian activity around the work zone, which could lead to potential hindrances in the vicinity.\n\nLastly, maintenance equipment such as a ladder, orange buckets, and a broom have been placed on the pedestrian pathway adjacent to the parked truck. The presence of these items near the driving path signifies that workers might move them or that their own activity could result in people inadvertently entering the road, prompting the ego vehicle to proceed with caution and be prepared for abrupt stops or maneuvers."
    },
    {
        "prompt": "Please describe the object inside the red rectangle in the image and explain why it affect ego car driving.",
        "image_path": "/share/home/fengxiaocheng/whzhong/qmli/CODA-LM-main/data/test/images/2742.jpg",
        "answer": "The traffic image showcases a variety of road users, traffic signs, and other elements that play significant roles in influencing the driving behavior of the ego vehicle. \n\nStarting with vehicles, there is a dark-colored sedan positioned directly in front of the ego car, situated in the center of the driving lane. This sedan dictates the pace for the ego vehicle, meaning its speed and movements will significantly impact how the ego car maneuvers. A bright-colored taxi is noticed on the left side, slightly ahead of the ego vehicle. The placement of the taxi is critical as it may impede the ego vehicle\u2019s ability to change lanes or respond to merging traffic efficiently. Additionally, an array of construction vehicles, including a wheel loader on the right side, signifies an ongoing work zone. These construction vehicles may enter or exit the roadway unexpectedly, affecting traffic flow and necessitating a cautious approach from the ego car, such as being prepared to stop or slow down. An orange truck is observed in the left lane, ahead of the ego car, and appears to be in motion. This truck\u2019s presence in the adjacent lane is vital for the ego car to consider if intending to change lanes or to anticipate potential merging actions.\n\nVulnerable road users include a pedestrian wearing a high-visibility jacket, crossing the road in front of the ego car, and another standing by the roadside barrier. Their presence necessitates a slowdown or even a complete stop of the ego vehicle to ensure their safety. Additionally, several construction workers are present around the construction vehicles. Their potential to unpredictably enter the roadway commands extra caution and readiness to stop from the ego car.\n\nTraffic signs are visible, indicating a temporary lane closure on the right side, suggesting a change in traffic patterns that the ego car must adapt to by staying in the open lane. Moreover, there are barricades and signage on the right that constitute road work ahead, fostering reduced speed limits and heightened caution due to potential unexpected obstacles or shifts in traffic flow.\n\nOrange and white traffic cones placed on the right side mark the perimeter for construction works, guiding the ego car away from hazards or closed lanes. \n\nBarriers such as orange and white temporary traffic barriers, along with metal and plastic barriers on the right side of the road, delineate the construction zone. These barriers not only signal potential hazards but also alter the normal path of travel, requiring slower speeds and increased vigilance.\n\nOther objects include construction equipment and materials scattered on the right, along with a person who might be directing traffic. This situation often leads to reduced speed limits and a need for heightened awareness to prevent any collisions.\n\nIn summary, the traffic situation depicted requires careful attention to various vehicles, pedestrians, and static elements like traffic signs, cones, and barriers. Each presents a unique consideration for the ego car, affecting its driving behavior through speed adjustments, lane changes, and overall increased caution.\n"
    },
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
