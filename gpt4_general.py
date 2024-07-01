# python3
# Please install OpenAI SDK first：`pip3 install openai`
from openai import OpenAI
import base64
import json

client = OpenAI(api_key="sk-4vfdkP4cczqNHUZtBbE655309785453f9d55A428EdFc3aBe", base_url="https://openkey.cloud/v1")
# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')
  
# input_file = '/share/home/fengxiaocheng/whzhong/qmli/CODA-LM-main/data/CODA-LM/Test/vqa_anno/general_perception.jsonl'
input_file = '/share/home/fengxiaocheng/whzhong/qmli/CODA-LM-main/data/CODA-LM/Test/vqa_anno/region_perception.jsonl'
# output_file = '/share/home/fengxiaocheng/whzhong/qmli/gpt4/gpt4_eccv_test/general_perception_answer.jsonl'
output_file = '/share/home/fengxiaocheng/whzhong/qmli/gpt4/gpt4_eccv_test/region_perception_answer.jsonl'

with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
    for line in infile:
        # 解析当前行的JSON数据
        data = json.loads(line)
        
        # 提取信息
        question_id = data['question_id']
        image_path = "/share/home/fengxiaocheng/whzhong/qmli/CODA-LM-main/data/"+data['image']
        question_text = data['question']
        base64_image = encode_image(image_path)
        # 从模型获取的响应
        response = client.chat.completions.create(
        model="gpt-4o-2024-05-13",
        messages=[
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
                ],
                stream=False
        )
        model_response = response.choices[0].message.content
        print(model_response)
        # 在数据中增加answer字段
        data['answer'] = model_response
        
        # 将更新后的数据写回新的jsonl文件
        outfile.write(json.dumps(data) + '\n')


print("处理完成，已将答案添加至每个条目并保存至", output_file)