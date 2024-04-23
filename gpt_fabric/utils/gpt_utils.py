import re
import numpy as np
import os 
import random

system_prompt = '''**Cloth Folding Robot**
Role: You are the brain of a cloth folding robot. The robot would pick one spot on the cloth (referred to as the "pick point"), lift it by a small amount, drag it over to another spot (referred to as "the place point"), and finally release it.

Inputs:
- Method of folding: A description of how the cloth should be folded
- Cloth corners: The robot sees the cloth lying on a table from the top and gets a depth image for it. This depth image is then processed to extract the corner points for the cloth. The pixel co-ordinates for the corners will be given to you as an input. The format of each pixel coordinate would be [x-coordinate, y-coordinate]
- Cloth Center: The robot will be given the [x-coordinate, y-coordinate] pair corresponding to the center of the initial cloth configuration

Task:
- Thought Process: Note down possible ways of picking and placing the cloth and their potential effects
- Planning: Provide a pair of pick and place point from the cloth corners provided as input for folding the cloth. 

Output:
- Planning (MOST IMPORTANT): Pick Point = (x 1, y 1) and Place Point = (x 2, y 2)
- Thought Process: Why did you choose these points and not something else?

PLEASE OUTPUT THE PICK POINT AND THE PLACE POINT FIRST AND THEN OUTPUT THE THOUGHT PROCESS INVOLVED
'''

image_analysis_instruction = '''
I will be providing you with two images. In each image, you will see a background that's divided into four quadrants with alternating shades of gray. Think of this background as a flat surface on which a cloth is kept.
This cloth could be seen in the centre of these images as a geometric shape coloured with orange and pink.
There is also a black arrow in the first image, which essentially represents an action where someone would pick a point on the cloth corresponding to the black dot from where the arrow originates. This would be represented as the picking point. On the other hand, the point where the tip of the black arrow is located corresponds to the location where the chosen picking point is placed. This is referred to as the placing point.

This sequence of action of picking a point on the cloth and place it somewhere results in a fold, whose result can be seen in the next image. So basically we are folding the cloth in the first image to get to the second image.
I want you to describe the instructions for the folding step that someone could follow to achieve the same fold. 
Look at the relative location of the tip of the arrow with respect to the center of the image. Depending on whether this is near the center or a diagonally opposite point or a point along the given edge, choose your placing point as the center or a diagonally opposite point or a point along the given edge respectively. 

IMPORTANT: INLCUDE THE PICKING AND PLACING POINT INFORMATION IN THE RESPONSE. YOU MUST SPECIFY WHERE SHOULD THE PLACING POINT BE.

RETURN YOUR OUTPUT IN THE BELOW FORMAT ONLY:
- Instructions: The instructions for the given folding step.
- Explanation: Why did you choose this pair of picking and placing points.
'''

gpt_v_demonstrations = {
    "DoubleTriangle": {
        "data": ["7", "15", "19", "29", "33"],
        "instruction": "- Use this pair of images to guide your response",
        "gpt-demonstrations": ["5", "15", "35", "47", "82", "23", "91", "66", "75", "59"]
    },
    "AllCornersInward": {
        "data": [],
        "instruction": "- Use this pair of images to guide your response",
        "gpt-demonstrations": ["4", "15", "35", "47", "82", "23", "91", "64", "75", "67"]
    },
    "CornersEdgesInward": {
        "data": [],
        "instruction": "- Use this pair of images to guide your response",
        "gpt-demonstrations": []
    },
    "DoubleStraight": {
        "data": ["7", "15", "19", "29", "33"],
        "instruction": "- Use this pair of images to guide your response",
        "gpt-demonstrations": ["4", "15", "35", "47", "82", "23", "91", "64", "75", "67"]
    }
}

def get_user_prompt(corners, center, autoPrompt, instruction, task, step):
    '''
    This code consists of the specific CoT prompts designed for the different kinds of folds involved here    
    '''
    center = str(center)
    corners = str(corners)
    cloth_info = "- Cloth corners: " + corners + "\n- Cloth center: " + center

    if autoPrompt:
        return instruction + "\n" + cloth_info
    else:
        return "Not implemented"
            
def parse_output(output):
    '''
    This function parses the string output returned by the LLM and returns the pick and place point coordinates that can be integrated in the code
    '''
    # Define regular expressions to match pick point and place point patterns
    pick_point_pattern = re.compile(r'Pick Point = \((\d+), (\d+)\)')
    place_point_pattern = re.compile(r'Place Point = \((\d+), (\d+)\)')

    # Use regular expressions to find matches in the text
    pick_match = pick_point_pattern.search(output)
    place_match = place_point_pattern.search(output)

    # Extract x and y values for pick point
    pick_point = None
    if pick_match:
        pick_point = np.array(tuple(map(int, pick_match.groups())))
    else:
        pick_point = np.array([None, None])

    # Extract x and y values for place point
    place_point = None
    if place_match:
        place_point = np.array(tuple(map(int, place_match.groups())))
    else:
        place_point = np.array([None, None])

    return pick_point, place_point

def analyze_images_gpt(image_list, task, action_id):
    '''
    This function takes the paths of the demonstration images and returns the description about what's happening
    '''
    import base64
    import requests
    # This uses Daniel's API-key
    api_key="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"

    # Function to encode image
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    first_image = encode_image(image_list[0])
    second_image = encode_image(image_list[1])

    # Getting information corresponding to the demonstrations that we'd use
    gpt_vision_demonstrations_local = gpt_v_demonstrations[task]
    baseline_demo_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'demonstrations', task)
    demo_images_list = gpt_vision_demonstrations_local["data"]
    demonstration_dictionary_list = []
    for demo_image_id in demo_images_list:
        demo_first_image = encode_image(os.path.join(baseline_demo_path, demo_image_id, "rgbviz", str(action_id) + ".png"))
        demo_second_image = encode_image(os.path.join(baseline_demo_path, demo_image_id, "rgbviz", str(action_id + 1) + ".png"))
        demo_instruction = gpt_vision_demonstrations_local["instruction"]
        user_prompt_dictionary = {
            "role": "user",
            "content": [
                {"type": "text", "text": image_analysis_instruction},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{demo_first_image}"
                    }
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{demo_second_image}"
                    }
                }
            ]
        }
        assistant_response_dictionary = {
            "role": "assistant",
            "content": [
                {"type": "text", "text": demo_instruction}
            ]
        }
        demonstration_dictionary_list += [user_prompt_dictionary, assistant_response_dictionary]

    headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
    }

    instruction = ""
    while instruction == "":
        payload = {
            "model": "gpt-4-vision-preview",
            "messages": [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": "You will be given some demonstration user prompts and the corresponding output that is expected from you. Use these examples to guide your response for the actual query."
                        }
                    ]
                }
            ] + demonstration_dictionary_list + 
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": image_analysis_instruction},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{first_image}"
                            }
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{second_image}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 1000,
            "temperature": 0.2
        }

        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        response = response.json()
        if 'choices' in response:
            text = response['choices'][0]['message']['content']
            match = re.search(r'Instructions: ([^\n]+)\.', text)
            if match:
                instruction = match.group(1)
    return instruction
