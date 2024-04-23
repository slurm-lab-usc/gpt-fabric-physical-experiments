import os
import os.path as osp
import argparse
import time
import json
import base64
import re
import requests
from openai import OpenAI
import csv
import cv2 as cv
import math
import numpy as np

import datetime
from collections import deque
from matplotlib import pyplot as plt
from PIL import Image


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


api_key = "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"

api_key_enyu = "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"

api_key_daniel = "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"


def switch_user(api_key):
    client = OpenAI(api_key=api_key)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    return client, headers


client, headers = switch_user(api_key_daniel)


class RGBD_manipulation_real_world():
    def __init__(self, goal_image_path, obs_dir, system_prompt_path, direction_seg=8, distance_seg=4):

        self.goal_image = cv.imread(goal_image_path)
        self.system_prompt_path = system_prompt_path
        processed_goal_image, self.max_coverage, corners, center_point, self.goal_width, self.goal_height = self.preprocess(
            self.goal_image)
        self.obs_dir = obs_dir
        self.side_length = min(int(np.sqrt(self.max_coverage)),self.goal_width)
        print(f"self.side_length is {self.side_length}")
        cv.imwrite(osp.join(self.obs_dir,"goal_processed.png"),processed_goal_image)

        self.directions = []
        self.distances = []
        for i in range(1, direction_seg + 1):
            direction = str(i) + "/" + str(direction_seg // 2) + "*pi"
            self.directions.append(direction)
        for j in range(1, distance_seg + 1):
            distance = str(j / distance_seg)
            self.distances.append(distance)

        self.str_directions = f"[{', '.join(self.directions)}]"
        self.str_distances = f"[{', '.join(self.distances)}]"

    def preprocess(self, img):
        img_copy = img.copy()
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        blur = cv.medianBlur(gray, 5)
        th3 = cv.adaptiveThreshold(blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)

        corners = cv.goodFeaturesToTrack(gray, 15, 0.1, 20)
        corners = np.int0(corners)
        new_corners = np.squeeze(corners, axis=1)
        in_bound_corners = []

        # Find contours from the binary image
        contours, _ = cv.findContours(th3, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        possible_contours = []
        for contour in contours:
            x, y, w, h = cv.boundingRect(contour)
            # Draw the bounding box on the original image
            if w < 400 and w > 50 and h < 400 and h > 50:
                possible_contours.append(contour)

        if len(possible_contours) != 1:
            best_contour = max(possible_contours, key=cv.contourArea)
        else:
            best_contour = possible_contours[0]

        x, y, w, h = cv.boundingRect(best_contour)

        top = y - 20
        bottom = y + h + 20
        left = x - 20
        right = x + w + 20


        bound_x_min=100
        bound_y_min=100

        bound_x_max=540
        bound_y_max=380

        for corner in new_corners:
            px, py = corner

            if (px > left and px < right and py < bottom and py > top) and (px > bound_x_min and px < bound_x_max and py < bound_y_max and py > bound_y_min):
                cv.circle(img_copy, (px, py), 4, 255, -1)
                in_bound_corners.append(corner)

        new_x = int(x + 0.5 * w)
        new_y = int(y + 0.5 * h)
        cv.circle(img_copy, (new_x, new_y), 4, (255, 255, 255), -1)
        center_point = [new_x, new_y]

        cv.rectangle(img_copy, (x, y), (x + w, y + h), (255, 0, 255), 1)

        coverage_pix = cv.contourArea(best_contour)

        return img_copy, coverage_pix, in_bound_corners, center_point, w, h

    def vis_result(self, place_pixel, pick_pixel=None, img_path=None, img=None):
        if img_path:
            img = cv.imread(img_path)
        if pick_pixel is not None:
            cv.circle(img, (int(pick_pixel[0]), int(pick_pixel[1])), 5, (0, 255, 0), 2)
            cv.arrowedLine(img, (int(pick_pixel[0]), int(pick_pixel[1])), (int(place_pixel[0]), int(place_pixel[1])),
                           (0, 255, 0), 2)
            cv.circle(img, (int(place_pixel[0]), int(place_pixel[1])), 5, (128, 0, 128), 2)
        else:
            cv.circle(img, (int(place_pixel[0]), int(place_pixel[1])), 3, (0, 0, 255), 2)
        return img

    def response_process(self, response, messages=None):
        if 'choices' not in response.json():
            print(response.json())

            return None, None


        else:
            response_message = response.json()['choices'][0]['message']['content']
            print(response_message)

            pick_pattern = r'Pick point:.*?\[(.*?)\]'
            direction_pattern = r'Moving direction:.*?(\d+/\d+)'
            distance_pattern = r'Moving distance:.*?(\d+\.?\d*)'

            pick_match = re.search(pick_pattern, response_message)
            direction_match = re.search(direction_pattern, response_message)
            distance_match = re.search(distance_pattern, response_message)

            if not pick_match:
                return None, None

            pick_coords = [int(val) for val in pick_match.group(1).split(',')]
            pick_pixel = pick_coords

            moving_direction = direction_match.group(1) if direction_match else None
            if moving_direction is None:
                return None, None

            numerator, denominator = moving_direction.split('/')
            moving_direction = float(numerator) / float(denominator)

            moving_distance = float(distance_match.group(1)) if distance_match else None
            if moving_distance is None:
                return None, None

            actual_direction = moving_direction * np.pi
            actual_distance = (moving_distance)*self.side_length

            delta_x = actual_distance * np.cos(actual_direction)
            delta_y = actual_distance * np.sin(actual_direction)


            actual_x=int(pick_pixel[0] + delta_x)
            actual_y=int(pick_pixel[1] - delta_y)

            def intersect_point(x, y, a, b, x_min, y_min, x_max, y_max):
                dx = x - a
                dy = y - b
                intersections = []

                # Check intersection with left border (x = x_min)
                if dx != 0:
                    t = (x_min - a) / dx
                    y_left = b + t * dy
                    if y_min <= y_left <= y_max and t >= 0:
                        intersections.append((x_min, y_left))

                # Check intersection with right border (x = x_max)
                if dx != 0:
                    t = (x_max - a) / dx
                    y_right = b + t * dy
                    if y_min <= y_right <= y_max and t >= 0:
                        intersections.append((x_max, y_right))

                # Check intersection with bottom border (y = y_min)
                if dy != 0:
                    t = (y_min - b) / dy
                    x_bottom = a + t * dx
                    if x_min <= x_bottom <= x_max and t >= 0:
                        intersections.append((x_bottom, y_min))

                # Check intersection with top border (y = y_max)
                if dy != 0:
                    t = (y_max - b) / dy
                    x_top = a + t * dx
                    if x_min <= x_top <= x_max and t >= 0:
                        intersections.append((x_top, y_max))

                return intersections


            x_min, y_min, x_max, y_max = 20, 50, 620, 460  # Area borders

            intersections = intersect_point(actual_x, actual_y, pick_pixel[0], pick_pixel[1], x_min, y_min, x_max, y_max)
            print("Intersection points:", intersections)
            print(f"predicted points:[{actual_x,actual_y}]")


            out_bound_check= actual_x>x_min and actual_x<x_max and actual_y>y_min and actual_y<y_max

            place_pixel = [int(actual_x), int(actual_y)] if out_bound_check else intersections[0]

            print(f"pick_pixel is {pick_pixel}\n")
            print(f"place_pixel is {place_pixel}\n")

        return pick_pixel, [int(place_pixel[0]), int(place_pixel[1])]

    def _cal_direction(self, start, end):
        vector = [end[0] - start[0], start[1] - end[1]]

        angle = np.arctan2(vector[1], vector[0])
        angle = angle / np.pi
        if angle < 0.125:
            angle += 2

        return angle

    def recal(self, response_message, place_pixel, pick_pixel, center, img, last_pick_point=None,
              last_pick_point_oppo=None):

        img = self.vis_result(place_pixel=place_pixel, pick_pixel=pick_pixel, img=img.copy())
        vis_result_path = self.paths['processed vis image']
        cv.imwrite(vis_result_path, img)
        encoded_vis_result = encode_image(vis_result_path)

        correct_message = []
        text_correct_message = """
        I am providing you the visualization result of your predicted pick-and-place action. In the image you can see a green circle which is your predicted picking point and a green arrow which pointing to the your predicted move direction and a purple circle at the end of that arrow denoting the estimated placing point.\n
        """

        if last_pick_point is not None:
            pick_check = (abs(pick_pixel[0] - last_pick_point[0]) > 50) or (
                        abs(pick_pixel[1] - last_pick_point[1]) > 50)
            pick_oppo_check = (abs(pick_pixel[0] - last_pick_point_oppo[0]) > 50) or (
                        abs(pick_pixel[1] - last_pick_point_oppo[1]) > 50)

            if pick_check and pick_oppo_check:
                position_message = "By calculation, the chosen picking point is not near the last picking point or it's symetric point, you can stick with this picking point."

            elif pick_check:
                position_message = f"By calculation, the chosen picking point is near the last picking point's symetric point. The chosen picking point is [{pick_pixel[0]},{pick_pixel[1]}] and the last picking point's symetric point is [{last_pick_point_oppo[0]},{last_pick_point_oppo[1]}] so the pick point is within 100 pixel range of that point, please choose another point to pick."
            else:
                position_message = f"By calculation, the chosen picking point is near the last picking point. The chosen picking point is [{pick_pixel[0]},{pick_pixel[1]}] and the last picking point is [{last_pick_point[0]},{last_pick_point[1]}] so the pick point is within 100 pixel range of that point, please choose another point to pick."

            text_correct_message += position_message
        else:
            pick_check = True
            pick_oppo_check = True

        print(f"Proximity_check result is : last pick point:{pick_check}, last pick point oppo: {pick_oppo_check}\n")

        direction_pattern = r'Moving direction:.*?(\d+/\d+)'
        direction_match = re.search(direction_pattern, response_message)
        moving_direction = direction_match.group(1)
        numerator, denominator = moving_direction.split('/')
        moving_direction = float(numerator) / float(denominator)

        print(f"the moving_direction is {moving_direction} with type {type(moving_direction)}")

        direction = self._cal_direction(center, pick_pixel)

        print(f"the cal_direction is {direction} with type {type(direction)}")

        difference = np.abs(moving_direction - direction)

        print(f"difference is {difference}")

        possible_directions = []
        for i in range(1, 9):
            possible_directions.append(i / 4)

        possible_directions = np.array(possible_directions)
        possible_directions_diff = np.abs(possible_directions - direction)

        choice = np.argmin(possible_directions_diff)
        str_direction = self.directions[choice]

        left = choice - 1
        right = choice + 2

        if left < 0:
            left = 8 + left
        if right > 8:
            right = right - 8
        if left < right:
            accept_direction_list = possible_directions[left:right]
            str_direction_list = self.directions[left:right]
        else:
            accept_direction_list = possible_directions[left:] + possible_directions[:right]
            str_direction_list = self.directions[left:] + self.directions[:right]

        str_direction_list = f"[{','.join(str_direction_list)}]"
        direction_check = (moving_direction in accept_direction_list) or difference < 0.25

        print(f"direction_check result is : {direction_check}\n")

        if direction_check and pick_check and pick_oppo_check:
            direction_message = "\n By calculating the pick point you choose and center point, the direction starting from the center point to the picking point is roughly " + str_direction + ". The direction you predicted falls in the acceptable range."
        elif pick_check and pick_oppo_check:
            direction_message = "\n The picking point is a acceptable choice as it's not near to last picking point or it's symmetric point. But by calculating the pick point you choose and center point, the direction starting from the center point to the picking point is roughly " + str_direction + ". The direction you predicted doesn't falls in the acceptable range. Please use " + str_direction + "as the moving direction if you want to pick the same picking point ."
        else:
            direction_message = "\n The picking point is not an accept choice as it's near to last picking point or it's symmetric point. The predicted moving direction is also incorrect."

        text_correct_message += direction_message

        check_result = direction_check and pick_oppo_check and pick_check

        correction_message = """

        Based on the assist of previous calculation, do you think your predicted move will help flatten the fabric? If so, you can repeat your answer. If you don't think this move will help flatten the fabric, you should give a new prediction following the same output format.

        """

        text_correct_message += correction_message

        text_content = {
            "type": "text",
            "text": text_correct_message,
        }
        image_content = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{encoded_vis_result}",
                "detail": "high"
            }
        }

        correct_message.append(text_content)
        correct_message.append(image_content)
        return correct_message, check_result, direction_check

    def get_pick_place(self, messages, headers):

        payload = {
            "model": "gpt-4-vision-preview",
            "messages": messages,
            "max_tokens": 1024,
            "temperature": 0.1,
            "top_p": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0
        }
        re_cal = True

        while re_cal:
            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

            # print(response.json())
            pick_point, place_point = self.response_process(response)

            if 'choices' in response.json():
                response_message = response.json()['choices'][0]['message']['content']

                if pick_point is not None:

                    re_cal = False
                    break

                else:
                    re_cal = True
                    time.sleep(30)
                    format_error_message = "The output given by you has format error, please output your result according to the given format."

                    messages.append(

                        {
                            "role": "assistant",
                            "content": response_message,

                        })

                    messages.append(
                        {
                            "role": "user",
                            "content": [

                                {"type": "text",
                                 "text": format_error_message,
                                 }
                            ]
                        }
                    )

            else:
                re_cal = True
                time.sleep(30)
                format_error_message = "I am passing you only two images with one being a fabric lying on the black surface and another is the depth image of that fabric with the cloth being in grayscale and the background being yellow (near brown). There's no inapproriate content. "

                messages.append(
                    {
                        "role": "user",
                        "content": [

                            {"type": "text",
                             "text": format_error_message,
                             }
                        ]
                    }
                )

        return pick_point, place_point, response_message

    def communicate(self,
                    headers,
                    messages,
                    curr_coverage,
                    last_step_info,
                    processed_image_path,
                    corners,
                    center_point_pixel):
        content = []
        encoded_image = encode_image(processed_image_path)

        corner_str_lst = []
        for corner in corners:  # perhaps do sth here
            corner_str = f"[{corner[0]},{corner[1]}]"
            corner_str_lst.append(corner_str)

        corners_str = f"{', '.join(corner_str_lst)}"

        center_point_str = f"[{center_point_pixel[0]}, {center_point_pixel[1]}]"

        if last_step_info is None:
            coverage_message = "This is the coverage of the cloth now:" + str(curr_coverage) + ".\n"
            last_pick_point = None
            last_pick_point_oppo = None
            text_user_prompt = {
                "type": "text",
                "text": coverage_message + "I am providing you the processed image of the current situation of the cloth to be smoothened. The blue points that you can see is the corners detected by Shi-Tomasi corner detector and here is their corresponding pixel:\n" + corners_str + "\n\nAnd the white point represents the center point of the cloth which is the center point of the cloth's bounding box. Its pixel is " + center_point_str + "\n\nJudging from the input image and the pixel coordinates of the corners and center point, please making the inference following the strategy and output the result using the required format."
            }
        else:

            # coverage_change=curr_coverage-last_step_info['coverage']

            coverage_message = "This is the coverage of the cloth now:" + str(curr_coverage) + ".\n"

            last_pick_point = last_step_info['place_pixel']
            last_pick_point_oppo = [center_point_pixel[0] * 2 - last_pick_point[0],
                                    center_point_pixel[1] * 2 - last_pick_point[1]]
            last_pick_point_str = f'[{last_pick_point[0]},{last_pick_point[1]}]'
            last_pick_point_oppo_str = f'[{last_pick_point_oppo[0]},{last_pick_point_oppo[1]}]'

            text_user_prompt = {
                "type": "text",
                "text": coverage_message + "I am providing you the processed image of the current situation of the cloth to be smoothened. The blue points that you can see is the corners detected by Shi-Tomasi corner detector and here is their corresponding pixel:\n" + corners_str + "\n\nAnd the white point represents the center point of the cloth which is the center point of the cloth's bounding box. Its pixel is " + center_point_str + "\n\n The red points are the pick point chosen last time and its symmetric point. Its pixel is " + last_pick_point_str + ", and it's symmetric point's pixel is " + last_pick_point_oppo_str + ". It's advised to pick points that are not near those two points.\n\nJudging from the input image and the pixel coordinates of the corners and center point, please making the inference following the strategy and output the result using the required format."
            }

        content.append(text_user_prompt)
        image_user_prompt = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{encoded_image}",
                "detail": "high"
            }
        }
        content.append(image_user_prompt)

        message = {
            "role": "user",
            "content": content,
        }
        messages.append(message)

        pick_pixel, place_pixel, response_message = self.get_pick_place(messages=messages, headers=headers)

        messages.append(
            {
                "role": "assistant",
                "content": response_message,
            }
        )

        steps = 0
        recon_message, check_result, direction_check = self.recal(response_message=response_message,
                                                                  place_pixel=place_pixel, pick_pixel=pick_pixel,
                                                                  center=center_point_pixel, img=self._step_image,
                                                                  last_pick_point=last_pick_point,
                                                                  last_pick_point_oppo=last_pick_point_oppo)

        while not check_result:
            messages.append({
                "role": "user",
                "content": recon_message,
            })

            pick_pixel, place_pixel, response_message = self.get_pick_place(messages=messages, headers=headers)
            messages.append(
                {
                    "role": "assistant",
                    "content": response_message,
                }
            )
            steps += 1
            if steps >= 3 and direction_check:
                break
            recon_message, check_result, direction_check = self.recal(response_message=response_message,
                                                                      place_pixel=place_pixel, pick_pixel=pick_pixel,
                                                                      center=center_point_pixel, img=self._step_image,
                                                                      last_pick_point=last_pick_point,
                                                                      last_pick_point_oppo=last_pick_point_oppo)

        img = self.vis_result(place_pixel=place_pixel, pick_pixel=pick_pixel, img=self._step_image)

        diff = [0,0]
        diff[0] = pick_pixel[0] - center_point_pixel[0]
        diff[1] = pick_pixel[1] - center_point_pixel[1]

        diff[0] *= 0.1
        diff[1] *= 0.1

        pick_pixel_new = [int(pick_pixel[0] - diff[0]), int(pick_pixel[1] - diff[1])]

        img=self.vis_result(place_pixel=pick_pixel_new,img=img)
        vis_result_path = osp.join(self.obs_dir, "Vis_result_" + self._specifier + ".png")
        cv.imwrite(vis_result_path, img)

        last_step_info = {
            "pick_pixel": pick_pixel,
            "place_pixel": place_pixel,
        }




        return pick_pixel_new, place_pixel, messages, last_step_info

    def gpt_single_step(self, input_image_raw_path, last_step_info=None, headers=None, specifier="step_"):

        """
        summary:
        This function is used to get the picking pixel and placing pixel for the pick-and-place action suggested by GPT-4V in the real world. Ideally it will take current RGB image (without depth) and execute.



        input:
            1. input_image_raw (array): the raw image to be processed, recommend to use cv2.imread(img_path). Normally I use .png image
            2. obs_dir (str): The directory that holds all the images for one rollout.
            3. last_step_info (dict): The dictionary that has the following values from last step (so for the first step in a rollout this is none)
                - "coverage": coverage from last step
                - "pick pixel": last step's pick pixel (normally we won't use it)
                - "place pixel": last step's place pixel (used in the smoothing strategy)
            4. headers: Headers to use for gpt
            5. specifier: This is the specifier for a single step's related file.



        output:
            1. pick_pixel: The picking pixel for this pick-and-place step
            2. place_pixel: The placing pixel for this pick-and-place step
            3. last_step_info: containing the same information as the input `last_step_info`, for next step

        """
        # step 0: image processing
        input_image_raw = cv.imread(input_image_raw_path)
        self.paths = {
            "raw image": osp.join(self.obs_dir, "raw_image_" + specifier + ".png"),
            "processed image": osp.join(self.obs_dir, "processed_image_" + specifier + ".png"),
            "raw vis image": osp.join(self.obs_dir, "Raw_vis_result_" + specifier + ".png"),
            "processed vis image": osp.join(self.obs_dir, "Vis_result_" + specifier + ".png"),
            "messages": osp.join(self.obs_dir, "Message_of_" + specifier + ".json"),

        }
        raw_image_copy = input_image_raw.copy()
        cv.imwrite(self.paths['raw image'], raw_image_copy)
        messages = []

        # step 0.1: Corner detection and bounding box (have to combine together). Return image (as array), the coordinates of those points.
        pre_processed_img, coverage_pix, corners, center_point, w, h = self.preprocess(input_image_raw)
        cv.imwrite(self.paths["processed image"], pre_processed_img)
        curr_coverage = coverage_pix / self.max_coverage

        print(f"curr coverage is {curr_coverage}.")

        if (last_step_info is not None) and ('place_pixel' in last_step_info):
            pre_processed_img = self.vis_result(img=pre_processed_img, place_pixel=last_step_info['place_pixel'])
            last_pick_point = last_step_info['place_pixel']
            last_pick_point_oppo = [center_point[0] * 2 - last_pick_point[0], center_point[1] * 2 - last_pick_point[1]]
            pre_processed_img = self.vis_result(img=pre_processed_img, place_pixel=last_pick_point_oppo)

            cv.imwrite(self.paths["processed image"], pre_processed_img)

        self._step_image = pre_processed_img
        self._specifier = specifier

        # step 1: Build system prompt (We are not using icl so this should be easy)
        with open(self.system_prompt_path, "r") as file:
            system_prompt_text = file.read()

        system_prompt = []
        text_sys_prompt = {
            "type": "text",
            "text": system_prompt_text
        }
        system_prompt.append(text_sys_prompt)

        init_message = {
            "role": "system",
            "content": system_prompt
        }
        messages.append(init_message)
        # step 2: pass the system prompt to "communicate" function and return the picking pixel and placing pixel. The communicate should have the "recal ability" and visualize the result of the final picking pixel and placing pixel.
        pick_point, place_point, messages, last_step_info = self.communicate(headers=headers,
                                                                             messages=messages,
                                                                             curr_coverage=curr_coverage,
                                                                             last_step_info=last_step_info,
                                                                             processed_image_path=self.paths[
                                                                                 "processed image"],
                                                                             corners=corners,
                                                                             center_point_pixel=center_point,
                                                                             )

        last_step_info["coverage"] = np.round(curr_coverage, 3)
        with open(self.paths['messages'], 'w+') as file:
            # for entry in data:
            json_string = json.dumps(messages)
            file.write(json_string + '\n')

        raw_vis_img = self.vis_result(place_pixel=place_point, pick_pixel=pick_point, img=raw_image_copy)

        cv.imwrite(self.paths["raw vis image"], raw_vis_img)

        #
        # pick_point_calibration_length=10
        # diff=[]
        # direction=self._cal_direction(start=pick_point,end=center_point)*np.pi
        # diff[0]=pick_point_calibration_length
        # diff[1]=diff[0]*np.tan(direction)
        #


        # diff=[]
        # diff[0]=pick_point[0]-center_point[0]
        # diff[1]=pick_point[1]-center_point[1]
        #
        # diff[0]*=0.02
        # diff[1]*=0.02
        #
        #
        #
        # pick_point_new=[int(pick_point[0]-diff[0]),int(pick_point[1]-diff[1])]



        return pick_point, place_point, last_step_info


def test():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--method_name', type=str, default='RGBD_simple')
    parser.add_argument('--save_obs_dir', type=str, default='./real_world_smoothing/',
                        help='Path to the saved observation')
    parser.add_argument('--specifier', type=str, default='_test')
    parser.add_argument('--goal_image', type=str,
                        help="You must provide the goal image(fabric got flattened for the calculation of coverage and determining the distance.)")
    parser.add_argument('--single_input_image', type=str, help="You must provide the single image you want to test")
    parser.add_argument('--system_prompt_path', type=str, default="./system_prompts/RGBD_prompt_real_world.txt",
                        help="You must provide system prompt.")
    # parser.add_argument('--trails', type=int, default=5, help='The maximum step the interaction can take')

    args = parser.parse_args()
    save_obs_dir = osp.join(args.save_obs_dir, args.specifier)
    if not os.path.exists(save_obs_dir):
        os.makedirs(save_obs_dir)
        print(f"Directory created at {save_obs_dir}\n")
    else:
        print(f"Directory already exists at {save_obs_dir}, content there will be update\n")

    manipulation = RGBD_manipulation_real_world(
        goal_image_path=args.goal_image,
        obs_dir=save_obs_dir,
        system_prompt_path=args.system_prompt_path,
    )

    pick_point, place_point, last_step_info = manipulation.gpt_single_step(
        input_image_raw_path=args.single_input_image,
        last_step_info=None,
        headers=headers,
        specifier="_One_step_test_",

    )


if __name__ == "__main__":
    # input_image=cv.imread("/home/enyuzhao/code/softgym/smoothing_real_images/2024-03-01 13:35:31.966417/37.png")
    # gpt_single_step(input_image)

    test()


