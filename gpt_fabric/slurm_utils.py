import numpy as np
from PIL import Image
import cv2
import os
from matplotlib import pyplot as plt
import pickle
from pprint import pprint
from scipy.optimize import minimize, differential_evolution
from gpt_fabric_demo.gpt_fabric.utils.gpt_utils import analyze_images_gpt, get_user_prompt, system_prompt, \
    gpt_v_demonstrations, parse_output
import json
from openai import OpenAI


def save_depth_as_matrix(image_path, output_path=None, save_matrix=True, should_crop=True):
    '''
    This function takes the path of the image and saves the depth image in a form where the background is 0
    We would pass this matrix to the LLM API in order to get the picking and placing pixels
    '''
    image = Image.open(image_path)
    if should_crop:
        if image.size != 128:
            image = image.resize((128, 128))

    image_array = np.array(image) / 255

    mask = image_array.copy()
    mask[mask > 0.646] = 0
    mask[mask != 0] = 1

    image_array = image_array * mask
    image_array = image_array * 100
    if save_matrix:
        np.savetxt(output_path, np.round(image_array, decimals=2), fmt='%.2f')
    return image_array


def find_pixel_center_of_cloth(image_path, should_crop=True):
    '''
    This function would be used to get the pixel center corresponding to the initial cloth configuration
    '''
    # image_matrix = save_depth_as_matrix(image_path, None, False, should_crop)

    cropped_rgb_img = cv2.imread(image_path)

    cropped_rgb_img = cv2.cvtColor(cropped_rgb_img, cv2.COLOR_BGR2RGB)

    gray = cv2.cvtColor(cropped_rgb_img, cv2.COLOR_BGR2GRAY)

    sum_rgb = np.sum(cropped_rgb_img, axis=2)

    mask = sum_rgb > 200

    inverse_mask = ~mask

    gray[mask] = 0

    gray[inverse_mask] = 250

    # Find indices of non-zero values
    nonzero_indices = np.nonzero(gray)

    # Calculate the center coordinates
    center_x = int(np.mean(nonzero_indices[1]))
    center_y = int(np.mean(nonzero_indices[0]))

    return np.array([center_x, center_y])


def find_corners(image_path, should_crop=True):
    '''
    This function will use the OpenCV methods to detect the cloth corners from the given depth image
    '''
    # image_matrix = save_depth_as_matrix(image_path, None, False, should_crop)
    # cv2.imwrite("./to_be_deleted.png", image_matrix)
    #
    # img = cv2.imread("./to_be_deleted.png")
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cropped_rgb_img = cv2.imread(image_path)

    cropped_rgb_img = cv2.cvtColor(cropped_rgb_img, cv2.COLOR_BGR2RGB)

    gray = cv2.cvtColor(cropped_rgb_img, cv2.COLOR_BGR2GRAY)

    sum_rgb = np.sum(cropped_rgb_img, axis=2)

    mask = sum_rgb > 200

    inverse_mask = ~mask

    gray[mask] = 0

    gray[inverse_mask] = 250

    blurred_image = cv2.GaussianBlur(gray, (9, 9), 0)

    # Threshold the blurred image
    _, thresholded_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Define a kernel for morphological operations
    kernel = np.ones((10, 10), np.uint8)

    # Perform dilation to expand the white regions
    dilated_image = cv2.dilate(thresholded_image, kernel, iterations=1)

    # Perform erosion to shrink the white regions
    eroded_image = cv2.erode(dilated_image, kernel, iterations=1)

    # cv2.imshow('Original Bounding Box', thresholded_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # cv2.imshow('Padded Bounding Box', eroded_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Now get the corners and show them
    corner_coordinates = cv2.goodFeaturesToTrack(image=eroded_image, maxCorners=4, qualityLevel=0.1, minDistance=150,
                                                 useHarrisDetector=True)
    corner_coordinates = np.intp(corner_coordinates)

    # Implementing Harris corner detection by myself to get the corners. Does not work though
    # corners = cv2.cornerHarris(gray,2,3,0.04)
    # corners_thresholded = corners > 0.01 * corners.max()
    # corner_coordinates = np.array(np.where(corners_thresholded)).T

    # Using OpenCV.goodFeaturesToTrack() function to get the corners
    # corner_coordinates = cv2.goodFeaturesToTrack(image=gray, maxCorners=27, qualityLevel=0.04, minDistance=10,
    #                                              useHarrisDetector=True)
    # corner_coordinates = np.intp(corner_coordinates)

    # Plotting the original image with the detected corners
    for i in corner_coordinates:
        x, y = i.ravel()
        cv2.circle(cropped_rgb_img, (x, y), 3, 255, -1)
    plt.imshow(cropped_rgb_img)
    corner_img_name = image_path.rsplit(".", 1)[0]
    plt.savefig(corner_img_name + "_corner.png")

    # os.remove("./to_be_deleted.png")

    return corner_coordinates


### These functions are all corresponding to the helper functions that we'd use for the real world experiments
def crop_input_image(input_rgb, cropped_gray_image, cropped_rgb_image):
    '''
    This function call will be used for taking the input RGB and Depth images to be cropped
    The final saved image should be a depth image with the cloth around the center
    Note that this function call returns the pivot pixel coordinate for handling the real pick, place pixels
    '''
    # Load the image
    image = cv2.imread(input_rgb)
    pivot_coordinate = np.array([0, 0])

    # Crop the image initially since there's a big bounding box present already
    height, width = image.shape[:2]
    image = image[20:height - 20, 20:width - 20]
    pivot_coordinate += np.array([20, 20])

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply edge detection
    edges = cv2.Canny(gray, 50, 150)

    # Find contours in the edge-detected image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_area = -10
    max_contour_index = -1
    for (i, contour) in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        if w * h > max_area:
            max_area = w * h
            max_contour_index = i

    max_contour = contours[max_contour_index]

    # Get the bounding box of the closest contour
    x, y, w, h = cv2.boundingRect(max_contour)

    # Add padding around the bounding box
    padding = 40
    x -= padding
    y -= padding
    w += 2 * padding
    h += 2 * padding

    # Ensure the bounding box stays within the image bounds
    x = max(0, x)
    y = max(0, y)
    w = min(image.shape[1] - x, w)
    h = min(image.shape[0] - y, h)

    # Crop the region of interest with padding
    cropped_image = image[y:y + h, x:x + w]
    pivot_coordinate += np.array([x, y])

    cropped_gray = gray[y:y + h, x:x + w]
    cv2.imwrite(cropped_gray_image, cropped_gray)
    cv2.imwrite(cropped_rgb_image, cropped_image)

    # Display the cropped image to see if it's all working alright :)
    # cv2.imshow('Cropped Image', cropped_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Return the pivot pixel coordinates to be used later
    return pivot_coordinate


def get_initial_cloth_center(initial_input_rgb, intial_cropped_gray_image, intial_cropped_rgb_image):
    '''
    This function takes the RGB and Depth images of the initial cloth configuration and returns the actual pixel coordinate for the center
    '''
    # First crop the initial cloth images to get the new depth image
    initial_pivot_coordinate = crop_input_image(initial_input_rgb, intial_cropped_gray_image, intial_cropped_rgb_image)

    # Call the function to get the center for the saved initial cropped depth image
    initial_local_cloth_center = find_pixel_center_of_cloth(intial_cropped_rgb_image, False)

    # Add the pivot coordinate to return the true coordinates of the cloth center
    return initial_pivot_coordinate + np.array(initial_local_cloth_center)


def gpt_for_real_world(input_rgb, cropped_gray_image, cropped_rgb_image, cloth_center, task, current_step):
    '''
    This function call will be used for calling the overall GPT pipeline for the real-world experiments
    Args:
        input_rgb: The file path corresponding to the RGB image of the current cloth configuration
        input_depth: The file path corresponding to the depth image of the current cloth configuration
        cropped_depth_image: The file path where the cropped depth image is expected to be saved
        cloth_center: The pixel coordinates for the initial cloth center in the actual image
        task: The folding task that we wish to perform. Use one of DoubleTriangle, AllCornersInward, CornersEdgesInward, DoubleStraight
        current_step: The number of folding steps executed for the current test case thus far (starts with 0)
    '''
    # Setting up the chain to interact with OpenAI. Using Daniel's API key for now
    client = OpenAI(api_key="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")

    # Crop the input RGB and Depth images to get the cropped version of theirs
    pivot_coordinate = crop_input_image(input_rgb, cropped_gray_image, cropped_rgb_image)

    # Get the local cloth center for the current image configuration
    cloth_center = cloth_center - pivot_coordinate

    # Get the cloth corners for the current cloth configuration
    cloth_corners = find_corners(cropped_rgb_image, False)

    # Getting the template folding instruction images from the demonstrations
    demo_root_path = os.path.join(os.path.dirname(__file__), "data", "demo", task, "rgbviz")
    start_image = os.path.join(demo_root_path, str(current_step) + ".png")
    last_image = os.path.join(demo_root_path, str(current_step + 1) + ".png")
    instruction = analyze_images_gpt([start_image, last_image], task, current_step)

    # Getting the user prompt based on the information that we have so far
    user_prompt = get_user_prompt(cloth_corners, cloth_center, True, instruction, task, None)
    print("The user prompt was: ", user_prompt)

    # Getting the demonstrations that would be used for the specific task
    indices = gpt_v_demonstrations[task]["gpt-demonstrations"]

    # Imp: The information corresponding to the demonstrations is assumed to be in utils folder
    demonstration_dictionary_list = []
    gpt_demonstrations_path = os.path.join(os.path.dirname(__file__), "utils", "gpt-demonstrations", task, "demonstrations.json")
    with open(gpt_demonstrations_path, 'r') as f:
        gpt_demonstrations = json.load(f)

    # Fetching the information from the demonstrations as In-context data
    for index in indices:
        step_dictionary = gpt_demonstrations[str(index)][str(current_step + 1)]
        user_prompt_dictionary = {
            "role": "user",
            "content": step_dictionary["user-prompt"]
        }
        assistant_response_dictionary = {
            "role": "assistant",
            "content": step_dictionary["assistant-response"]
        }
        demonstration_dictionary_list += [user_prompt_dictionary, assistant_response_dictionary]

    # Making a call to the OpenAI API after we have the demonstrations
    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[
                     {
                         "role": "system",
                         "content": system_prompt
                     }] + demonstration_dictionary_list +
                 [{
                     "role": "user",
                     "content": user_prompt
                 }],
        temperature=0,
        max_tokens=769,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    # Parsing the above output to get the pixel coorindates for pick and place
    test_pick_pixel, test_place_pixel = parse_output(response.choices[0].message.content)
    print("The system response was: ", response.choices[0].message.content)
    test_pick_pixel = test_pick_pixel.astype(int)
    test_place_pixel = test_place_pixel.astype(int)

    # Returning the pick and the place point after adjusting with the pivot coordinate
    return pivot_coordinate + test_pick_pixel, pivot_coordinate + test_place_pixel
