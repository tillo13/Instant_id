import os
import random
import csv
import gc
import glob
from datetime import datetime
import time
from pathlib import Path
from style_template import style_list
from PIL import Image, ImageOps

# Default Configuration variables
INPUT_FOLDER_NAME = 'sample_images'
OUTPUT_FOLDER_NAME = 'generated_images'
LOG_FILENAME = 'generation_log.csv'
logfile_path = os.path.join(os.getcwd(), LOG_FILENAME)

PROMPT = "human, sharp focus"
NEGATIVE_PROMPT = "(blurry, blur, text, abstract, glitch, lowres, low quality, worst quality:1.2), (text:1.2), watermark, painting, drawing, illustration, glitch, deformed, mutated, cross-eyed, ugly, disfigured"
IDENTITYNET_STRENGTH_RATIO_RANGE = (1.0, 1.5)
ADAPTER_STRENGTH_RATIO_RANGE = (0.7, 1.0)
NUM_INFERENCE_STEPS_RANGE = (40, 60)
GUIDANCE_SCALE_RANGE = (7.0, 12.0)
MAX_SIDE = 1280
MIN_SIDE = 1024
NUMBER_OF_LOOPS = 1

# Dynamically create the STYLES list from imported style_list
STYLES = [style["name"] for style in style_list]
USE_RANDOM_STYLE = False

def choose_random_style():
    return random.choice(STYLES)

def get_random_image_file(input_folder):
    valid_extensions = [".jpg", ".jpeg", ".png"]
    files = [file for file in Path(input_folder).glob("*") if file.suffix.lower() in valid_extensions]
    if not files:
        raise FileNotFoundError(f"No images found in directory {input_folder}")
    return str(random.choice(files))

def resize_and_pad_image(image_path, max_side, min_side, pad_color=(255, 255, 255)):
    # Open an image using PIL
    image = Image.open(image_path)

    # Calculate the scale and new size
    ratio = min(min_side / min(image.size), max_side / max(image.size))
    new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))

    # Resize the image
    image = image.resize(new_size, Image.BILINEAR)
    
    # Calculate padding
    delta_w = max_side - new_size[0]
    delta_h = max_side - new_size[1]

    # Pad the resized image to make it square
    padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
    image = ImageOps.expand(image, padding, pad_color)

    return image

def log_to_csv(logfile_path, image_name, new_file_name='Unknown', identitynet_strength_ratio=0.0, adapter_strength_ratio=0.0, num_inference_steps=0, guidance_scale=0.0, seed=0, success=True, error_message='', style_name="", prompt="", negative_prompt="", time_taken=0.0, current_timestamp=""):
    os.makedirs(os.path.dirname(logfile_path), exist_ok=True)
    file_exists = os.path.isfile(logfile_path)

    with open(logfile_path, 'a', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['image_name', 'new_file_name', 'identitynet_strength_ratio', 'adapter_strength_ratio', 'num_inference_steps', 'guidance_scale', 'seed', 'success', 'error_message', 'style_name', 'prompt', 'negative_prompt', 'time_taken', 'current_timestamp']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            'image_name': image_name,
            'new_file_name': new_file_name,
            'identitynet_strength_ratio': identitynet_strength_ratio,
            'adapter_strength_ratio': adapter_strength_ratio,
            'num_inference_steps': num_inference_steps,
            'guidance_scale': guidance_scale,
            'seed': seed,
            'success': success,
            'error_message': error_message,
            'style_name': style_name,
            'prompt': prompt,
            'negative_prompt': negative_prompt,
            'time_taken': time_taken,
            'current_timestamp': current_timestamp
        })

def initial_image(generate_image_func):
    overall_start_time = time.time()
    total_time_taken = 0.0 

    # Initialize a counter for processed images at the beginning of the function
    processed_images_count = 0

    # List all image files in the `INPUT_FOLDER_NAME`
    image_files = glob.glob(f'{INPUT_FOLDER_NAME}/*.png') + \
                  glob.glob(f'{INPUT_FOLDER_NAME}/*.jpg') + \
                  glob.glob(f'{INPUT_FOLDER_NAME}/*.jpeg')
    
    # Check if we found any images
    if not image_files:
        raise FileNotFoundError(f"No images found in directory {INPUT_FOLDER_NAME}")
    
    # Print the count of detected image files
    print(f"Processing a total of {len(image_files)} image(s) in '{INPUT_FOLDER_NAME}'")

    # Shuffle the image files randomly
    random.shuffle(image_files)
    
    total_images = len(image_files)  # Get the total number of images to process
    
    for loop in range(NUMBER_OF_LOOPS):
        print(f"Starting loop {loop+1} of {NUMBER_OF_LOOPS}")

        for image_number, face_image_path in enumerate(image_files, start=1):
            loop_start_time = datetime.now()
            face_image = [face_image_path]
            basename = os.path.basename(face_image_path)
            processed_images_count += 1

            # Resize and pad the image before processing
            processed_image = resize_and_pad_image(
                image_path=face_image_path,
                max_side=MAX_SIDE,
                min_side=MIN_SIDE
            )

            if USE_RANDOM_STYLE:
                style_name = choose_random_style()
            else:
                style_name = "(No style)"

            identitynet_strength_ratio = random.uniform(*IDENTITYNET_STRENGTH_RATIO_RANGE)
            adapter_strength_ratio = random.uniform(*ADAPTER_STRENGTH_RATIO_RANGE)
            num_inference_steps = random.randint(*NUM_INFERENCE_STEPS_RANGE)
            guidance_scale = random.uniform(*GUIDANCE_SCALE_RANGE)
            seed = random.randint(0, 2**32 - 1)
            
            # Print settings for the current image BEFORE processing it
            print_generation_settings(basename, style_name, identitynet_strength_ratio, 
                                      adapter_strength_ratio, num_inference_steps, guidance_scale, seed,
                                      image_number, total_images)

            # Here, the generate_image_func is supposedly called and image processing happens
            _, _, generated_file_paths = generate_image_func(
                face_image=face_image,
                pose_image=None,
                prompt=PROMPT,
                negative_prompt=NEGATIVE_PROMPT,
                style_name=style_name,
                enhance_face_region=True,
                num_steps=num_inference_steps,
                identitynet_strength_ratio=identitynet_strength_ratio,
                adapter_strength_ratio=adapter_strength_ratio,
                guidance_scale=guidance_scale,
                seed=seed
            )
            
            loop_end_time = datetime.now()
            loop_time_taken = (loop_end_time - loop_start_time).total_seconds()

            # Immediately print the time taken and current time.
            print(f"Time taken to process image: {loop_time_taken:.2f} seconds")

            # Update the total time taken with this image's processing time
            total_time_taken += loop_time_taken

            # Calculate the average time taken per image
            average_time_per_image = total_time_taken / image_number

            current_timestamp = loop_end_time.strftime("%Y-%m-%d %H:%M:%S")  # Current time after processing
            print(f"Current timestamp: {current_timestamp}")

            # Calculate estimated remaining time considering the images left in this loop and the additional loops
            remaining_images_this_loop = total_images - image_number
            remaining_images_in_additional_loops = (NUMBER_OF_LOOPS - (loop + 1)) * total_images
            total_remaining_images = remaining_images_this_loop + remaining_images_in_additional_loops
            estimated_time_remaining = average_time_per_image * total_remaining_images

            # Display the estimated time remaining including remaining loops
            print(f"Estimated time remaining (including loops): {estimated_time_remaining // 60:.0f} minutes, {estimated_time_remaining % 60:.0f} seconds")

            # Display the overall average time per image in seconds
            print(f"Overall average time per image: {average_time_per_image:.2f} seconds")

            # Display the total number of remaining images to process including looping
            print(f"Total remaining images to process (including loops): {total_remaining_images}")


            success = True  # Assuming generation was successful.
            error_message = ""  # Assuming no error.

            # Log to CSV after the image generation.
            for generated_file_path in generated_file_paths:
                new_file_name = os.path.basename(generated_file_path)
                log_to_csv(logfile_path, basename, new_file_name, identitynet_strength_ratio,
                           adapter_strength_ratio, num_inference_steps, guidance_scale, seed, success,
                           error_message, style_name, PROMPT, NEGATIVE_PROMPT, loop_time_taken, current_timestamp)
            
                
            del generated_file_paths  # Explicitly delete large variables
            gc.collect()  # Call garbage collection


    # At the end of the initial_image() function, add:
    total_elapsed_time = time.time() - overall_start_time
    print("\n===FINAL SUMMARY===")
    print(f"Total loops completed: {NUMBER_OF_LOOPS}")
    print(f"Total images processed per loop: {len(image_files)}")
    print(f"Overall total images processed: {NUMBER_OF_LOOPS * len(image_files)}") # Multiplied by the number of loops
    print(f"Overall total time: {total_elapsed_time / 60:.2f} minutes")

               
def print_generation_settings(basename, style_name, identitynet_strength_ratio, adapter_strength_ratio, num_inference_steps, guidance_scale, seed, image_number, total_images):
    print("===IMAGE GENERATION DATA SUMMARY===")
    # Print settings for the current image
    print(f"- Image {image_number} of {total_images}\n"
          f"- Filename: {basename}\n"
          f"- Style: {style_name}\n"
          f"- IdentityNet strength ratio: {identitynet_strength_ratio:0.2f}\n"
          f"- Adapter strength ratio: {adapter_strength_ratio:0.2f}\n"
          f"- Number of inference steps: {num_inference_steps}\n"
          f"- Guidance scale: {guidance_scale:0.2f}\n"
          f"- Seed: {seed}\n"
          f"- Input folder name: {INPUT_FOLDER_NAME}\n"
          f"- Output folder name: {OUTPUT_FOLDER_NAME}\n"
          f"- Prompt: {PROMPT}\n"
          f"- Negative prompt: {NEGATIVE_PROMPT}\n"
          f"- Number of loops: {NUMBER_OF_LOOPS}\n"
          f"- Use random style: {USE_RANDOM_STYLE}\n")
    print("===DEFINING COMPLETE, GENERATING IMAGE...===")