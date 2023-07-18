from PIL import Image
import os
from collections import defaultdict

def check_image_dimensions(folders):
    # iterate over each folder
    for folder in folders:
        size_counts = defaultdict(list)  # Create a dictionary to store filenames of unique image sizes

        # iterate over each file in the directory
        for filename in os.listdir(folder):
            # check if the file is an image
            if filename.endswith(".png"):
                # open the image file
                with Image.open(os.path.join(folder, filename)) as img:
                    size_counts[img.size].append(filename)  # Add the filename to the list for this image size

        if len(size_counts) == 0:  # If no png images found in the directory
            print(f"No PNG images found in {folder}")
        else:  
            for size, filenames in size_counts.items():
                if size != (2475, 3504):
                    for filename in filenames:
                        print(filename)

# specify your directories here
directories = ["/homes/es314/pitch_Mask_RCNN/only_position/homophonic_only_pos/Images", "/homes/es314/pitch_Mask_RCNN/only_position/monophonic_only_pos/Images", "/homes/es314/pitch_Mask_RCNN/only_position/pianoform_only_pos/Images", "/homes/es314/pitch_Mask_RCNN/only_position/polyphonic_only_pos/Images"]
check_image_dimensions(directories)
