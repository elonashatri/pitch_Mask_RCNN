import os
import glob

# Define the root directory for your XML files
xml_root_dir = '/homes/es314/pitch_Mask_RCNN/only_position/train_test_val_records/'

# Define the directory for your image files
img_root_dir = '/homes/es314/pitch_Mask_RCNN/only_position/Images'

# Define the subsets you want to go through
subsets = ['train', 'val', 'test']

# Iterate through each subset
for subset in subsets:
    # Get all XML files in the subset
    xml_files = glob.glob(os.path.join(xml_root_dir, subset, '*.xml'))
    
    for xml_file in xml_files:
        xml_filename = os.path.basename(xml_file)
        # Remove the extension to get the base filename
        base_filename = os.path.splitext(xml_filename)[0]
        
        # Generate the corresponding image filename
        img_filename = base_filename + '.png'
        
        # Construct the full path to the image
        img_path = os.path.join(img_root_dir, img_filename)
        
        # Check if the corresponding image exists
        if not os.path.exists(img_path):
            print(f"Deleting {xml_file} as corresponding image was not found.")
            os.remove(xml_file)  # Remove the XML file
