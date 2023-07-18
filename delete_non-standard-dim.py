from PIL import Image
import os

def delete_images_with_different_sizes(folders, target_size=(2475, 3504)):
    # iterate over each folder
    for folder in folders:
        image_folder = os.path.join(folder, "Images")
        xml_folder = os.path.join(os.path.dirname(folder), "xml_by_page")
        # iterate over each file in the directory
        for filename in os.listdir(image_folder):
            # check if the file is an image
            if filename.endswith(".png"):
                # open the image file
                with Image.open(os.path.join(image_folder, filename)) as img:
                    if img.size != target_size:
                        os.remove(os.path.join(image_folder, filename))  # delete the image file
                        xml_file_path = os.path.join(xml_folder, filename.replace(".png", ".xml"))
                        if os.path.exists(xml_file_path):
                            os.remove(xml_file_path)  # delete the xml file
                        print(f"Deleted image and xml files for {filename}")

# specify your directories here
directories = ["/homes/es314/pitch_Mask_RCNN/only_position/homophonic_only_pos", "/homes/es314/pitch_Mask_RCNN/only_position/monophonic_only_pos", "/homes/es314/pitch_Mask_RCNN/only_position/pianoform_only_pos", "/homes/es314/pitch_Mask_RCNN/only_position/polyphonic_only_pos"]
delete_images_with_different_sizes(directories)
