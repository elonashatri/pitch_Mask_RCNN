import os
import glob
import shutil
from sklearn.model_selection import train_test_split

# Assuming that these paths are defined in your project
ROOT_PATH = "/homes/es314/pitch_Mask_RCNN/only_position/"
FOLDERS = ["generated_data_only_pos", "monophonic_only_pos", "polyphonic_only_pos", "pianoform_only_pos"]

class DoremiDataset:
    def __init__(self):
        self.dataset_path = ROOT_PATH  # adjust according to your needs
        self.train_dir = os.path.join(self.dataset_path, "train")
        self.val_dir = os.path.join(self.dataset_path, "val")
        self.test_dir = os.path.join(self.dataset_path, "test")

        # Create the directories if they don't exist
        os.makedirs(self.train_dir, exist_ok=True)
        os.makedirs(self.val_dir, exist_ok=True)
        os.makedirs(self.test_dir, exist_ok=True)

    def load_Doremi(self):
        all_xml_files = []
        for folder in FOLDERS:
            dataset_dir = os.path.join(self.dataset_path, folder, "xml_by_page", "*.xml")
            available_xml = glob.glob(dataset_dir)
            all_xml_files.extend(available_xml)

        return all_xml_files

    def split_data(self, all_xml_files):
        train_files, test_files = train_test_split(all_xml_files, test_size=0.3, random_state=42)
        val_files, test_files = train_test_split(test_files, test_size=0.5, random_state=42)

        # Copy the files into the new directories
        for f in train_files:
            shutil.copy(f, self.train_dir)
        for f in val_files:
            shutil.copy(f, self.val_dir)
        for f in test_files:
            shutil.copy(f, self.test_dir)

        return train_files, val_files, test_files


# Test the DoremiDataset class
doremi_dataset = DoremiDataset()
all_xml_files = doremi_dataset.load_Doremi()
train_files, val_files, test_files = doremi_dataset.split_data(all_xml_files)
