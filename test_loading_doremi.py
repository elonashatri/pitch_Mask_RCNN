import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
from tqdm import tqdm
from xml.dom import minidom
import glob
from sklearn.model_selection import train_test_split

# Assuming that these paths are defined in your project
ROOT_PATH = "/homes/es314/pitch_Mask_RCNN/only_position/"
FOLDERS = ["generated_data_only_pos", "monophonic_only_pos", "polyphonic_only_pos", "pianoform_only_pos"]

class DoremiDataset:
    def __init__(self):
        self.classname_set = set()

    def load_Doremi(self):
        all_xml_files = []
        for folder in FOLDERS:
            dataset_dir = ROOT_PATH + folder + "/xml_by_page/*.xml"
            print(dataset_dir)
            available_xml = glob.glob(dataset_dir)
            all_xml_files.extend(available_xml)

        return all_xml_files

    def split_data(self, all_xml_files):
        train_files, test_files = train_test_split(all_xml_files, test_size=0.3, random_state=42)
        val_files, test_files = train_test_split(test_files, test_size=0.5, random_state=42)
        return train_files, val_files, test_files

    def process_files(self, files):
        for xml_file in tqdm(files, desc="XML Files"):
            filename = os.path.basename(xml_file)
            filename = filename[:-4]

            xmldoc = minidom.parse(xml_file)

            img_filename = filename + '.png'

            img_path = ROOT_PATH + "Images/" + img_filename  # assuming the images are stored in the "Images" subfolder
            img_height = 3504
            img_width = 2474

            nodes = xmldoc.getElementsByTagName('Node')

            masks_info = []
            for node in nodes:
                this_mask_info = {}

                node_classname_el = node.getElementsByTagName('ClassName')[0]
                node_classname = node_classname_el.firstChild.data
                self.classname_set.add(node_classname)

                node_top = node.getElementsByTagName('Top')[0]
                node_top_int = int(node_top.firstChild.data)

                node_left = node.getElementsByTagName('Left')[0]
                node_left_int = int(node_left.firstChild.data)

                node_width = node.getElementsByTagName('Width')[0]
                node_width_int = int(node_width.firstChild.data)

                node_height = node.getElementsByTagName('Height')[0]
                node_height_int = int(node_height.firstChild.data)

                node_mask = str(node.getElementsByTagName('Mask')[0].firstChild.data)
                node_mask = node_mask.replace('0: ', '')
                node_mask = node_mask.replace('1: ', '')
                split_mask = node_mask.split(' ')
                split_mask = split_mask[:-1]
                
                notehead_counts = list(map(int, list(split_mask)))

                this_mask_info["classname"] = node_classname
                this_mask_info["bbox_top"] = node_top_int
                this_mask_info["bbox_left"] = node_left_int
                this_mask_info["bbox_width"] = node_width_int
                this_mask_info["bbox_height"] = node_height_int
                this_mask_info["mask_arr"] = notehead_counts

                masks_info.append(this_mask_info)

            print("Image added: ID - {}, Path - {}, Width - {}, Height - {}, Masks Info - {}".format(img_filename, img_path, img_width, img_height, masks_info))

    def save_mapping(self, mapping_path):
        classname_list = list(self.classname_set)
        id_classname_dict = [{"id": i+1, "name": classname} for i, classname in enumerate(classname_list)]
        with open(mapping_path, 'w') as json_file:
            json.dump(id_classname_dict, json_file, indent=4)
        print("Mapping saved: {}".format(mapping_path))


# Test the DoremiDataset class
doremi_dataset = DoremiDataset()
all_xml_files = doremi_dataset.load_Doremi()
train_files, val_files, test_files = doremi_dataset.split_data(all_xml_files)

print("Processing training files...")
doremi_dataset.process_files(train_files)
print("Processing validation files...")
doremi_dataset.process_files(val_files)
print("Processing test files...")
doremi_dataset.process_files(test_files)

# doremi_dataset.save_mapping("/homes/es314/pitch_Mask_RCNN/mapping.json")
