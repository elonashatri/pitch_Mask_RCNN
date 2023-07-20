"""
Mask R-CNN
Train on the toy Doremi dataset and implement color splash effect.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 doremi_train.py train --dataset=/path/to/Doremi/dataset --weights=coco
 python3 doremi_train.py train --dataset=/homes/es314/pitch_Mask_RCNN/only_position/train_test_val_records --weights=coco
    # Resume training a model that you had trained earlier
    python3 Doremi.py train --dataset=/path/to/Doremi/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 Doremi.py train --dataset=/path/to/Doremi/dataset --weights=imagenet

    # Apply color splash to an image
    python3 Doremi.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    # Apply color splash to video using the last weights you trained
    python3 Doremi.py splash --weights=last --video=<URL or path to file>
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
import sys
print(sys.executable)
import os
import json
import datetime
import numpy as np
import skimage.draw
import glob
from tqdm import tqdm
from xml.dom import minidom
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.saving import hdf5_format

# Root directory of the project
ROOT_DIR = os.path.abspath("/homes/es314/pitch_Mask_RCNN/only_position/train_test_val_records")

# Import Mask RCNN
# sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = COCO_WEIGHTS_PATH = '/homes/es314/pitch_Mask_RCNN/logs/mask_rcnn_Doremi.h5'

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = '/homes/es314/pitch_Mask_RCNN/logs'

############################################################
#  Configurations
############################################################


class DoremiConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "Doremi"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 50 + 1  # Background + Doremi

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9

    IMAGE_MIN_DIM = 800
    IMAGE_MAX_DIM = 1024


############################################################
#  Dataset
############################################################
ROOT_PATH = "/homes/es314/pitch_Mask_RCNN/only_position/"
# FOLDERS = ["homophonic_data_only_pos", "monophonic_only_pos", "polyphonic_only_pos", "pianoform_only_pos"]

############################################################
#  Dataset
############################################################

class DoremiDataset(utils.Dataset):
    def load_Doremi(self, subset):
        """
        Load a subset of the Doremi dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        with open('classnames.json') as json_file:  # replace with your classnames file
            data = json.load(json_file)
            for id_class in data:
                self.add_class("doremi", id_class["id"], id_class["name"])

        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir ='/homes/es314/pitch_Mask_RCNN/only_position/train_test_val_records/'+subset+'/*.xml'  # replace with your xml data path
        print("--------------", dataset_dir)
        available_xml = glob.glob(dataset_dir)

        for xml_file in tqdm(available_xml, desc="XML Files"):
            filename = os.path.basename(xml_file)
            filename = filename[:-4]  # Remove .xml from end of file

            xmldoc = minidom.parse(xml_file)

            page = xmldoc.getElementsByTagName('Page')
            # page_index_str = page[0].attributes['pageIndex'].value

            # page_index_int = int(page_index_str) + 1
            # leading_zeroes = str(page_index_int).zfill(3)
            img_filename = filename
            img_filename = img_filename+'.png'

            img_path = '/homes/es314/pitch_Mask_RCNN/only_position/Images/' + img_filename  # replace with your image data path
            img_height = 3504
            img_width = 2474

            nodes = xmldoc.getElementsByTagName('Node')

            instances_count = len(xmldoc.getElementsByTagName('ClassName'))

            masks_info = []
            for node in nodes:
                this_mask_info = {}

                node_classname_el = node.getElementsByTagName('ClassName')[0]
                node_classname = node_classname_el.firstChild.data
                
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

            self.add_image(
                    "doremi",
                    image_id=img_filename,  
                    path=img_path,
                    img_width=img_width, img_height=img_height,
                    masks_info=masks_info)

    # ... continue with your load_mask and image_reference functions here

            # print("Image added: ID - {}, Path - {}, Width - {}, Height - {}, Masks Info - {}".format(img_filename, img_path, img_width, img_height, masks_info))

    # def save_mapping(self, mapping_path):
    #     classname_list = list(self.classname_set)
    #     id_classname_dict = [{"id": i+1, "name": classname} for i, classname in enumerate(classname_list)]
    #     with open(mapping_path, 'w') as json_file:
    #         json.dump(id_classname_dict, json_file, indent=4)
    #     print("Mapping saved: {}".format(mapping_path))


def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = DoremiDataset("train")
    dataset_train.load_Doremi("train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = DoremiDataset("val")
    dataset_val.load_Doremi("val")
    dataset_val.prepare()
    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=30,
                layers='heads')


def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash


def detect_and_color_splash(model, image_path=None, video_path=None):
    assert image_path or video_path

    # Image or video?
    if image_path:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(args.image))
        # Read image
        image = skimage.io.imread(args.image)
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        # Color splash
        splash = color_splash(image, r['masks'])
        # Save output
        file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        skimage.io.imsave(file_name, splash)
    elif video_path:
        import cv2
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer
        file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))

        count = 0
        success = True
        while success:
            print("frame: ", count)
            # Read next image
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                # Detect objects
                r = model.detect([image], verbose=0)[0]
                # Color splash
                splash = color_splash(image, r['masks'])
                # RGB -> BGR to save image to video
                splash = splash[..., ::-1]
                # Add image to video writer
                vwriter.write(splash)
                count += 1
        vwriter.release()
    print("Saved to ", file_name)


############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect Doremis.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/Doremi/dataset/",
                        help='Directory of the Doremi dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "splash":
        assert args.image or args.video,\
               "Provide --image or --video to apply color splash"

    print("Weights: ", args.weights)
    # print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = DoremiConfig()
    else:
        class InferenceConfig(DoremiConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "splash":
        detect_and_color_splash(model, image_path=args.image,
                                video_path=args.video)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))
