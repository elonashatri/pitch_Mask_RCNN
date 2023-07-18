import argparse
import os

import pandas as pd

# [{
#     "id": 1,
#     "name": "6stringTabClef"
# },


# item {
#   id: 1
#   name: '16th_flag'
# }


# []
# {
#     "id": 2,
#     "name": "accidentalDoubleSharp"
# },
# ]
#/homes/es314/DOREMI_version_2/1-4th_DOREMI/xml_annotations/classnames.csv
#/import/c4dm-05/elona/doremi_v3_1_4th/train_validation_test_records/classnames.csv
def generate_mapping(args):
    annotations = pd.read_csv(os.path.join(args.dataset_dir, "classnames.csv"))

    class_names = annotations["class_name"].unique()
    class_names.sort()


    with open(args.mapping_output_path, "w") as f:
        for i, class_name in enumerate(class_names):
            f.write("""
{{
    "id": {},
    "name": "{}"
}},
""".format(i + 1, class_name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", default="/homes/es314/pitch_Mask_RCNN/only_position/monophonic_only_pos/")
    parser.add_argument("--mapping_output_path", default="/homes/es314/pitch_Mask_RCNN/mapping.json")
    args = parser.parse_args()

    generate_mapping(args)
