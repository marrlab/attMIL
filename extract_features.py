import sys
import os
import cv2
import zviloader
import gzip
import pickle
import mrcnn.model as modellib
from mrcnn.config import Config

# Load the mask R-CNN model and set the config for inference
ROOT_DIR = os.path.abspath(".")
sys.path.append(ROOT_DIR)
MODEL_DIR = os.path.join(ROOT_DIR, "model")


class RBCConfig(Config):
    NAME = "coco"
    IMAGES_PER_GPU = 2
    NUM_CLASSES = 1 + 1


config = RBCConfig()

model = modellib.MaskRCNN(mode="inference", config=config, model_dir=MODEL_DIR)
model.load_weights(os.path.join(MODEL_DIR, "maskrcnn/mask_rcnn_coco_0004.h5"), by_name=True)


# Path of the dataset directory organized as: Section >> Class >> Patient >> Sample >> Image files

dataset_root_dir = "/Users/ario.sadafi/Data/UZH/UZH-Organized/MILDATAset"

section = "Test"  # rerun for every section, Train, Val, Test

for cls in os.listdir(os.path.join(dataset_root_dir,section)):
    if cls.startswith('.'): continue
    for patient in os.listdir(os.path.join(dataset_root_dir, section, cls)):
        if patient.startswith('.'): continue
        for sample in os.listdir(os.path.join(dataset_root_dir, section, cls, patient)):
            if sample.startswith('.'): continue

            id = 0
            # in meta data, save the class label, patient id, sample id, and list of the images analyzed.
            meta = {'label': cls,
                    'patient': patient,
                    'sample': sample,
                    'images_in_dir:': len(os.listdir(os.path.join(dataset_root_dir,section, cls, patient, sample)))
            }
            data = []

            # for every image in a sample, read all image files (zvi or jpg)
            for image in os.listdir(os.path.join(dataset_root_dir, section, cls,patient,sample)):
                if os.path.isdir(os.path.join(dataset_root_dir, section, cls, patient, sample, image)): continue
                if image.endswith('zvi'):
                    # print(os.path.join(dataset_root_dir,cls,patient,sample,image))
                    try:
                        im = zviloader.zviloader(os.path.join(dataset_root_dir, section, cls, patient, sample, image))
                    except:
                        continue
                elif image.endswith('png'):
                    im = cv2.imread(os.path.join(dataset_root_dir,section, cls,patient,sample,image))

                else:
                    continue

                # Do a forward pass and extract the features
                r = model.detect([im, im])[0]

                data.append({'image_filename': image,
                        'image': im,
                        'feats': r["feats"],
                        'rois': r["rois"],
                        })

            # save all of the extracted features as well as the meta data
            with gzip.open("data-features/" + section + "/" + cls + "_" + patient + "_" + sample + ".dat", 'wb') as f:
                pickle.dump({"meta": meta, "data": data}, f)
