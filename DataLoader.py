import numpy as np
import os
import torch
import torch.utils.data as data_utils
from torchvision import datasets, transforms
import pickle
import gzip
from zviloader import zviloader
import cv2



class DataLoader:

    def __init__(self, foldnumber, mode="smp", imgmode="bag", train=True):

        self.mode = mode
        self.imgmode = imgmode
        self.train = train
        self._load_classes()
        self._load_folds(foldnumber)
        self.current_fold = foldnumber


        if self.imgmode == "bag":
            if self.train:
                self.train_bags_list, self.train_labels_list, self.train_meta_list = self._create_bags()
            else:
                self.test_bags_list, self.test_labels_list, self.test_meta_list = self._create_bags()
        elif self.imgmode == "img":
            path = "/Users/ario.sadafi/Data/UZH-Organized/MILDATAset/"
            if self.train:
                self.train_img_list, self.train_labels_list, self.train_meta_list = self._create_imageList(path)
            else:
                self.test_img_list, self.test_labels_list, self.test_meta_list = self._create_imageList(path)
        else:
            print("Wrong imgmode")
            exit()


    def _create_imageList(self, path):
        print("Loading Data...")

        if self.train:
            file_list = self.train_list
        else:
            file_list = self.test_list




        label_list = []
        img_list = []
        meta_list = []

        i = 0
        for bits in os.listdir(path):
            if bits.startswith("."): continue
            clss = os.listdir(os.path.join(path,bits))
            for cls in clss:
                if cls not in self.classes:
                    continue
                patients = os.listdir(os.path.join(path,bits , cls))
                for p in patients:
                    if p.startswith("."): continue

                    found = False
                    for f in file_list:
                        if f.split("_")[0].lower() == cls.lower() and f.split("_")[1].lower() == p.lower():
                            found = True
                            break;

                    if not found:
                        continue

                    samples = os.listdir(os.path.join(path,bits , cls, p))
                    for s in samples:
                        if s.startswith("."): continue
                        imgs = os.listdir(os.path.join(path, bits, cls, p, s))
                        bag = [os.path.join(path,bits, cls, p, s, im) for im in imgs if not im.startswith(".")]
                        img_list.append(bag)
                        label_list.append(self.classes.index(cls))

        return [img_list, label_list, meta_list]


    def _create_bags(self):
        print("Loading Data...")
        if self.train:
            if os.path.exists("data-features/train-"+ self.mode + str(self.current_fold) + ".pkl"):
                with gzip.open("data-features/train-"+ self.mode + str(self.current_fold) + ".pkl", "rb") as f:
                    [label_list, bag_list, meta_list] = pickle.load(f)
            else:
                [label_list, bag_list, meta_list] = self._analyzePath()
                with gzip.open("data-features/train-" + self.mode + str(self.current_fold) + ".pkl", 'wb') as f:
                    pickle.dump([label_list, bag_list, meta_list], f)
        else:
            if os.path.exists("data-features/test-"+ self.mode + str(self.current_fold) + ".pkl"):
                with gzip.open("data-features/test-"+ self.mode + str(self.current_fold) + ".pkl", "rb") as f:
                    [label_list, bag_list, meta_list] = pickle.load(f)
            else:
                [label_list, bag_list, meta_list] = self._analyzePath()
                with gzip.open("data-features/test-"+ self.mode + str(self.current_fold) + ".pkl", 'wb') as f:
                    pickle.dump([label_list, bag_list, meta_list], f)

        print("Done")
        label_list = self._correctlabellist(label_list)
        return bag_list, label_list, meta_list



    def _analyzePath(self):
        label_list = []
        bag_list = []
        meta_list = []
        path = "data-features/all"
        if self.train:
            file_list = self.train_list
        else:
            file_list = self.test_list

        files = []
        all_files = os.listdir(path)
        for patient in file_list:
            for x in all_files:
                if x.split("_")[0] + "_" + x.split("_")[1] == patient:
                    files.append(x)


        i = 0
        for file in files:
            i -=- 1
            if file.split(".")[1] != "dat": continue
            with gzip.open(os.path.join(path, file), 'rb') as f:
                datapack = pickle.load(f)
                data = datapack['data']

                if self.mode == "smp":
                    features = None
                    for d in data:
                        feats = d['feats']
                        feats = np.rollaxis(feats,3,1)
                        if features is None:
                            features = feats
                        else:
                            features = np.append(features, feats, axis=0)

                    if features is not None:
                        label_list.append(datapack["meta"]["label"])
                        bag_list.append(features)
                        meta_list.append([datapack["meta"],])
                        print(str(i)+"/"+str(len(files)) + "\t" + file)


                elif self.mode == "img":
                    for d in data:
                        feats = d['feats']
                        feats = np.rollaxis(feats, 3, 1)
                        if feats.shape[0] == 0:
                            continue
                        if feats is not None:
                            label_list.append(datapack["meta"]["label"])
                            bag_list.append(feats)

                    print(str(i) + "/" + str(len(files)) + "\t" + file)

        return [label_list, bag_list, meta_list]


    def __len__(self):
        if self.train:
            return len(self.train_labels_list)
        else:
            return len(self.test_labels_list)


    def __getitem__(self, index):
        if self.imgmode == "bag":
            if self.train:
                bag = self.train_bags_list[index]
                label = np.zeros(len(self.classes))
                label[self.train_labels_list[index]] = 1
            else:
                bag = self.test_bags_list[index]
                label = np.zeros(len(self.classes))
                label[self.test_labels_list[index]] = 1

            return bag, label
        elif self.imgmode == "img":
            if self.train:
                bag = self.train_img_list[index]
                label = np.zeros(len(self.classes))
                label[self.train_labels_list[index]] = 1
            else:
                bag = self.test_img_list[index]
                label = np.zeros(len(self.classes))
                label[self.test_labels_list[index]] = 1

            imagebag = []
            for im in bag:
                if im[-3:] == 'zvi':
                    image = zviloader(im)
                elif im[-3:] == 'png':
                    image = cv2.imread(im)
                else:
                    continue

                if len(image.shape) < 3:
                    continue


                if image.shape[0] != 572 or image.shape[1] != 572:
                    image = cv2.resize(image, (572, 572))

                image = np.rollaxis(image, 2, 0)


                imagebag.append(image)

            return np.array(imagebag), label



    def _load_classes(self):
        # classes = np.unique(self.train_labels_list)
        self.classes = []
        with open("data-features/classes.txt") as f:
            clsdata = f.readlines()
            for cls in clsdata:
                self.classes.append(cls.strip("\n"))



    def _load_folds(self, foldnumber):
        self.train_list = []
        self.test_list = []

        with open("folds.pkl", "rb") as f:
            folds = pickle.load(f)

        self.train_list, self.test_list = folds[foldnumber]



    def _get_classes(self):
        return self.classes

    def _correctlabellist(self, label_list):
        newlist = []
        for label in label_list:
            newlist.append(self.classes.index(label))

        return newlist








if __name__ == "__main__":
    d = DataLoader(2,"img", "bag", train=False)

    for data, label in d:
        print( data.shape)

    print("hi")