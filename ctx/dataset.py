from PIL import Image
from pathlib import Path
import numpy as np
from pycocotools.coco import COCO as pyCOCO
import json
import itertools
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
from transformers import CLIPProcessor
from torchvision.transforms import functional as F


LENGTH_LIMIT = 75


def collate_tokens(batch):
    captions, input_ids, attention_mask, lengths = [], [], [], []
    for cap, tok in batch:
        assert tok["input_ids"].shape == tok["attention_mask"].shape
        captions.append(cap)

        l = tok["input_ids"].shape[1]
        if l < LENGTH_LIMIT:
            input_ids.append(tok["input_ids"])
            attention_mask.append(tok["attention_mask"])
            lengths.append(l)
        else:
            input_ids.append(tok["input_ids"][:, :LENGTH_LIMIT])
            attention_mask.append(tok["attention_mask"][:, :LENGTH_LIMIT])
            lengths.append(LENGTH_LIMIT)

    max_len = max(lengths)
    input_pad, atten_pad = [], []
    for i in range(len(input_ids)):
        l = input_ids[i].shape[1]
        if l < max_len:
            p = torch.zeros(size=(1, max_len - l), dtype=input_ids[i].dtype)
            input_pad.append(torch.cat([input_ids[i], p], dim=1))
            
            p = torch.zeros(size=(1, max_len - l), dtype=attention_mask[i].dtype)
            atten_pad.append(torch.cat([attention_mask[i], p], dim=1))
        else:
            input_pad.append(input_ids[i])
            atten_pad.append(attention_mask[i])
    
    input_pad = torch.cat(input_pad)
    atten_pad = torch.cat(atten_pad)
    assert input_pad.shape[1] <= LENGTH_LIMIT
    assert atten_pad.shape[1] <= LENGTH_LIMIT
    assert input_pad.shape == atten_pad.shape

    tokens = {"input_ids": input_pad, "attention_mask": atten_pad}

    return captions, tokens


class VisualGenomeCaptions(Dataset):
    def __init__(self, ann_dir):
        super().__init__()
        self.tokenizer = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32").tokenizer
        escapes = ''.join([chr(char) for char in range(0, 32)])
        self.translator = str.maketrans('', '', escapes)

        self.caps = self.parse_annotations(Path(ann_dir))

    @staticmethod
    def combination(l1, l2):
        return [" ".join(x) for x in itertools.product(l1, l2)]
    
    def process_word(self, s):
        return s.lower().strip().translate(self.translator)
    
    def process_synset(self, s):
        return s.lower().strip().translate(self.translator).split(".")[0]
    
    def parse_annotations(self, ann_dir):
        print("loading object attributes...")
        objs = {}
        with open(ann_dir/"attributes.json", "r") as f:
            attributes = json.load(f)
        for x in tqdm(attributes, dynamic_ncols=True):
            for a in x["attributes"]:
                _names = set(self.process_synset(y) for y in a.get("synsets", list()))
                _attrs = set(self.process_word(y) for y in a.get("attributes", list()))

                for n in _names:
                    try:
                        objs[n] |= _attrs
                    except KeyError:
                        objs[n] = _attrs
        del attributes

        print("loading object relationships...")
        rels = set()
        with open(ann_dir/"relationships.json", "r") as f:
            relationships = json.load(f)
        for x in tqdm(relationships, dynamic_ncols=True):
            for r in x["relationships"]:
                _pred = self.process_word(r["predicate"])
                _subj = set(self.process_synset(y) for y in r["subject"]["synsets"])
                _obj = set(self.process_synset(y) for y in r["object"]["synsets"])

                for s in _subj:
                    for o in _obj:
                        rels.add(f"{s}<sep>{_pred}<sep>{o}")
        del relationships

        print("parsing object attributes...")
        caps_obj = []
        for o in tqdm(objs.keys()):
            for a in objs[o]:
                if a != "":
                    caps_obj.append(f"{a} {o}")


        print("parsing object relationships...")
        caps_rel = []
        for r in tqdm(rels):
            s, p, o = r.split("<sep>")
            caps_rel.append(f"{s} {p} {o}")

        caps = np.unique(caps_obj + caps_rel).tolist()

        return caps

    def __len__(self):
        return len(self.caps)

    def __getitem__(self, index):
        tokens = self.tokenizer(self.caps[index], padding=True, return_tensors="pt")
        
        return self.caps[index], tokens


def collate_crops(data):
    orig_image, five_images, nine_images, captions, idx = zip(*data)

    orig_image = torch.stack(list(orig_image), dim=0)
    five_images = torch.stack(list(five_images), dim=0)
    nine_images = torch.stack(list(nine_images), dim=0)
    captions = list(captions)
    idx = torch.LongTensor(list(idx))

    return orig_image, five_images, nine_images, captions, idx


class CocoImageCrops(Dataset):
    def __init__(self, ann_dir, img_root, transform=None):
        self.transform = transform
        self.data = self.parse(Path(ann_dir), Path(img_root))

    @staticmethod
    def parse(ann_dir, img_root):
        ids = (
            np.load(ann_dir/"coco_train_ids.npy"),
            np.concatenate([
                np.load(ann_dir/"coco_restval_ids.npy"),
                np.load(ann_dir/"coco_dev_ids.npy"),
                np.load(ann_dir/"coco_test_ids.npy")
            ]),
        )
        coco = (
            pyCOCO(ann_dir/"captions_train2014.json"),
            pyCOCO(ann_dir/"captions_val2014.json"),
        )
        img_root = (img_root/"train2014", img_root/"val2014")

        data = {}
        for i in range(len(ids)):
            for idx in ids[i]:
                img_id = coco[i].anns[idx]["image_id"]
                img_file = img_root[i]/coco[i].loadImgs(img_id)[0]["file_name"]
                caption = coco[i].anns[idx]["caption"].strip()

                if img_id in data:
                    data[img_id]["captions"].append(caption)
                else:
                    data[img_id] = {
                        "image_id": img_id,
                        "image_file": img_file,
                        "captions": [caption, ]
                    }
        
        data = list(data.values())
        data.sort(key=lambda x: x["image_id"])

        return data
        
    def five_crop(self, image, ratio=0.6):
        w, h = image.size
        hw = (h*ratio, w*ratio)

        return F.five_crop(image, hw)

    def nine_crop(self, image, ratio=0.4):
        w, h = image.size

        t = (0, int((0.5-ratio/2)*h), int((1.0 - ratio)*h))
        b = (int(ratio*h), int((0.5+ratio/2)*h), h)
        l = (0, int((0.5-ratio/2)*w), int((1.0 - ratio)*w))
        r = (int(ratio*w), int((0.5+ratio/2)*w), w)
        h, w = list(zip(t, b)), list(zip(l, r))

        images = []
        for s in itertools.product(h, w):
            h, w = s
            top, left = h[0], w[0]
            height, width = h[1]-h[0], w[1]-w[0]
            images.append(F.crop(image, top, left, height, width))
        
        return images

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = Image.open(self.data[index]["image_file"])
        image = image.convert("RGB")

        five_images = self.five_crop(image)
        nine_images = self.nine_crop(image)

        if self.transform is not None:
            orig_image = self.transform(image)
            five_images = torch.stack([self.transform(x) for x in five_images])
            nine_images = torch.stack([self.transform(x) for x in nine_images])
        
        captions = self.data[index]["captions"]
        idx = self.data[index]["image_id"]

        return orig_image, five_images, nine_images, captions, idx
