import argparse
from pathlib import Path
import h5py
import numpy as np
import math
import faiss
from tqdm import tqdm
import shutil

import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T
from pytorch_lightning import Trainer, LightningModule, seed_everything
from transformers import CLIPModel, CLIPProcessor

import sys
sys.path.append('.')
from dataset import CocoImageCrops, collate_crops


class CaptionRetriever(LightningModule):
    def __init__(self, caption_db, save_dir, k):
        super().__init__()

        self.save_dir = Path(save_dir)
        self.k = k

        self.keys, self.features, self.text = self.load_caption_db(caption_db)
        self.index = self.build_index(idx_file=self.save_dir/"faiss.index")
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    
    @staticmethod
    def load_caption_db(caption_db):
        print("Loading caption db")
        keys, features, text = [], [], []
        with h5py.File(caption_db, "r") as f:
            for i in tqdm(range(len(f))):
                keys_i = f[f"{i}/keys"][:]
                features_i = f[f"{i}/features"][:]
                text_i = [str(x, "utf-8") for x in f[f"{i}/captions"][:]]

                keys.append(keys_i)
                features.append(features_i)
                text.extend(text_i)
        keys = np.concatenate(keys)
        features = np.concatenate(features)

        return keys, features, text
    
    def build_index(self, idx_file):
        print("Building db index")
        n, d = self.keys.shape
        K = round(8 * math.sqrt(n))
        index = faiss.index_factory(d, f"IVF{K},Flat", faiss.METRIC_INNER_PRODUCT)

        assert not index.is_trained
        index.train(self.keys)
        assert index.is_trained
        index.add(self.keys)
        index.nprobe = max(1, K//10)

        faiss.write_index(index, str(idx_file))

        return index

    def search(self, images, topk):
        features = self.clip.vision_model(pixel_values=images)[1]
        query = self.clip.visual_projection(features)
        query = query / query.norm(dim=-1, keepdim=True)
        D, I = self.index.search(query.detach().cpu().numpy(), topk)

        return D, I
    
    def test_step(self, batch, batch_idx):
        orig_imgs, five_imgs, nine_imgs, gt_caps, ids = batch
        N = len(orig_imgs)
        
        with h5py.File(self.save_dir/"txt_ctx.hdf5", "a") as f:
            D_o, I_o = self.search(orig_imgs, topk=self.k)  # N x self.k
            
            D_f, I_f = self.search(torch.flatten(five_imgs, end_dim=1), topk=self.k)  # N*5 x self.k
            D_f, I_f = D_f.reshape(N, 5, self.k), I_f.reshape(N, 5, self.k)
            
            D_n, I_n = self.search(torch.flatten(nine_imgs, end_dim=1), topk=self.k)  # N*9 x self.k
            D_n, I_n = D_n.reshape(N, 9, self.k), I_n.reshape(N, 9, self.k)

            for i in range(N):
                g1 = f.create_group(str(int(ids[i])))

                texts = [self.text[j] for j in I_o[i]]
                features = self.features[I_o[i]]
                scores = D_o[i]
                g2 = g1.create_group("whole")
                g2.create_dataset("features", data=features)
                g2.create_dataset("scores", data=scores)
                g2.create_dataset("texts", data=texts)

                texts = [
                    [
                        self.text[I_f[i, j, k]]
                        for k in range(self.k)
                    ]
                    for j in range(5)
                ]
                features = self.features[I_f[i].flatten()].reshape((5, self.k, -1))
                scores = D_f[i]
                g3 = g1.create_group("five")
                g3.create_dataset("features", data=features)
                g3.create_dataset("scores", data=scores)
                g3.create_dataset("texts", data=texts)

                texts = [
                    [
                        self.text[I_n[i, j, k]]
                        for k in range(self.k)
                    ]
                    for j in range(9)
                ]
                features = self.features[I_n[i].flatten()].reshape((9, self.k, -1))
                scores = D_n[i]
                g4 = g1.create_group("nine")
                g4.create_dataset("features", data=features)
                g4.create_dataset("scores", data=scores)
                g4.create_dataset("texts", data=texts)



def build_ctx_caps(args):
    transform = T.Compose([
        CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32").feature_extractor,
        lambda x: torch.FloatTensor(x["pixel_values"][0]),
    ])
    dset = CocoImageCrops(args.dataset_root/"annotations", args.dataset_root, transform)
    dloader = DataLoader(
        dataset=dset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        collate_fn=collate_crops
    )

    cap_retr = CaptionRetriever(
        caption_db=args.caption_db,
        save_dir=args.save_dir,
        k=args.k
    )

    trainer = Trainer(
        gpus=[args.device, ],
        deterministic=True,
        benchmark=False,
        default_root_dir=args.save_dir
    )
    trainer.test(cap_retr, dloader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Retrieve captions')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--exp_name', type=str, default='retrieved_captions')
    parser.add_argument('--dataset_root', type=str, default='datasets/coco_captions')
    parser.add_argument('--caption_db', type=str, default='outputs/captions_db/caption_db.hdf5')
    parser.add_argument('--k', type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=12)
    args = parser.parse_args()
    
    args.dataset_root = Path(args.dataset_root)
    setattr(args, "save_dir", Path("outputs")/args.exp_name)
    shutil.rmtree(args.save_dir, ignore_errors=True)
    args.save_dir.mkdir(parents=True, exist_ok=True)
    print(args)

    seed_everything(1, workers=True)

    build_ctx_caps(args)
