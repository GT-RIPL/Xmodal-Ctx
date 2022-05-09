import argparse
from pathlib import Path
import h5py
import shutil

import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T
from pytorch_lightning import Trainer, LightningModule, seed_everything
from transformers import CLIPModel, CLIPProcessor

import sys
sys.path.append('.')
from dataset import CocoImageCrops, collate_crops


class ImageEncoder(LightningModule):
    def __init__(self, save_dir):
        super().__init__()

        self.save_dir = Path(save_dir)
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").vision_model
    
    def test_step(self, batch, batch_idx):
        orig_imgs, _, _, _, ids = batch

        features = self.model(pixel_values=orig_imgs)
        features = features.pooler_output
        features = features.detach().cpu().numpy()

        with h5py.File(self.save_dir/"vis_ctx.hdf5", "a") as f:
            f.attrs["fdim"] = features.shape[-1]
            for i in range(len(orig_imgs)):
                f.create_dataset(str(int(ids[i])), data=features[i])


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

    img_encoder = ImageEncoder(args.save_dir)

    trainer = Trainer(
        gpus=[args.device, ],
        deterministic=True,
        benchmark=False,
        default_root_dir=args.save_dir
    )
    trainer.test(img_encoder, dloader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Encode images')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--exp_name', type=str, default='image_features')
    parser.add_argument('--dataset_root', type=str, default='datasets/coco_captions')
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
