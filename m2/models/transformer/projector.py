import torch
from torch import nn
from models.transformer.utils import sinusoid_encoding_table


class Projector(nn.Module):
    def __init__(self, f_obj, f_vis, f_txt, f_out, drop_rate=0.3):
        super().__init__()

        # for objects O
        self.obj_mlp1 = nn.Sequential(
            nn.LayerNorm(f_obj), nn.Linear(f_obj, f_out), nn.Dropout(p=drop_rate)
        )
        self.obj_mlp2 = nn.Sequential(
            nn.LayerNorm(f_vis), nn.Linear(f_vis, f_out), nn.Dropout(p=drop_rate)
        )

        # for txt_ctx
        self.txt_keys = ("whole", "five", "nine")
        for k in self.txt_keys:
            mlp1 = nn.Sequential(
                nn.LayerNorm(f_txt), nn.Linear(f_txt, f_out), nn.Dropout(p=drop_rate)
            )
            mlp2 = nn.Sequential(
                nn.LayerNorm(f_vis), nn.Linear(f_vis, f_out), nn.Dropout(p=drop_rate)
            )
            setattr(self, f"txt_mlp1_{k}", mlp1)
            setattr(self, f"txt_mlp2_{k}", mlp2)

            if k == "whole":
                num_embeddings = 1
            elif k == "five":
                num_embeddings = 5
            elif k == "nine":
                num_embeddings = 9
            else:
                raise KeyError

            pos = nn.Embedding.from_pretrained(
                sinusoid_encoding_table(num_embeddings, f_out), freeze=True
            )
            setattr(self, f"txt_pos_{k}", pos)
        
    def forward(self, obj, vis_ctx, txt_ctx):
        img = vis_ctx[:, None, :]
        embed = []

        # object O
        obj_mask = (torch.sum(torch.abs(obj), dim=-1) == 0)  # N x S
        obj_embed = self.obj_mlp1(obj) + self.obj_mlp2(img)
        obj_embed[obj_mask] = 0.
        embed.append(obj_embed)

        # ctx T
        for k in self.txt_keys:
            pos_k = txt_ctx[k]["pos"]
            embed_k = txt_ctx[k]["embed"]
            mlp1 = getattr(self, f"txt_mlp1_{k}")
            mlp2 = getattr(self, f"txt_mlp2_{k}")
            mlp_pos = getattr(self, f"txt_pos_{k}")
            embed_k = mlp1(embed_k) + mlp2(img) + mlp_pos(pos_k)
            embed.append(embed_k)
        
        return torch.cat(embed, dim=1)
