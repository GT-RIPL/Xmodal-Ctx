import random
from data import (
    ImageDetectionsField, TextField, TxtCtxField, VisCtxField, RawField
)
from data import COCO, DataLoader
import evaluation
from evaluation import PTBTokenizer, Cider
from models.transformer import (
    Transformer, MemoryAugmentedEncoder, MeshedDecoder, ScaledDotProductAttentionMemory,
    Projector
)
from transformers.optimization import (
    get_constant_schedule_with_warmup, AdamW
)
import torch
from torch import nn
from torch.nn import NLLLoss
from torch.utils.tensorboard import SummaryWriter
import argparse, os, pickle
from tqdm import tqdm
import numpy as np
import itertools
import multiprocessing
from shutil import copyfile
from pathlib import Path
import itertools
import shutil
import json
import math


random.seed(1234)
torch.manual_seed(1234)
np.random.seed(1234)


def evaluate_loss(model, dataloader, loss_fn, text_field):
    # Validation loss
    model.eval()

    running_loss = 0.0
    with tqdm(desc='Epoch %d - validation' % e, unit='it', total=len(dataloader), dynamic_ncols=True) as pbar:
        with torch.no_grad():
            for it, data in enumerate(dataloader):
                txt_ctx = {
                    k1: {
                        k2: v2.to(device, non_blocking=True)
                        for k2, v2 in v1.items()
                    }
                    for k1, v1 in data["txt_ctx"].items()
                }
                vis_ctx = data["vis_ctx"].to(device, non_blocking=True)
                obj = data["object"].to(device, non_blocking=True)
                captions = data["text"].to(device)

                out = model(obj=obj, vis_ctx=vis_ctx, txt_ctx=txt_ctx, seq=captions, mode="xe")
                out = out[:, :-1].contiguous()
                captions_gt = captions[:, 1:].contiguous()
                loss = loss_fn(out.view(-1, len(text_field.vocab)), captions_gt.view(-1))
                
                running_loss += loss.item()
                pbar.set_postfix(loss=running_loss / (it + 1))
                pbar.update()

    loss = running_loss / len(dataloader)
    ret = {"loss": loss}

    return ret


def evaluate_metrics(model, dataloader, text_field):
    model.eval()

    gen, gts = {}, {}
    with tqdm(desc='Epoch %d - evaluation' % e, unit='it', total=len(dataloader), dynamic_ncols=True) as pbar:
        with torch.no_grad():
            for it, data in enumerate(dataloader):
                txt_ctx = {
                    k1: {
                        k2: v2.to(device, non_blocking=True)
                        for k2, v2 in v1.items()
                    }
                    for k1, v1 in data["txt_ctx"].items()
                }
                vis_ctx = data["vis_ctx"].to(device, non_blocking=True)
                obj = data["object"].to(device, non_blocking=True)

                out, _ = model(
                    obj=obj, vis_ctx=vis_ctx, txt_ctx=txt_ctx, max_len=20, mode="rl",
                    eos_idx=text_field.vocab.stoi['<eos>'], beam_size=5, out_size=1,
                )

                caps_gen = text_field.decode(out, join_words=False)
                for i, (gts_i, gen_i) in enumerate(zip(data["text"], caps_gen)):
                    gen_i = ' '.join([k for k, g in itertools.groupby(gen_i)])
                    gen['%d_%d' % (it, i)] = [gen_i, ]
                    gts['%d_%d' % (it, i)] = gts_i
                pbar.update()

    gts = evaluation.PTBTokenizer.tokenize(gts)
    gen = evaluation.PTBTokenizer.tokenize(gen)
    scores, _ = evaluation.compute_scores(gts, gen)

    return scores


def train_xe(model, dataloader, optim, text_field):
    # Training with cross-entropy
    model.train()
    running_loss = 0.0
    with tqdm(desc='Epoch %d - train' % e, unit='it', total=len(dataloader), dynamic_ncols=True) as pbar:
        for it, data in enumerate(dataloader):
            txt_ctx = {
                k1: {
                    k2: v2.to(device, non_blocking=True)
                    for k2, v2 in v1.items()
                }
                for k1, v1 in data["txt_ctx"].items()
            }
            vis_ctx = data["vis_ctx"].to(device, non_blocking=True)
            obj = data["object"].to(device, non_blocking=True)
            captions = data["text"].to(device, non_blocking=True)

            out = model(obj=obj, vis_ctx=vis_ctx, txt_ctx=txt_ctx, seq=captions, mode="xe")
            out = out[:, :-1].contiguous()
            captions_gt = captions[:, 1:].contiguous()
            loss = loss_fn(out.view(-1, len(text_field.vocab)), captions_gt.view(-1))

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()

            running_loss += loss.item()
            pbar.set_postfix(loss=running_loss / (it + 1))
            pbar.update()
            scheduler.step()

    loss = running_loss / len(dataloader)
    ret = {"loss": loss}
    
    return ret


def train_scst(model, dataloader, optim, cider, text_field):
    # Training with self-critical
    model.train()

    tokenizer_pool = multiprocessing.Pool()
    running_reward = .0
    running_reward_baseline = .0
    running_loss = .0
    seq_len = 20
    beam_size = 5
    with tqdm(desc='Epoch %d - train' % e, unit='it', total=len(dataloader), dynamic_ncols=True) as pbar:
        for it, data in enumerate(dataloader):
            txt_ctx = {
                k1: {
                    k2: v2.to(device, non_blocking=True)
                    for k2, v2 in v1.items()
                }
                for k1, v1 in data["txt_ctx"].items()
            }
            vis_ctx = data["vis_ctx"].to(device, non_blocking=True)
            obj = data["object"].to(device, non_blocking=True)

            out, log_prob = model(
                obj=obj, vis_ctx=vis_ctx, txt_ctx=txt_ctx, max_len=seq_len, mode="rl",
                eos_idx=text_field.vocab.stoi['<eos>'], beam_size=beam_size, out_size=beam_size,
            )
            
            # Rewards
            caps_gen = text_field.decode(out.view(-1, seq_len))
            caps_gt = list(itertools.chain(*([c, ] * beam_size for c in data["text"])))
            caps_gen, caps_gt = tokenizer_pool.map(evaluation.PTBTokenizer.tokenize, [caps_gen, caps_gt])
            reward = cider.compute_score(caps_gt, caps_gen)[1].astype(np.float32)
            reward = torch.from_numpy(reward).to(device).view(obj.shape[0], beam_size)
            reward_baseline = torch.mean(reward, -1, keepdim=True)
            loss = -torch.mean(log_prob, -1) * (reward - reward_baseline)
            loss = loss.mean()

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()

            running_loss += loss.item()
            running_reward += reward.mean().item()
            running_reward_baseline += reward_baseline.mean().item()
            pbar.set_postfix(loss=running_loss / (it + 1), reward=running_reward / (it + 1),
                             reward_baseline=running_reward_baseline / (it + 1))
            pbar.update()
    
    tokenizer_pool.close()
    loss = running_loss / len(dataloader)
    reward = running_reward / len(dataloader)
    reward_baseline = running_reward_baseline / len(dataloader)
    ret = {
        "loss": loss,
        "reward": reward,
        "reward_baseline": reward_baseline
    }

    return ret


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Meshed-Memory Transformer')
    parser.add_argument('--exp_name', type=str, default='[m2][xmodal-ctx]')
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--bs_reduct', type=int, default=5)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--m', type=int, default=40)
    parser.add_argument('--topk', type=int, default=12)
    parser.add_argument('--warmup', type=int, default=10000)
    parser.add_argument('--lr_xe', type=float, default=1e-4)
    parser.add_argument('--lr_rl', type=float, default=5e-6)
    parser.add_argument('--wd_rl', type=float, default=0.05)
    parser.add_argument('--drop_rate', type=float, default=0.3)
    parser.add_argument('--devices', nargs='+', type=int, default=[0])
    parser.add_argument('--dataset_root', type=str, default="./datasets")
    parser.add_argument('--obj_file', type=str, default="oscar.hdf5")
    parser.add_argument('--preload', action='store_true')
    parser.add_argument('--resume_last', action='store_true')
    parser.add_argument('--resume_best', action='store_true')
    args = parser.parse_args()
    
    args.dataset_root = Path(args.dataset_root)
    setattr(args, "save_dir", Path("outputs")/args.exp_name)
    if not (args.resume_last or args.resume_best):
        shutil.rmtree(args.save_dir, ignore_errors=True)
    args.save_dir.mkdir(parents=True, exist_ok=True)
    
    print(args)
    print('Meshed-Memory Transformer Training')

    device = torch.device(args.devices[0])
    writer = SummaryWriter(log_dir=args.save_dir/"tensorboard")

    # Create the dataset
    object_field = ImageDetectionsField(
        obj_file=args.dataset_root/args.obj_file,
        max_detections=50, preload=args.preload
    )
    text_field = TextField(
        init_token='<bos>', eos_token='<eos>', lower=True, tokenize='spacy',
        remove_punctuation=True, nopoints=False
    )
    txt_ctx_filed = TxtCtxField(
        ctx_file=args.dataset_root/"txt_ctx.hdf5", k=args.topk, preload=args.preload
    )
    vis_ctx_filed = VisCtxField(
        ctx_file=args.dataset_root/"vis_ctx.hdf5", preload=args.preload
    )

    fields = {
        "object": object_field, "text": text_field, "img_id": RawField(),
        "txt_ctx": txt_ctx_filed, "vis_ctx": vis_ctx_filed
    }
    dset = args.dataset_root/"annotations"
    dataset = COCO(fields, dset, dset)
    train_dataset, val_dataset, test_dataset = dataset.splits

    fields = {
        "object": object_field, "text": RawField(), "img_id": RawField(),
        "txt_ctx": txt_ctx_filed, "vis_ctx": vis_ctx_filed
    }
    dict_dataset_train = train_dataset.image_dictionary(fields)
    dict_dataset_val = val_dataset.image_dictionary(fields)
    dict_dataset_test = test_dataset.image_dictionary(fields)
    
    ref_caps_train = list(train_dataset.text)
    cider_train = Cider(PTBTokenizer.tokenize(ref_caps_train))
    
    # build vocabulary
    vocab_file = 'vocab/vocab_coco.pkl'
    if not os.path.isfile(vocab_file):
        print("Building vocabulary")
        text_field.build_vocab(train_dataset, val_dataset, min_freq=5)
        pickle.dump(text_field.vocab, open(vocab_file, 'wb'))
    else:
        text_field.vocab = pickle.load(open(vocab_file, 'rb'))

    # Model and dataloaders
    encoder = MemoryAugmentedEncoder(
        3, 0, attention_module=ScaledDotProductAttentionMemory,
        attention_module_kwargs={'m': args.m}
    )
    decoder = MeshedDecoder(len(text_field.vocab), 54, 3, text_field.vocab.stoi['<pad>'])
    projector = Projector(
        f_obj=2054, f_vis=vis_ctx_filed.fdim, f_txt=512,
        f_out=encoder.d_model, drop_rate=args.drop_rate
    )

    model = Transformer(
        bos_idx=text_field.vocab.stoi['<bos>'],
        encoder=encoder, decoder=decoder, projector=projector
    ).to(device)
    model = nn.DataParallel(model, device_ids=args.devices)

    # optimizer
    no_decay = [
        n for n, m in model.named_modules()
        if any(isinstance(m, nd) for nd in [nn.LayerNorm, ])
    ]
    grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not \
                    any(nd in n for nd in no_decay)], 'weight_decay': 0.05},
        {'params': [p for n, p in model.named_parameters() if \
                    any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optim = AdamW(grouped_parameters, lr=args.lr_xe, eps=1e-8)
    scheduler = get_constant_schedule_with_warmup(optim, num_warmup_steps=args.warmup)

    # Initial conditions
    loss_fn = NLLLoss(ignore_index=text_field.vocab.stoi['<pad>'])
    use_rl = False
    best_cider = .0
    patience = 0
    start_epoch = 0

    # resume training
    if args.resume_last or args.resume_best:
        fname = "ckpt_last.pth" if args.resume_last else "ckpt_best.pth"
        fname = args.save_dir/fname

        if os.path.exists(fname):
            data = torch.load(fname)
            torch.set_rng_state(data['torch_rng_state'])
            torch.cuda.set_rng_state(data['cuda_rng_state'])
            np.random.set_state(data['numpy_rng_state'])
            random.setstate(data['random_rng_state'])
            model.load_state_dict(data['model'], strict=False)
            optim.load_state_dict(data['optimizer'])
            scheduler.load_state_dict(data['scheduler'])
            start_epoch = data['epoch'] + 1
            best_cider = data['best_cider']
            patience = data['patience']
            use_rl = data['use_rl']
            print('Resuming from epoch %d, validation loss %f, and best cider %f' % (
                data['epoch'], data['val_loss'], data['best_cider']))

    print("Training starts")
    for e in range(start_epoch, start_epoch + 100):
        dataloader_train = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=True
        )
        dataloader_val = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, drop_last=False
        )
        dict_dataloader_train = DataLoader(
            dict_dataset_train, batch_size=math.floor(args.batch_size//args.bs_reduct), shuffle=True, num_workers=1, drop_last=True
        )
        dict_dataloader_val = DataLoader(
            dict_dataset_val, batch_size=math.floor(args.batch_size//5), shuffle=False, num_workers=1, drop_last=False
        )
        dict_dataloader_test = DataLoader(
            dict_dataset_test, batch_size=math.floor(args.batch_size//5), shuffle=False, num_workers=1, drop_last=False
        )

        # training epoch
        if not use_rl:
            ret = train_xe(model, dataloader_train, optim, text_field)
            for k, v in ret.items():
                writer.add_scalar(f'data/train_{k}', v, e)
        else:
            ret = train_scst(model, dict_dataloader_train, optim, cider_train, text_field)
            for k, v in ret.items():
                writer.add_scalar(f'data/train_{k}', v, e)

        # Validation loss
        ret = evaluate_loss(model, dataloader_val, loss_fn, text_field)
        for k, v in ret.items():
            writer.add_scalar(f'data/val_{k}', v, e)
        val_loss = ret["loss"]

        # Validation scores
        val_scores = evaluate_metrics(model, dict_dataloader_val, text_field)
        print("Validation scores", val_scores)
        val_cider = val_scores['CIDEr']
        writer.add_scalar('data/val_cider', val_cider, e)
        writer.add_scalar('data/val_bleu1', val_scores['BLEU'][0], e)
        writer.add_scalar('data/val_bleu4', val_scores['BLEU'][3], e)
        writer.add_scalar('data/val_meteor', val_scores['METEOR'], e)
        writer.add_scalar('data/val_rouge', val_scores['ROUGE'], e)
        writer.add_scalar('data/val_spice', val_scores['SPICE'], e)

        # Test scores
        test_scores = evaluate_metrics(model, dict_dataloader_test, text_field)
        print("Test scores", test_scores)
        writer.add_scalar('data/test_cider', test_scores['CIDEr'], e)
        writer.add_scalar('data/test_bleu1', test_scores['BLEU'][0], e)
        writer.add_scalar('data/test_bleu4', test_scores['BLEU'][3], e)
        writer.add_scalar('data/test_meteor', test_scores['METEOR'], e)
        writer.add_scalar('data/test_rouge', test_scores['ROUGE'], e)
        writer.add_scalar('data/test_spice', test_scores['SPICE'], e)

        # Prepare for next epoch
        best = False
        if val_cider >= best_cider:
            best_cider = val_cider
            patience = 0
            best = True
            with open(args.save_dir/"best_val_scores.json", "w") as f:
                json.dump(val_scores, f)
            with open(args.save_dir/"best_test_scores.json", "w") as f:
                json.dump(test_scores, f)
        else:
            patience += 1

        switch_to_rl = False
        exit_train = False
        if patience == 5:
            if not use_rl:
                use_rl = True
                switch_to_rl = True
                patience = 0
                grouped_parameters = [
                    {'params': [p for n, p in model.named_parameters() if not \
                                any(nd in n for nd in no_decay)], 'weight_decay': args.wd_rl},
                    {'params': [p for n, p in model.named_parameters() if \
                        any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
                ]
                optim = AdamW(grouped_parameters, lr=args.lr_rl, eps=1e-8)
                print("Switching to RL")
            else:
                print('patience reached.')
                exit_train = True

        if switch_to_rl and not best:
            data = torch.load(args.save_dir/'ckpt_best.pth')
            torch.set_rng_state(data['torch_rng_state'])
            torch.cuda.set_rng_state(data['cuda_rng_state'])
            np.random.set_state(data['numpy_rng_state'])
            random.setstate(data['random_rng_state'])
            model.load_state_dict(data['model'], strict=False)
            print('Resuming from epoch %d, validation loss %f, and best cider %f' % (
                data['epoch'], data['val_loss'], data['best_cider']))

        torch.save({
            'torch_rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state(),
            'numpy_rng_state': np.random.get_state(),
            'random_rng_state': random.getstate(),
            'epoch': e,
            'val_loss': val_loss,
            'val_cider': val_cider,
            "val_scores": val_scores,
            "test_scores": test_scores,
            'model': model.state_dict(),
            'optimizer': optim.state_dict(),
            'scheduler': scheduler.state_dict(),
            'patience': patience,
            'best_cider': best_cider,
            'use_rl': use_rl,
        }, args.save_dir/'ckpt_last.pth')

        if best:
            copyfile(args.save_dir/'ckpt_last.pth', args.save_dir/'ckpt_best.pth')

        if exit_train:
            writer.close()
            break
