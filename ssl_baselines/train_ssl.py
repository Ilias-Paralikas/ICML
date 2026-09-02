"""Train a semi-supervised segmentation BASELINE (Mean Teacher / UA-MT / BCP) on
any dataset registered in this repo, for comparison against our method.

Runs directly on `dataset_registry.build_dataset()` — same loaders, same
labeled/unlabeled split, same seed as our own training notebook — and scores
through `utils.evaluation.evaluate`, so the Dice / IoU-Jaccard / HD numbers are
computed identically to our method's. Nothing is re-formatted to h5 / nii.gz.

Backbone: a plain 2D U-Net (the standard SSL4MIS/BCP backbone), sized from the
dataset's `in_channels` / `num_classes`.

Example:
  python ssl_baselines/train_ssl.py --dataset drishtigs --method bcp \\
      --labeled_fraction 0.1 --epochs 120 --batch_size 16 --seed 42

Run the same (dataset, labeled_fraction, seed) you use for our method so the
comparison is apples-to-apples. Sweep labeled_fraction (0.01 / 0.05 / 0.1 / 0.2).
"""
import argparse
import copy
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))

from data_handlers.data_loading.dataset_registry import build_dataset  # noqa: E402
from utils.augmentation import GeometricAug, NoiseAug                   # noqa: E402
from utils.evaluation import evaluate                                   # noqa: E402
from utils.file_management import get_version_folder, serialize_config  # noqa: E402
from ssl_baselines.methods import METHODS, default_cfg                  # noqa: E402
from ssl_baselines.unet import UNet, SegWrapper                         # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--dataset', required=True,
                   help="registry name: cardiacUDA|camus|celebamaskhq|isic2017|montgomery|chaos_ct|drishtigs")
    p.add_argument('--method', required=True, choices=list(METHODS))
    p.add_argument('--labeled_fraction', type=float, default=0.1)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--max_iters', type=int, default=10000,
                   help='total training iterations (dataset-size-independent budget)')
    p.add_argument('--eval_every_iters', type=int, default=500)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--eval_batch_size', type=int, default=16)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--base_channels', type=int, default=32)
    p.add_argument('--compute_hd', action='store_true', help='also compute Hausdorff in per-epoch eval (slow)')
    p.add_argument('--eval_ema', action='store_true', help='evaluate the EMA teacher instead of the student')
    p.add_argument('--bcp_pretrain_iters', type=int, default=None,
                   help='override BCP phase-1 (copy-paste supervised pre-train) length')
    p.add_argument('--no_aug', action='store_true', help='disable geometric/noise augmentation')
    p.add_argument('--amp', action='store_true', help='mixed precision')
    p.add_argument('--smoke', action='store_true', help='tiny net + 3 iters/epoch + 2 epochs, for a quick sanity run')
    return p.parse_args()


def set_seed(s):
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.smoke:
        args.base_channels = 8
        args.max_iters = 6
        args.eval_every_iters = 3

    labeled_loader, unlabeled_loader, val_loader, test_loader, num_classes, in_channels = build_dataset(
        args.dataset, batch_size=args.batch_size, labeled_fraction=args.labeled_fraction,
        seed=args.seed, eval_batch_size=args.eval_batch_size,
    )
    print(f'[{args.dataset}] num_classes={num_classes} in_channels={in_channels} | '
          f'labeled batches={len(labeled_loader)} unlabeled={len(unlabeled_loader)} '
          f'val={len(val_loader) if val_loader else None} test={len(test_loader) if test_loader else None}')

    dropout = 0.5 if args.method == 'uamt' else 0.0
    student = UNet(in_channels, num_classes, base=args.base_channels, dropout=dropout).to(device)
    teacher = copy.deepcopy(student).to(device)
    method_cfg = {**default_cfg(args.method), 'eval_ema': args.eval_ema}
    if args.method == 'bcp' and args.bcp_pretrain_iters is not None:
        method_cfg['pretrain_iters'] = args.bcp_pretrain_iters
    method = METHODS[args.method](student, teacher, num_classes, device, method_cfg)
    optimizer = torch.optim.Adam(student.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp)

    geo = None if args.no_aug else GeometricAug(degrees=10, translate=(0.05, 0.05),
                                                scale=(0.9, 1.1), shear=5)
    noise = None if args.no_aug else NoiseAug(noise_std=0.03, brightness=0.1, contrast=0.1)

    # path encodes the run scenario so you can tell runs apart without opening a file:
    #   cache/model_weights/ssl_baselines/<dataset>/<method>/lf<frac>_seed<seed>/<N>/
    root = os.path.join('cache/model_weights/ssl_baselines', args.dataset, args.method,
                        f'lf{args.labeled_fraction:g}_seed{args.seed}')
    run_dir = get_version_folder(root)
    cfg_dump = {**vars(args), 'num_classes': num_classes, 'in_channels': in_channels,
                'method_cfg': method.cfg}
    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump(serialize_config(cfg_dump), f, indent=2)
    print(f'run dir: {run_dir}')

    n_params = sum(p.numel() for p in student.parameters() if p.requires_grad)
    print(f'U-Net trainable params: {n_params:,}')

    global_it = 0
    best_dice = -1.0
    best_path = os.path.join(run_dir, 'best_model.pt')
    labeled_iter = iter(labeled_loader)
    win_loss = win_sup = win_cons = 0.0
    win_n = 0

    def next_labeled():
        nonlocal labeled_iter
        try:
            return next(labeled_iter)
        except StopIteration:
            labeled_iter = iter(labeled_loader)
            return next(labeled_iter)

    def match_bs(x, y, n):
        """Repeat/truncate the labelled batch along dim 0 to exactly n, so it
        pairs 1:1 with the unlabelled batch (BCP's copy-paste mixes them
        element-wise; cycled labelled loaders yield partial last batches)."""
        b = x.shape[0]
        if b == n:
            return x, y
        if b > n:
            return x[:n], y[:n]
        reps = (n + b - 1) // b
        tile = lambda t: t.repeat(reps, *([1] * (t.dim() - 1)))[:n]
        return tile(x), tile(y)

    student.train()
    teacher.train()
    while global_it < args.max_iters:
        for u_batch in unlabeled_loader:
            if global_it >= args.max_iters:
                break
            u_img = u_batch[0].to(device)
            l_img, l_lab, _ = next_labeled()
            l_img, l_lab = l_img.to(device), l_lab.to(device)
            l_img, l_lab = match_bs(l_img, l_lab, u_img.shape[0])

            if geo is not None:
                l_img, l_lab = geo(l_img, l_lab)
                u_img = geo(u_img)
            l_in = noise(l_img) if noise is not None else l_img

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', enabled=args.amp):
                out = method.step(l_in, l_lab, u_img, global_it)
            scaler.scale(out['loss']).backward()
            scaler.step(optimizer)
            scaler.update()
            method.after_step(global_it)

            win_loss += float(out['loss'].item())
            win_sup += out.get('sup', 0.0)
            win_cons += out.get('cons', 0.0)
            win_n += 1
            global_it += 1

            if global_it % args.eval_every_iters == 0 or global_it == args.max_iters:
                print(f'iter {global_it}/{args.max_iters}  loss={win_loss / win_n:.4f}  '
                      f'sup={win_sup / win_n:.4f}  cons={win_cons / win_n:.4f}', flush=True)
                win_loss = win_sup = win_cons = 0.0
                win_n = 0
                torch.save(student.state_dict(), os.path.join(run_dir, 'last_model.pt'))
                if val_loader is not None:
                    eval_net = method.eval_model()
                    cd, ci, ch = evaluate(SegWrapper(eval_net), val_loader, num_classes, device,
                                          desc=f'val@{global_it}', compute_hd=args.compute_hd)
                    eval_net.train()
                    sel_dice = cd.mean().item()   # mean over ALL classes, background included
                    if sel_dice > best_dice:
                        best_dice = sel_dice
                        torch.save(eval_net.state_dict(), best_path)
                        print(f'  new best Dice(all) {best_dice:.4f} '
                              f'[fg {cd[1:].mean().item():.4f}] -> {best_path}', flush=True)

    # ---- final test, once, on the best checkpoint (val) or last (no val) ----
    print('\n===== FINAL TEST =====')
    final_net = method.eval_model()
    if os.path.exists(best_path):
        final_net.load_state_dict(torch.load(best_path, map_location=device))
        print(f'(loaded best-val checkpoint, Dice(all)={best_dice:.4f})')
    if test_loader is not None:
        cd, ci, ch = evaluate(SegWrapper(final_net), test_loader, num_classes, device,
                              desc='test', compute_hd=True)
        result = {'dataset': args.dataset, 'method': args.method,
                  'labeled_fraction': args.labeled_fraction, 'seed': args.seed,
                  'selection_metric': 'dice_mean_all_classes',
                  'test_dice_per_class': cd.tolist(), 'test_iou_per_class': ci.tolist(),
                  'test_hd_per_class': ch.tolist() if ch is not None else None,
                  'test_dice_all_mean': cd.mean().item(),
                  'test_iou_all_mean': ci.mean().item(),
                  'test_dice_fg_mean': cd[1:].mean().item(),
                  'test_iou_fg_mean': ci[1:].mean().item()}
        with open(os.path.join(run_dir, 'test_result.json'), 'w') as f:
            json.dump(result, f, indent=2)
        print(f'\nwrote {os.path.join(run_dir, "test_result.json")}')
        print(f'test  Dice(all)={result["test_dice_all_mean"]:.4f}  IoU(all)={result["test_iou_all_mean"]:.4f}  '
              f'|  Dice(fg)={result["test_dice_fg_mean"]:.4f}  IoU(fg)={result["test_iou_fg_mean"]:.4f}')
    else:
        print('no test split for this dataset.')


if __name__ == '__main__':
    main()
