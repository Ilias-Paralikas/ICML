"""Gather every ssl_baselines/*/*/*/test_result.json into one table.

  python ssl_baselines/collate_results.py            # print a table
  python ssl_baselines/collate_results.py --json out.json
"""
import argparse
import glob
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='cache/model_weights/ssl_baselines')
    ap.add_argument('--json', default=None, help='also dump the collected rows here')
    args = ap.parse_args()

    rows = []
    # recursive: handles both the old flat <ds>/<method>/<N>/ and the new
    # <ds>/<method>/lf<f>_seed<s>/<N>/ layout
    for p in sorted(glob.glob(os.path.join(args.root, '**', 'test_result.json'), recursive=True)):
        with open(p) as f:
            r = json.load(f)
        r['run_dir'] = os.path.dirname(p)
        r['_mtime'] = os.path.getmtime(p)
        rows.append(r)

    if not rows:
        print(f'no test_result.json under {args.root}')
        return

    # one row per (dataset, method, labeled_fraction, seed): prefer a run scored with
    # the all-classes selection metric, then the most recent.
    best = {}
    for r in rows:
        k = (r['dataset'], r['method'], r['labeled_fraction'], r['seed'])
        cur = best.get(k)
        key_r = (r.get('selection_metric') == 'dice_mean_all_classes', r['_mtime'])
        key_cur = cur and (cur.get('selection_metric') == 'dice_mean_all_classes', cur['_mtime'])
        if cur is None or key_r > key_cur:
            best[k] = r
    rows = list(best.values())

    hdr = f"{'dataset':<12}{'method':<10}{'lab.f':>7}{'seed':>5}" \
          f"{'Dice(all)':>11}{'IoU(all)':>10}{'Dice(fg)':>10}{'IoU(fg)':>9}"
    print(hdr)
    print('-' * len(hdr))
    for r in sorted(rows, key=lambda x: (x['dataset'], x['method'], -x['labeled_fraction'])):
        # fall back to per-class for runs written before the all-mean fields existed
        dice_all = r.get('test_dice_all_mean',
                         sum(r['test_dice_per_class']) / len(r['test_dice_per_class']))
        iou_all = r.get('test_iou_all_mean',
                        sum(r['test_iou_per_class']) / len(r['test_iou_per_class']))
        print(f"{r['dataset']:<12}{r['method']:<10}{r['labeled_fraction']:>7}{r['seed']:>5}"
              f"{dice_all:>11.4f}{iou_all:>10.4f}"
              f"{r['test_dice_fg_mean']:>10.4f}{r['test_iou_fg_mean']:>9.4f}")

    if args.json:
        with open(args.json, 'w') as f:
            json.dump(rows, f, indent=2)
        print(f'\nwrote {args.json}')


if __name__ == '__main__':
    main()
