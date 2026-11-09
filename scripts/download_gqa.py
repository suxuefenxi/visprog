"""下载 GQA testdev-balanced（12578 题 / 398 图），抽样后保存为 questions.jsonl 与 images/。"""
import argparse
import json
import os
import random


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='data/gqa', help='输出目录')
    ap.add_argument('--num', type=int, default=500, help='抽样题数；<=0 表示全量')
    ap.add_argument('--seed', type=int, default=0, help='抽样随机种子')
    ap.add_argument('--hf-endpoint', default=None, help='HF 镜像地址，如 https://hf-mirror.com')
    args = ap.parse_args()

    # 必须在导入 datasets 之前设置镜像
    if args.hf_endpoint:
        os.environ['HF_ENDPOINT'] = args.hf_endpoint
    from datasets import load_dataset

    repo = 'lmms-lab/GQA'
    qs = load_dataset(repo, 'testdev_balanced_instructions', split='testdev')
    qs = qs.select_columns(['id', 'imageId', 'question', 'answer', 'types'])
    print(f'testdev_balanced 总题数: {len(qs)}')

    indices = list(range(len(qs)))
    if 0 < args.num < len(qs):
        indices = sorted(random.Random(args.seed).sample(indices, args.num))

    records = []
    for i in indices:
        row = qs[i]
        records.append(dict(
            qid=row['id'],
            image_id=row['imageId'],
            question=row['question'],
            answer=row['answer'].strip().lower(),
            structural=row['types']['structural'],
            semantic=row['types']['semantic'],
        ))

    img_dir = os.path.join(args.out, 'images')
    os.makedirs(img_dir, exist_ok=True)
    need = {r['image_id'] for r in records}

    imgs = load_dataset(repo, 'testdev_balanced_images', split='testdev')
    # 只读 id 列，避免解码全部图片
    id2idx = {img_id: k for k, img_id in enumerate(imgs['id'])}
    missing = need - set(id2idx)
    if missing:
        raise RuntimeError(f'有 {len(missing)} 张图片在图片集中找不到，例如 {sorted(missing)[:3]}')

    for img_id in sorted(need):
        path = os.path.join(img_dir, f'{img_id}.jpg')
        if os.path.exists(path):
            continue
        imgs[id2idx[img_id]]['image'].convert('RGB').save(path, quality=95)

    q_path = os.path.join(args.out, 'questions.jsonl')
    with open(q_path, 'w', encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    print(f'已保存 {len(records)} 题、{len(need)} 张图 -> {args.out}')


if __name__ == '__main__':
    main()
