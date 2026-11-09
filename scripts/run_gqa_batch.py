"""VisProg 在 GQA 上批量推理并统计准确率；支持断点续跑。"""
import argparse
import collections
import contextlib
import io
import json
import os
import sys
import time
from functools import partial
from html import escape

# 保证能从仓库根目录导入 engine / prompts / vis_utils
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from PIL import Image

def normalize(x):
    """把程序输出规整为 GQA 的小写字符串答案；非文本结果视为空答案。"""
    if isinstance(x, bool):
        return 'yes' if x else 'no'
    if isinstance(x, (str, int, float)):
        return str(x).strip().lower().rstrip('.')
    return ''


def run_program(interpreter, prog, image, inspect=False):
    """执行程序，返回 (结果, html)；inspect=False 时 html 为空串。"""
    # 解释器每步都会 print，静音避免刷屏
    with contextlib.redirect_stdout(io.StringIO()):
        if inspect:
            result, _, html_str = interpreter.execute(prog, dict(IMAGE=image), inspect=True)
            return result, html_str
        result, _ = interpreter.execute(prog, dict(IMAGE=image), inspect=False)
    return result, ''


def save_html(path, rec, steps_html):
    """在每步可视化前加上题目、标准答案、预测和程序，便于对照查看。"""
    head = (
        f"<h3>{escape(rec['question'])}</h3>"
        f"<p>qid={rec['qid']} 标准答案: <b>{escape(rec['answer'])}</b> "
        f"预测: <b>{escape(rec['pred'])}</b> {'正确' if rec['correct'] else '错误'}</p>"
        f"<pre>{escape(rec['program'])}</pre>")
    with open(path, 'w', encoding='utf-8') as f:
        f.write(f'<meta charset="utf-8">{head}{steps_html}')


def load_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding='utf-8') as f:
        return [json.loads(ln) for ln in f if ln.strip()]


def summarize(results, out_dir):
    n = len(results)
    if n == 0:
        print('没有结果可统计')
        return
    acc = sum(r['correct'] for r in results) / n
    failed = sum(1 for r in results if r['error'])

    def group_acc(key):
        g = collections.defaultdict(list)
        for r in results:
            g[r[key]].append(r['correct'])
        return {k: dict(n=len(v), acc=round(sum(v) / len(v), 4)) for k, v in sorted(g.items())}

    summary = dict(
        num=n, accuracy=round(acc, 4), exec_failed=failed,
        by_structural=group_acc('structural'), by_semantic=group_acc('semantic'))
    with open(os.path.join(out_dir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f'\n总题数: {n}  准确率: {acc:.2%}  程序执行失败: {failed}')
    for key in ('by_structural', 'by_semantic'):
        print(f'-- {key}')
        for k, v in summary[key].items():
            print(f"  {k:<10} n={v['n']:<5} acc={v['acc']:.2%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/gqa', help='download_gqa.py 的输出目录')
    ap.add_argument('--out', default='outputs/gqa', help='结果根目录')
    ap.add_argument('--run-name', default='run0', help='本次实验名，结果存到 out/run-name')
    ap.add_argument('--limit', type=int, default=0, help='只跑前 N 题；0 表示全部')
    ap.add_argument('--model', default=None, help='LLM 模型名；默认读 OPENAI_MODEL，否则 gpt-3.5-turbo')
    ap.add_argument('--temperature', type=float, default=0.7)
    ap.add_argument('--prompt-method', choices=['all', 'random'], default='all')
    ap.add_argument('--num-prompts', type=int, default=8, help='prompt-method=random 时的示例数')
    ap.add_argument('--max-retries', type=int, default=3, help='程序生成/执行失败时的重试次数')
    ap.add_argument('--save-html', choices=['all', 'wrong', 'none'], default='all',
                    help='保存每题的执行可视化 html：全部 / 仅答错 / 不保存')
    ap.add_argument('--eval-only', action='store_true', help='不推理，只重新统计已有结果')
    args = ap.parse_args()

    out_dir = os.path.join(args.out, args.run_name)
    os.makedirs(out_dir, exist_ok=True)
    res_path = os.path.join(out_dir, 'results.jsonl')

    if args.eval_only:
        summarize(load_jsonl(res_path), out_dir)
        return

    questions = load_jsonl(os.path.join(args.data, 'questions.jsonl'))
    if args.limit > 0:
        questions = questions[:args.limit]
    done = {r['qid'] for r in load_jsonl(res_path)}
    todo = [q for q in questions if q['qid'] not in done]
    print(f'共 {len(questions)} 题，已完成 {len(done)}，待跑 {len(todo)}')

    # 放在参数解析之后再导入，--eval-only 时不必加载模型依赖
    from engine.utils import ProgramInterpreter, ProgramGenerator
    from prompts.gqa import create_prompt

    html_dir = os.path.join(out_dir, 'html')
    if args.save_html != 'none':
        os.makedirs(html_dir, exist_ok=True)

    interpreter = ProgramInterpreter(dataset='gqa')
    prompter = partial(create_prompt, method=args.prompt_method, num_prompts=args.num_prompts)
    generator = ProgramGenerator(prompter, temperature=args.temperature, model=args.model)

    with open(res_path, 'a', encoding='utf-8') as fout:
        for i, q in enumerate(todo, 1):
            image = Image.open(os.path.join(args.data, 'images', f"{q['image_id']}.jpg")).convert('RGB')
            image.thumbnail((640, 640), Image.Resampling.LANCZOS)

            prog, pred, err, steps_html = '', '', '', ''
            for attempt in range(1, args.max_retries + 1):
                try:
                    prog, _ = generator.generate(dict(question=q['question']))
                    result, steps_html = run_program(
                        interpreter, prog, image, inspect=args.save_html != 'none')
                    pred = normalize(result)
                    err = ''
                    break
                except Exception as e:
                    err = f'{type(e).__name__}: {e}'
                    pred, steps_html = '', ''
                    if attempt < args.max_retries:
                        time.sleep(1)

            rec = dict(q, pred=pred, correct=int(pred == q['answer']),
                       program=prog, error=err)
            # 执行失败的题没有可视化内容，不保存
            if steps_html and (args.save_html == 'all' or not rec['correct']):
                save_html(os.path.join(html_dir, f"{q['qid']}.html"), rec, steps_html)
            fout.write(json.dumps(rec, ensure_ascii=False) + '\n')
            fout.flush()
            if i % 10 == 0 or i == len(todo):
                print(f'[{i}/{len(todo)}] last: {q["question"]} -> {pred!r} (gt {q["answer"]!r})')

    summarize(load_jsonl(res_path), out_dir)


if __name__ == '__main__':
    main()
