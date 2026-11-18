"""Run all strategies and generate offline, self-contained reports."""
import argparse
import json
import time
from pathlib import Path
import chunkers
import ingest
from retriever import Retriever
from score import score_question, summarize

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'out'
WEIGHTS = [(1.0, 0.0), (0.5, 0.5), (0.0, 1.0)]


def make_report(results):
    lines = ['# RAG 分块与检索评测', '', '评分以固定黄金摘录规则计算，数值为真实运行结果。', '',
             '| 策略 | 向量权重 | BM25 权重 | 块数 | 平均块长（字符） | 总分 | 满分率 | 命中率 | 2 分占比 | 1 分占比 | 0 分占比 |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for run in results['runs']:
        s = run['summary']; n = len(run['questions'])
        p = lambda value: f'{100 * value:.1f}%'
        lines.append(f"| {run['strategy']} | {run['vector_weight']:.1f} | {run['bm25_weight']:.1f} | {run['chunk_count']} | {run['avg_chars']:.1f} | {s['total_score']}/{s['max_score']} | {p(s['full_rate'])} | {p(s['hit_rate'])} | {p(s['score_counts']['2']/n)} | {p(s['score_counts']['1']/n)} | {p(s['score_counts']['0']/n)} |")
    lines += ['', '## 分组明细', '', '| 策略 | 向量权重 | 分组 | 组别 | 题数 | 总分 | 命中率 |', '|---|---:|---|---|---:|---:|---:|']
    for run in results['runs']:
        for group, values in run['summary']['groups'].items():
            for name, value in sorted(values.items()):
                lines.append(f"| {run['strategy']} | {run['vector_weight']:.1f} | {group} | {name} | {value['questions']} | {value['total_score']}/{2 * value['questions']} | {value['hit_rate']:.1%} |")
    fixed = next(r for r in results['runs'] if r['strategy'] == 'fixed')
    recursive = next(r for r in results['runs'] if r['strategy'] == 'recursive')
    if recursive['summary']['total_score'] < fixed['summary']['total_score']:
        lines += ['', f"在向量权重 {fixed['vector_weight']:.1f}、BM25 权重 {fixed['bm25_weight']:.1f} 的配置下，recursive 的总分低于 fixed；报告保留实际结果。"]
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedder', choices=['mock', 'openai-compatible'], default='mock')
    parser.add_argument('--rerank', choices=['none', 'cross-encoder'], default='none')
    parser.add_argument('--top-k', type=int, default=5)
    parser.add_argument('--vector-weight', type=float)
    parser.add_argument('--bm25-weight', type=float)
    parser.add_argument('--jieba', action='store_true')
    args = parser.parse_args()
    if args.top_k < 1 or ((args.vector_weight is None) != (args.bm25_weight is None)):
        parser.error('top-k must be positive; provide both weights together')
    weights = WEIGHTS if args.vector_weight is None else [(args.vector_weight, args.bm25_weight)]
    if any(v < 0 or b < 0 or v + b == 0 for v, b in weights):
        parser.error('weights must be nonnegative and not both zero')
    started = time.perf_counter()
    docs = ingest.main()
    with (ROOT / '../data/golden.jsonl').open(encoding='utf-8') as f:
        gold = [json.loads(line) for line in f]
    with (ROOT / '../data/corpus.jsonl').open(encoding='utf-8') as f:
        categories = {x['path']: x['cat'] for x in map(json.loads, f)}
    runs, browser_chunks = [], {}
    for strategy in ('fixed', 'semantic', 'recursive'):
        chunks = chunkers.chunk(docs, strategy=strategy)
        retriever = Retriever(chunks, OUT / (strategy + '.sqlite'), args.embedder, args.jieba)
        browser_chunks[strategy] = [c.as_dict() for c in chunks]
        per_weight = [[] for _ in weights]
        for item in gold:
            scores = retriever.scores(item['question'])
            for idx, (vw, bw) in enumerate(weights):
                found = retriever.search(item['question'], args.top_k, vw, bw, args.rerank, scores)
                per_weight[idx].append({'qid': item['qid'], 'question': item['question'],
                    'cat': item.get('cat', categories.get(item['gold_doc'], 'unknown')),
                    'has_code_block': item['has_code_block'], 'score': score_question(item, found['hits']), **found})
        for (vw, bw), rows in zip(weights, per_weight):
            runs.append({'strategy': strategy, 'vector_weight': vw, 'bm25_weight': bw,
                         'chunk_count': len(chunks), 'avg_chars': sum(len(c.text) for c in chunks)/len(chunks),
                         'summary': summarize(rows), 'questions': rows})
        print(f'{strategy}: {len(chunks)} chunks; scores ' + ', '.join(f'{r[0]:g}/{r[1]:g}={sum(x["score"] for x in rows)}' for r, rows in zip(weights, per_weight)), flush=True)
    results = {'embedder': args.embedder, 'tokenizer': 'bigram+jieba requested' if args.jieba else 'bigram',
               'top_k': args.top_k, 'runs': runs}
    (OUT / 'results.json').write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    (OUT / 'report.md').write_text(make_report(results), encoding='utf-8')
    payload = json.dumps(browser_chunks, ensure_ascii=False, separators=(',', ':')).replace('<', '\\u003c')
    template = (ROOT / 'report_template.html').read_text(encoding='utf-8')
    (OUT / 'report.html').write_text(template.replace('/*DATA*/', payload), encoding='utf-8')
    print(f'Wrote out/results.json, out/report.md, out/report.html in {time.perf_counter()-started:.2f}s', flush=True)


if __name__ == '__main__':
    main()
