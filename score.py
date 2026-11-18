"""Fixed 2/1/0 excerpt coverage scoring."""
import itertools
import math
import re
import unicodedata
from collections import Counter, defaultdict
from chunkers import CODE_START, CODE_END


def normalize(text):
    text = re.sub(re.escape(CODE_START) + r'.*?' + re.escape(CODE_END), '[代码块]', text, flags=re.S)
    return ''.join(c.lower() for c in unicodedata.normalize('NFKC', text)
                   if not c.isspace() and not unicodedata.category(c).startswith('P'))


FRAG = 12       # 单次匹配的最长片段
COVER = 0.8     # 覆盖率阈值


def coverage(text, excerpt):
    """按顺序做最长片段匹配，返回摘录被 text 覆盖的比例。

    允许非连续覆盖（跳过的字符按未覆盖计），也允许覆盖跨块拼接：
    在 a+b 拼接文本上顺序匹配，片段可以正好跨过两块的接缝。
    """
    if not text or not excerpt:
        return 0.0
    i, covered = 0, 0
    while i < len(excerpt):
        best = 0
        for size in range(min(FRAG, len(excerpt) - i), 0, -1):
            if excerpt[i:i + size] in text:
                best = size
                break
        if best:
            covered += best
            i += best
        else:
            i += 1
    return covered / len(excerpt)


def covers(text, excerpt):
    return coverage(text, excerpt) >= COVER


def join_with_overlap(a, b):
    for size in range(min(len(a), len(b)), 0, -1):
        if a.endswith(b[:size]):
            return a + b[size:]
    return a + b


def score_question(gold, hits):
    excerpt = normalize(gold['gold_excerpt'])
    heading = normalize(gold['gold_heading'])
    candidates = []
    for hit in hits:
        if hit['path'] != gold['gold_doc']:
            continue
        if heading not in [normalize(re.sub(r'^#+\s*', '', p)) for p in hit['section_path']]:
            continue
        text = hit['text']
        prefix = ' > '.join(hit['section_path']) + '\n'
        if text.startswith(prefix):
            text = text[len(prefix):]
        candidates.append(normalize(text))
    if any(covers(text, excerpt) for text in candidates):
        return 2
    for a, b in itertools.permutations(candidates, 2):
        if covers(a + b, excerpt) or covers(join_with_overlap(a, b), excerpt):
            return 1
    return 0


def summarize(rows):
    groups = {'cat': defaultdict(list), 'has_code_block': defaultdict(list)}
    for row in rows:
        for key in groups:
            groups[key][str(row[key])].append(row['score'])
    counts = Counter(row['score'] for row in rows)
    count = len(rows)
    return {'total_score': sum(row['score'] for row in rows), 'max_score': 2 * count,
            'hit_rate': (counts[1] + counts[2]) / count if count else 0,
            'full_rate': counts[2] / count if count else 0,
            'score_counts': {str(i): counts[i] for i in (2, 1, 0)},
            'groups': {key: {name: {'questions': len(scores), 'total_score': sum(scores),
                                    'hit_rate': sum(s > 0 for s in scores) / len(scores)}
                             for name, scores in values.items()} for key, values in groups.items()}}
