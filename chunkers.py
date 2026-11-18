"""Deterministic section-aware chunking, with code blocks kept atomic."""
import re
from dataclasses import dataclass, asdict
from itertools import groupby

CODE_START = '⟦CODE⟧'
CODE_END = '⟦/CODE⟧'


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    path: str
    section_path: list
    text: str
    tokens_est: int

    def as_dict(self):
        return asdict(self)


def units(docs):
    for block in docs:
        text = block['text']
        if block['block_type'] == 'code':
            text = CODE_START + text + CODE_END
        yield block, text


def _pack(parts, limit, overlap=0):
    """Pack units; split text if needed but never split a code unit."""
    if limit <= 0 or overlap < 0 or overlap >= limit:
        raise ValueError('invalid chunk size or overlap')
    current = ''
    for part in parts:
        if part.startswith(CODE_START):
            if current and len(current) + len(part) > limit:
                yield current
                current = ''
            current += part
            if len(current) >= limit:
                yield current
                current = ''
            continue
        while part:
            take = limit - len(current)
            current += part[:take]
            part = part[take:]
            if len(current) == limit:
                yield current
                # An overlap is text-only; never duplicate a partial code block.
                tail = current[-overlap:] if overlap and part else ''
                if CODE_START in current and (CODE_END not in tail or CODE_START not in tail):
                    tail = ''
                current = tail
    if current.strip():
        yield current


def _terms(text):
    s = re.sub(r'\W+', '', text.lower())
    return {s[i:i+2] for i in range(len(s)-1)}


def _recursive(parts, limit, overlap):
    if limit <= 0 or overlap < 0 or overlap >= limit:
        raise ValueError('invalid chunk size or overlap')
    fragments = []
    for part in parts:
        if part.startswith(CODE_START):
            fragments.append(part)
            continue
        paragraphs = re.split(r'(?<=\n\n)', part)
        for paragraph in paragraphs:
            sentences = [paragraph] if len(paragraph) <= limit else re.findall(r'[^。！？\n]+[。！？\n]*|\n+', paragraph)
            for sentence in sentences:
                fragments.extend(sentence[i:i+limit] for i in range(0, len(sentence), limit))
    current = ''
    for fragment in fragments:
        if current and len(current) + len(fragment) > limit:
            yield current
            tail = current[-overlap:] if overlap else ''
            if CODE_START in current and (CODE_START not in tail or CODE_END not in tail):
                tail = ''
            current = tail if len(tail) + len(fragment) <= limit else ''
        current += fragment
        if len(current) >= limit:
            yield current
            current = ''
    if current.strip():
        yield current


def _semantic(parts, threshold, min_size=120):
    """按相邻句子的词面相似度切分，但短于 min_size 的组先不切，避免碎块。"""
    group = ''
    for part in parts:
        fragments = [part] if part.startswith(CODE_START) else re.findall(r'[^。！？\n]+[。！？\n]*|\n+', part)
        for fragment in fragments:
            a, b = _terms(group), _terms(fragment)
            similarity = len(a & b) / len(a | b) if a and b else 0
            if group and len(group) >= min_size and similarity < threshold:
                yield group
                group = ''
            group += fragment
    if group.strip():
        yield group


def chunk(docs, strategy='fixed', size=500, overlap=None, threshold=0.7, max_chars=800):
    if strategy not in ('fixed', 'semantic', 'recursive'):
        raise ValueError('Unknown chunking strategy')
    result = []
    # Keep sections separate, so every returned section_path describes its text.
    for (_, _), blocks in groupby(units(docs), key=lambda x: (x[0]['doc_id'], tuple(x[0]['section_path']))):
        blocks = list(blocks)
        meta = blocks[0][0]
        parts = [text + ('\n' if i < len(blocks) - 1 and not text.startswith(CODE_START) else '')
                 for i, (_, text) in enumerate(blocks)]
        if strategy == 'fixed':
            items = _pack(parts, size, 0 if overlap is None else overlap)
        elif strategy == 'semantic':
            items = _semantic(parts, threshold)
        else:
            items = _recursive(parts, max_chars, 150 if overlap is None else overlap)
        for item in items:
            if not item.strip():
                continue
            content = (' > '.join(meta['section_path']) + '\n' if strategy == 'recursive' else '') + item
            result.append(Chunk(f'{strategy}-{len(result):06}', meta['doc_id'], meta['path'],
                                list(meta['section_path']), content, (len(content) + 1) // 2))
    return result
