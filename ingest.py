"""Read Vue Markdown into ordered, section-labelled text and code blocks."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CORPUS = ROOT / 'corpus/docs-zh-cn/src'


def clean_inline(text):
    text = re.sub(r'!\[([^]]*)\]\([^)]*\)', '', text)
    text = re.sub(r'\[([^]]+)\]\([^)]*\)', r'\1', text)
    text = re.sub(r'\[([^]]+)\]\[[^]]*\]', r'\1', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'(?<!\\)[`*_~]', '', text)
    return text


def parse_markdown(markdown, metadata):
    sections = []
    blocks = []
    pending = []
    in_code = False
    fence = ''

    def emit(kind, lines):
        text = '\n'.join(lines).strip()
        if text:
            blocks.append({**{k: metadata[k] for k in ('doc_id', 'path', 'title')},
                           'section_path': list(sections), 'block_type': kind,
                           'text': text if kind == 'code' else clean_inline(text)})

    for line in markdown.splitlines():
        marker = re.match(r'^\s*(`{3,}|~{3,})', line)
        if in_code:
            if marker and marker.group(1)[0] == fence[0] and len(marker.group(1)) >= len(fence):
                emit('code', pending)
                pending = []
                in_code = False
            else:
                pending.append(line)
            continue
        if marker:
            emit('text', pending)
            pending = []
            in_code = True
            fence = marker.group(1)
            continue
        heading = re.match(r'^(#{1,3})\s+(.+?)\s*#*\s*$', line)
        if heading:
            emit('text', pending)
            pending = []
            level = len(heading.group(1))
            title = re.sub(r'\s*\{#[^}]+\}\s*$', '', heading.group(2))
            sections[:] = sections[:level - 1]
            sections.append('#' * level + ' ' + clean_inline(title))
        else:
            pending.append(line)
    emit('code' if in_code else 'text', pending)
    return blocks


def load_documents():
    docs = []
    with (ROOT / 'data/corpus.jsonl').open(encoding='utf-8') as source:
        for line in source:
            meta = json.loads(line)
            path = (CORPUS / meta['path']).resolve()
            if not path.is_relative_to(CORPUS.resolve()):
                raise ValueError('Invalid corpus path')
            docs.extend(parse_markdown(path.read_text(encoding='utf-8'), meta))
    return docs


def main():
    docs = load_documents()
    out = ROOT / 'out'
    out.mkdir(exist_ok=True)
    with (out / 'docs.jsonl').open('w', encoding='utf-8') as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + '\n')
    return docs


if __name__ == '__main__':
    main()
