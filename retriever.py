"""Small, deterministic hybrid retriever with SQLite vector persistence."""
import collections
import json
import math
import os
import re
import sqlite3
import struct
import urllib.request


def tokenize(text, use_jieba=False):
    text = text.lower()
    tokens = re.findall(r'[a-z0-9_]+|[\u4e00-\u9fff]+', text)
    result = []
    for token in tokens:
        if '\u4e00' <= token[0] <= '\u9fff':
            result.extend(token[i:i + 2] for i in range(max(1, len(token) - 1)))
        else:
            result.append(token)
    if use_jieba:
        try:
            import jieba
            result.extend(word for word in jieba.cut(text) if word.strip())
        except ImportError:
            pass
    return result


def fnv(text):
    h = 2166136261
    for byte in text.encode('utf-8'):
        h = ((h ^ byte) * 16777619) & 0xffffffff
    return h


def mock_embed(text):
    vec = [0.0] * 256
    text = re.sub(r'\s+', '', text.lower())
    grams = (text[i:i+2] for i in range(max(1, len(text)-1))) if text else ()
    for term in grams:
        vec[fnv(term) % 256] += 1.0
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec] if norm else vec


def openai_embed(texts):
    base, key, model = (os.environ[k] for k in ('BASE_URL', 'API_KEY', 'MODEL'))
    request = urllib.request.Request(base.rstrip('/') + '/embeddings',
        data=json.dumps({'model': model, 'input': texts}).encode(),
        headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=30) as response:
        data = json.load(response)['data']
    return [row['embedding'] for row in sorted(data, key=lambda row: row['index'])]


def minmax(values):
    if not values:
        return []
    low, high = min(values), max(values)
    return [(v - low) / (high - low) for v in values] if high > low else [0.0] * len(values)


class Retriever:
    def __init__(self, chunks, db_path, embedder='mock', use_jieba=False):
        self.chunks = chunks
        self.embedder = embedder
        self.use_jieba = use_jieba
        self.counts = [collections.Counter(tokenize(c.text, use_jieba)) for c in chunks]
        self.lengths = [sum(c.values()) for c in self.counts]
        self.avgdl = sum(self.lengths) / len(chunks) if chunks else 0
        self.postings = collections.defaultdict(list)
        for i, counts in enumerate(self.counts):
            for token, tf in counts.items():
                self.postings[token].append((i, tf))
        if embedder == 'mock':
            self.vectors = [mock_embed(c.text) for c in chunks]
        elif embedder == 'openai-compatible':
            self.vectors = []
            for i in range(0, len(chunks), 64):
                self.vectors.extend(openai_embed([c.text for c in chunks[i:i+64]]))
            self.vectors = [self._normalize(v) for v in self.vectors]
        else:
            raise ValueError('Unsupported embedder')
        with sqlite3.connect(db_path) as db:
            db.execute('CREATE TABLE IF NOT EXISTS chunks (chunk_id TEXT PRIMARY KEY, doc_id TEXT, path TEXT, section_path TEXT, text TEXT, embedding BLOB)')
            db.execute('DELETE FROM chunks')
            db.executemany('INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?)',
                [(c.chunk_id, c.doc_id, c.path, json.dumps(c.section_path, ensure_ascii=False), c.text,
                  struct.pack('<%sf' % len(v), *v)) for c, v in zip(chunks, self.vectors)])

    @staticmethod
    def _normalize(v):
        n = math.sqrt(sum(x*x for x in v))
        return [x/n for x in v] if n else v

    def scores(self, query):
        qv = mock_embed(query) if self.embedder == 'mock' else self._normalize(openai_embed([query])[0])
        if self.vectors and len(qv) != len(self.vectors[0]):
            raise ValueError('Embedding dimensions differ')
        nonzero = [(i, x) for i, x in enumerate(qv) if x]
        vector = [sum(v[i] * x for i, x in nonzero) for v in self.vectors]
        bm = [0.0] * len(self.chunks)
        for term in set(tokenize(query, self.use_jieba)):
            postings = self.postings.get(term, ())
            if not postings:
                continue
            idf = math.log(1 + (len(bm) - len(postings) + 0.5) / (len(postings) + 0.5))
            for i, tf in postings:
                bm[i] += idf * tf * 2.5 / (tf + 1.5 * (0.25 + 0.75 * self.lengths[i] / self.avgdl))
        return minmax(vector), minmax(bm)

    def search(self, query, top_k=5, vector_weight=0.5, bm25_weight=0.5, rerank='none', scores=None):
        vector, bm25 = scores if scores is not None else self.scores(query)
        order = sorted(range(len(self.chunks)), key=lambda i: (-(vector_weight * vector[i] + bm25_weight * bm25[i]), i))
        seen, hits = set(), []
        for i in order:
            c = self.chunks[i]
            key = (c.doc_id, c.text)
            if key in seen:
                continue
            seen.add(key)
            hits.append({**c.as_dict(), 'score': vector_weight * vector[i] + bm25_weight * bm25[i],
                         'vector_score': vector[i], 'bm25_score': bm25[i]})
            if len(hits) == top_k:
                break
        applied = False
        if rerank == 'cross-encoder' and hits:
            try:
                from sentence_transformers import CrossEncoder
                # Only a locally cached model may be used; no downloads.
                model = CrossEncoder(os.environ.get('RERANK_MODEL', 'cross-encoder/ms-marco-MiniLM-L-6-v2'), local_files_only=True)
                values = model.predict([(query, hit['text']) for hit in hits])
                for hit, value in zip(hits, values):
                    hit['rerank_score'] = float(value)
                hits.sort(key=lambda hit: -hit['rerank_score'])
                applied = True
            except Exception:
                pass
        return {'hits': hits, 'rerank_applied': applied, 'rerank_requested': rerank}
