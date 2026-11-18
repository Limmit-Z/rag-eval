from chunkers import Chunk
from retriever import Retriever, minmax, mock_embed, tokenize


def test_minmax_range_and_equal_scores():
    assert minmax([2, 4, 6]) == [0, .5, 1]
    assert minmax([9, 9]) == [0, 0]
    assert minmax([]) == []


def test_bm25_fusion_and_rerank_fallback(tmp_path, monkeypatch):
    chunks = [Chunk('1', 'a', 'a', ['# a'], '响应式响应式状态', 4),
              Chunk('2', 'b', 'b', ['# b'], '组件生命周期', 3)]
    r = Retriever(chunks, tmp_path / 'x.sqlite')
    vectors, bm = r.scores('响应式状态')
    assert bm == [1.0, 0.0]
    found = r.search('响应式状态', vector_weight=0, bm25_weight=1, rerank='cross-encoder')
    assert found['hits'][0]['chunk_id'] == '1'
    assert found['rerank_applied'] is False


def test_mock_is_reproducible_and_tokenizer_handles_chinese():
    assert mock_embed('响应式状态') == mock_embed('响应式状态')
    assert tokenize('响应式状态') == ['响应', '应式', '式状', '状态']
