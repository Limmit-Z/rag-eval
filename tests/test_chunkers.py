import pytest
from chunkers import CODE_START, CODE_END, chunk
from ingest import parse_markdown


@pytest.mark.parametrize('strategy', ['fixed', 'semantic', 'recursive'])
def test_long_code_is_kept_whole(strategy):
    code = 'let x = 1;\n' * 90
    blocks = parse_markdown('# 根\n## 节\n说明。\n```js\n' + code + '```\n后文。',
                            {'doc_id': 'd', 'path': 'x.md', 'title': 'x'})
    hits = chunk(blocks, strategy=strategy)
    containing = [c.text for c in hits if CODE_START in c.text]
    assert len(containing) == 1
    assert CODE_START + code.strip() + CODE_END in containing[0]
    assert all(c.section_path == ['# 根', '## 节'] for c in hits)


def test_fixed_boundaries_and_overlap():
    blocks = [{'doc_id': 'd', 'path': 'x', 'title': 'x', 'section_path': ['# 标题'], 'block_type': 'text', 'text': 'abcdefghij'}]
    hits = chunk(blocks, size=4, overlap=1)
    assert [c.text for c in hits] == ['abcd', 'defg', 'ghij']


def test_recursive_preserves_paragraph_boundaries():
    blocks = [{'doc_id': 'd', 'path': 'x', 'title': 'x', 'section_path': ['# 标题'], 'block_type': 'text', 'text': '甲乙丙\n\n丁戊己\n\n庚辛壬'}]
    hits = chunk(blocks, strategy='recursive', max_chars=9, overlap=0)
    assert hits[0].text == '# 标题\n甲乙丙\n\n'
    assert hits[1].text.startswith('# 标题\n丁戊己')
