import pytest
from score import score_question


GOLD = {'gold_doc': 'a.md', 'gold_heading': '标题', 'gold_excerpt': '甲乙丙丁戊己庚辛壬癸'}


def hit(text, path='a.md', heading='# 标题'):
    return {'text': text, 'path': path, 'section_path': [heading]}


@pytest.mark.parametrize('text', ['甲乙丙丁戊己庚辛壬癸', '甲，乙 丙丁戊己庚辛'])
def test_one_block_gets_two(text):
    assert score_question(GOLD, [hit(text)]) == 2


@pytest.mark.parametrize('parts', [('甲乙丙丁', '戊己庚辛'), ('己庚辛壬癸', '甲乙丙丁戊己')])
def test_two_blocks_get_one(parts):
    assert score_question(GOLD, [hit(part) for part in parts]) == 1


@pytest.mark.parametrize('hits', [[hit('甲乙丙丁')], [hit('甲乙丙丁戊己庚辛壬癸', path='b.md')],
                                   [hit('甲乙丙丁戊己庚辛壬癸', heading='# 别处')]])
def test_insufficient_or_wrong_section_gets_zero(hits):
    assert score_question(GOLD, hits) == 0


def test_code_placeholder_and_recursive_prefix():
    gold = {**GOLD, 'gold_excerpt': '前文[代码块]后文内容'}
    assert score_question(gold, [hit('# 标题\n前文⟦CODE⟧print(1)⟦/CODE⟧后文内容')]) == 2
