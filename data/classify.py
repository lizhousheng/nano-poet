"""Week 6 · 把每首唐诗判成 五绝/七绝/五律/七律/其他。

判定规则:按标点切分后,所有句子字数一致,而且
- 4 句 × 5 字 = 五言绝句
- 4 句 × 7 字 = 七言绝句
- 8 句 × 5 字 = 五言律诗
- 8 句 × 7 字 = 七言律诗
其余归 'other'(不进 SFT 数据集)。
"""
import re

STYLE_NAME = {
    'wuyan_jueju': '五言绝句',
    'qiyan_jueju': '七言绝句',
    'wuyan_lvshi': '五言律诗',
    'qiyan_lvshi': '七言律诗',
}

_SENTENCE_SEP = re.compile(r'[，。、;:?!，。、；：？！]')


def classify(paragraphs: list[str]) -> str:
    if not paragraphs:
        return 'other'
    full = ''.join(paragraphs)
    sentences = [s for s in _SENTENCE_SEP.split(full) if s.strip()]
    if not sentences:
        return 'other'

    char_counts = [len(s) for s in sentences]
    if len(set(char_counts)) > 1:                       # 句长不一致 → 不属于以上 4 类
        return 'other'

    n, k = len(sentences), char_counts[0]
    if n == 4 and k == 5: return 'wuyan_jueju'
    if n == 4 and k == 7: return 'qiyan_jueju'
    if n == 8 and k == 5: return 'wuyan_lvshi'
    if n == 8 and k == 7: return 'qiyan_lvshi'
    return 'other'
