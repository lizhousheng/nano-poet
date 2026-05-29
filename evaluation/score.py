"""Week 6/7 · 自动评分一首生成出来的诗。

维度:
- 句数是否匹配体裁(4 / 8)
- 每句字数是否匹配体裁(5 / 7)
- 偶数句尾字是否押韵(用 pypinyin 取韵尾)
- 题目相关字是否出现
- 是否过度重复(三连或以上)

返回 (score, details)。Week 7 用它做 DPO 偏好对的自动标注。

依赖:`pip install pypinyin`
"""
import re

from pypinyin import Style, lazy_pinyin

TARGET_LINES = {'五言绝句': 4, '七言绝句': 4, '五言律诗': 8, '七言律诗': 8}
TARGET_CHARS = {'五言绝句': 5, '七言绝句': 7, '五言律诗': 5, '七言律诗': 7}

_SENTENCE_SEP = re.compile(r'[，。、;:?!，。、；：？！]')


def get_rhyme(char: str) -> str | None:
    try:
        finals = lazy_pinyin(char, style=Style.FINALS)
        if not finals:
            return None
        f = finals[0]
        if len(f) <= 2:
            return f
        return f[-2:] if f[-1] in 'gn' else f[-1:]
    except Exception:
        return None


def rhymes(a: str, b: str) -> bool:
    ra, rb = get_rhyme(a), get_rhyme(b)
    return ra is not None and rb is not None and ra == rb


def score_poem(
    poem_text: str,
    target_style: str,
    target_title: str | None = None,
) -> tuple[int, dict[str, str]]:
    score = 0
    details: dict[str, str] = {}

    sentences = [s.strip() for s in _SENTENCE_SEP.split(poem_text) if s.strip()]
    if not sentences:
        return -50, {'fatal': '无法拆出有效句子'}

    target_lines = TARGET_LINES[target_style]
    target_chars = TARGET_CHARS[target_style]
    n_lines      = len(sentences)
    line_lens    = [len(s) for s in sentences]

    # 句数
    if n_lines == target_lines:
        score += 10
        details['句数'] = f'OK {n_lines}'
    else:
        score -= 5
        details['句数'] = f'BAD {n_lines} (期望 {target_lines})'

    # 字数
    correct = sum(1 for l in line_lens if l == target_chars)
    score += correct * 2
    for l in line_lens:
        if abs(l - target_chars) >= 2:
            score -= 3
    details['字数'] = f'{correct}/{n_lines} 句字数正确'

    # 押韵(偶数句尾字)
    even_endings = [sentences[i][-1] for i in range(1, n_lines, 2) if sentences[i]]
    if len(even_endings) >= 2:
        pairs = sum(
            1 for i in range(len(even_endings) - 1)
            if rhymes(even_endings[i], even_endings[i + 1])
        )
        score += pairs * 4
        details['押韵'] = f'+{pairs * 4} ({pairs}/{len(even_endings) - 1} 对) [{",".join(even_endings)}]'
    else:
        details['押韵'] = '句数太少无法判断'

    # 题目相关
    if target_title:
        relevant = sum(1 for c in target_title if c in poem_text)
        score += relevant * 3
        details['题目相关'] = f'+{relevant * 3} ({relevant}/{len(target_title)} 个题目字出现)'

    # 过度重复
    over_repeats = len(re.findall(r'(.)\1{2,}', poem_text))
    if over_repeats == 0:
        score += 3
        details['过度重复'] = 'OK 无'
    else:
        score -= over_repeats * 5
        details['过度重复'] = f'BAD {over_repeats} 处'

    return score, details
