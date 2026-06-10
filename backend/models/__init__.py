"""LipNet 베이스라인, CTC 디코더, 한국어 자모 어휘."""

from .baseline import LipNet, LipNetConfig
from .decoder import GreedyCTCDecoder

# 자모 41클래스 = 자음 19 + 모음 21 + blank(1) — blank는 인덱스 0에서 별도 처리
GRAPHEME_VOCAB: tuple[str, ...] = (
    # 자음 19
    "ㄱ", "ㄲ", "ㄴ", "ㄷ", "ㄸ", "ㄹ", "ㅁ", "ㅂ", "ㅃ", "ㅅ",
    "ㅆ", "ㅇ", "ㅈ", "ㅉ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ",
    # 모음 21
    "ㅏ", "ㅐ", "ㅑ", "ㅒ", "ㅓ", "ㅔ", "ㅕ", "ㅖ", "ㅗ", "ㅘ",
    "ㅙ", "ㅚ", "ㅛ", "ㅜ", "ㅝ", "ㅞ", "ㅟ", "ㅠ", "ㅡ", "ㅢ",
    "ㅣ",
)
assert len(GRAPHEME_VOCAB) == 40, f"expected 40 jamo (blank 제외), got {len(GRAPHEME_VOCAB)}"

VOCAB_SIZE: int = len(GRAPHEME_VOCAB) + 1   # +1 for blank

__all__ = ["LipNet", "LipNetConfig", "GreedyCTCDecoder", "GRAPHEME_VOCAB", "VOCAB_SIZE"]
