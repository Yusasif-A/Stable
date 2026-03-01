"""
Yoruba phoneme converter using epitran.
Converts Yoruba text to IPA phonemes compatible with the StableTTS symbol set.
Tone marks are stripped since they are not in the existing symbol vocabulary.
"""

import unicodedata
import epitran

from text.symbols import symbols

_VALID_SYMBOLS = set(symbols)

# Map IPA 'ɡ' (U+0261) to ASCII 'g' (U+0067) which is in the symbol set
_CHAR_MAP = {
    '\u0261': 'g',  # ɡ → g
}

_epi = None

def _get_epi():
    global _epi
    if _epi is None:
        _epi = epitran.Epitran('yor-Latn')
    return _epi


def yoruba_to_ipa(text: str) -> list:
    epi = _get_epi()
    ipa = epi.transliterate(text)

    # NFD decompose: splits precomposed chars like 'à' → 'a' + combining grave
    ipa = unicodedata.normalize('NFD', ipa)

    # Drop all combining diacritical marks (tone marks, nasalization ̃, tie ͡, etc.)
    ipa = ''.join(c for c in ipa if unicodedata.category(c) != 'Mn')

    # Apply character mappings (e.g. IPA ɡ → ASCII g)
    for src, dst in _CHAR_MAP.items():
        ipa = ipa.replace(src, dst)

    # Filter to only characters in the StableTTS symbol set
    result = [c for c in ipa if c in _VALID_SYMBOLS]
    return result
