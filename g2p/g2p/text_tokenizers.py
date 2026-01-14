# Copyright (c) 2024 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import re
import os
from typing import List, Pattern, Union
from phonemizer.utils import list2str, str2list
from phonemizer.backend import EspeakBackend
from phonemizer.backend.espeak.language_switch import LanguageSwitch
from phonemizer.backend.espeak.words_mismatch import WordMismatch
from phonemizer.punctuation import Punctuation
from phonemizer.separator import Separator


class TextTokenizer:
    """Phonemize Text."""

    def __init__(
        self,
        language="en-us",
        backend="espeak",
        separator=Separator(word="|_|", syllable="-", phone="|"),
        preserve_punctuation=True,
        with_stress: bool = False,
        tie: Union[bool, str] = False,
        language_switch: LanguageSwitch = "remove-flags",
        words_mismatch: WordMismatch = "ignore",
    ) -> None:
        self.language = language
        self.preserve_punctuation_marks = ",.?!;:'…"
        self.preserve_punctuation = preserve_punctuation
        self.with_stress = with_stress
        self.tie = tie
        self.language_switch = language_switch
        self.words_mismatch = words_mismatch
        self.separator = separator
        self._backend = None

    def _get_backend(self):
        """Lazy load the EspeakBackend to avoid import-time errors."""
        if self._backend is None:
            try:
                self._backend = EspeakBackend(
                    self.language,
                    punctuation_marks=self.preserve_punctuation_marks,
                    preserve_punctuation=self.preserve_punctuation,
                    with_stress=self.with_stress,
                    tie=self.tie,
                    language_switch=self.language_switch,
                    words_mismatch=self.words_mismatch,
                )
            except RuntimeError as e:
                print(f"Warning: espeak not installed on your system. Error: {e}")
                print("Please install espeak to enable phonemization functionality.")
                return None
        return self._backend

    # convert chinese punctuation to english punctuation
    def convert_chinese_punctuation(self, text: str) -> str:
        text = text.replace("，", ",")
        text = text.replace("。", ".")
        text = text.replace("！", "!")
        text = text.replace("？", "?")
        text = text.replace("；", ";")
        text = text.replace("：", ":")
        text = text.replace("、", ",")
        text = text.replace("‘", "'")
        text = text.replace("’", "'")
        text = text.replace("⋯", "…")
        text = text.replace("···", "…")
        text = text.replace("・・・", "…")
        text = text.replace("...", "…")
        return text

    def __call__(self, text, strip=True) -> List[str]:
        backend = self._get_backend()
        if backend is None:
            print("Warning: phonemization is not available due to missing espeak installation.")
            print("Returning original text without phonemization.")
            
            text_type = type(text)
            normalized_text = []
            for line in str2list(text):
                line = self.convert_chinese_punctuation(line.strip())
                line = re.sub(r"[^\w\s_,\.\?!;:\'…]", "", line)
                line = re.sub(r"\s*([,\.\?!;:\'…])\s*", r"\1", line)
                line = re.sub(r"\s+", " ", line)
                normalized_text.append(line)
            
            if text_type == str:
                phonemized = list2str(normalized_text)
                phonemized = re.sub(r"([,\.\?!;:\'…])", r"|\1|", phonemized)
                phonemized = re.sub(r"\|+", "|", phonemized)
                phonemized = phonemized.rstrip("|")
                return phonemized
            else:
                for i in range(len(normalized_text)):
                    normalized_text[i] = re.sub(r"([,\.\?!;:\'…])", r"|\1|", normalized_text[i])
                    normalized_text[i] = re.sub(r"\|+", "|", normalized_text[i])
                    normalized_text[i] = normalized_text[i].rstrip("|")
                return normalized_text

        text_type = type(text)
        normalized_text = []
        for line in str2list(text):
            line = self.convert_chinese_punctuation(line.strip())
            line = re.sub(r"[^\w\s_,\.\?!;:\'…]", "", line)
            line = re.sub(r"\s*([,\.\?!;:\'…])\s*", r"\1", line)
            line = re.sub(r"\s+", " ", line)
            normalized_text.append(line)
        # print("Normalized test: ", normalized_text[0])
        phonemized = backend.phonemize(
            normalized_text, separator=self.separator, strip=strip, njobs=1
        )
        if text_type == str:
            phonemized = re.sub(r"([,\.\?!;:\'…])", r"|\1|", list2str(phonemized))
            phonemized = re.sub(r"\|+", "|", phonemized)
            phonemized = phonemized.rstrip("|")
        else:
            for i in range(len(phonemized)):
                phonemized[i] = re.sub(r"([,\.\?!;:\'…])", r"|\1|", phonemized[i])
                phonemized[i] = re.sub(r"\|+", "|", phonemized[i])
                phonemized[i] = phonemized[i].rstrip("|")
        return phonemized
