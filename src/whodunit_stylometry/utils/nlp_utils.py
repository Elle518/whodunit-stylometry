"""Exploratory Data Analysis (EDA) utilities for the corpus."""

import re
from collections import Counter

import numpy as np
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.tokenizer import Tokenizer
from spacy.util import compile_infix_regex, compile_prefix_regex, compile_suffix_regex

# Matches tokens composed of Unicode letters, optionally containing
# internal apostrophes or periods, and allowing a single trailing period.
# It excludes digits, underscores, leading punctuation, and trailing
# apostrophes.
# Examples of matching tokens: "don't", "rock'n'roll", "a.m.", "hello", "etc."
# Examples of non-matching tokens: "123", "3rd", '3:30', "\n\n", "B.31", "£"
ALPHA_TOKENS = re.compile(r"^[^\W\d_\.']+(?:[.'][^\W\d_\.']+)*\.?$", re.UNICODE)

nlp = spacy.load("en_core_web_sm", disable=["tagger", "attribute_ruler", "lemmatizer", "ner"])


class CustomTokenizer:
    """spaCy-based tokenizer with custom rules for apostrophes and abbreviations.

    This tokenizer customizes spaCy's default tokenization behavior to preserve
    words containing apostrophes (such as contractions or expressions like
    "rock'n'roll") and abbreviations containing periods (such as "a.m.",
    "e.g.", or "U.S.A.") as single tokens when possible.

    It also adjusts infix rules to avoid splitting tokens on apostrophes while
    keeping spaCy's default prefix and suffix handling.

    Args:
        lang: Language code used to initialize the blank spaCy pipeline.
        max_length: Maximum allowed length of the input text for the spaCy
            pipeline.

    Attributes:
        nlp: The configured spaCy language pipeline.
        _token_match: Compiled full-match regex used to preserve specific token
            patterns as single tokens.
    """

    def __init__(
        self,
        lang: str = "en",
        max_length: int = 15_000_000,
    ):
        """Initialize the tokenizer with custom spaCy tokenization rules.

        This constructor creates a blank spaCy pipeline for the specified
        language, adjusts the maximum input length, defines regex patterns to
        preserve apostrophe-containing words and dotted abbreviations as single
        tokens, and replaces the default tokenizer with a customized one.

        Args:
            lang: Language code used to initialize the blank spaCy pipeline.
            max_length: Maximum allowed length of the input text for the spaCy
                pipeline.
        """
        self.nlp = spacy.blank(lang)
        self.nlp.max_length = max_length

        # Words with apostrophes: don't, we'll, rock'n'roll
        apos_pat = r"[A-Za-z]+(?:['’][A-Za-z]+)+"

        # Abbreviations with periods: a.m., p.m., i.e., e.g., U.S.A., etc.
        # Requires a trailing period to preserve it within the token.
        abbrev_pat = r"(?:[A-Za-z]\.){2,}|(?:[A-Za-z]{1,10}\.){2,}"

        self._token_match = re.compile(rf"(?:{apos_pat})|(?:{abbrev_pat})").fullmatch

        prefix_re = compile_prefix_regex(self.nlp.Defaults.prefixes)
        suffix_re = compile_suffix_regex(self.nlp.Defaults.suffixes)

        # Remove apostrophes from infix rules to avoid splitting don't,
        # rock'n'roll, etc.
        infixes = [inf for inf in self.nlp.Defaults.infixes if "'" not in inf and "’" not in inf]
        infix_re = compile_infix_regex(infixes)

        self.nlp.tokenizer = Tokenizer(
            self.nlp.vocab,
            rules={},
            prefix_search=prefix_re.search,
            suffix_search=suffix_re.search,
            infix_finditer=infix_re.finditer,
            token_match=self._token_match,
        )

    def tokenize(self, text: str) -> list[str]:
        """Tokenize text and return the token strings.

        Args:
            text: Input text to tokenize.

        Returns:
            A list of token strings produced by the customized spaCy tokenizer.
        """
        return [tok.text for tok in self.nlp(text)]

    def doc(self, text: str) -> spacy.tokens.Doc:
        """Process text and return the resulting spaCy Doc object.

        Args:
            text: Input text to tokenize and process.

        Returns:
            A spaCy ``Doc`` object generated from the input text.
        """
        return self.nlp(text)

    def __call__(self, text: str) -> spacy.tokens.Doc:
        """Process text with the tokenizer instance as a callable.

        This method allows the tokenizer instance to be used like a function,
        returning the same result as calling the underlying spaCy pipeline.

        Args:
            text: Input text to tokenize and process.

        Returns:
            A spaCy ``Doc`` object generated from the input text.
        """
        return self.nlp(text)


def fix_gutenberg_linebreaks(text: str) -> str:
    """Join wrapped lines inside paragraphs while preserving paragraph breaks.

    This replaces single line breaks with spaces, but keeps empty lines
    as paragraph separators.

    Args:
        text: Raw text that may contain hard line breaks inside paragraphs.

    Returns:
        Text with intra-paragraph line breaks removed and paragraph
        structure preserved.
    """
    # Normalizes line breaks on Windows/Mac to \n
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Replace single jumps with spaces, but retain double jumps.
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)

    return text


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace and line breaks in a text string.

    This function standardizes line endings, collapses consecutive spaces and
    tabs into a single space, removes leading spaces after newline characters,
    reduces runs of three or more consecutive newlines to two, and strips
    leading and trailing whitespace from the final result.

    Args:
        text: Input text to normalize.

    Returns:
        A normalized version of the input text with consistent whitespace.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n ", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_text_for_tokenization(text: str) -> str:
    """Normalize text formatting to improve tokenization consistency.

    This function applies a series of text transformations intended to make
    tokenization more uniform. It removes or standardizes certain punctuation
    marks and symbols, expands ellipses into a consistent representation,
    inserts spaces around selected punctuation characters, replaces typographic
    quotation marks with ASCII equivalents, and normalizes whitespace in the
    final output.

    Args:
        text: Input text to normalize before tokenization.

    Returns:
        A normalized text string with standardized punctuation and whitespace,
        suitable for downstream tokenization.
    """

    text = re.sub(r"~", "", text)
    text = re.sub(r"…", "...", text)
    text = re.sub(r"(\. ?){3,}", " ... ", text)
    text = re.sub(r"\.\.\.", " ... ", text)
    text = re.sub(r"_", "", text)
    text = re.sub(r"\[", " [", text)
    text = re.sub(r"\]", "] ", text)
    text = re.sub(r"——+|--+", " — ", text)
    text = re.sub(r"-", " - ", text)
    text = re.sub(r"—", " — ", text)
    text = re.sub(r"‘|’", "'", text)
    text = re.sub(r"“|”", '"', text)
    text = normalize_whitespace(text)

    return text


def split_paragraphs(text: str) -> list[str]:
    """Splits text into paragraphs separated by blank lines.

    Paragraphs are identified using one or more blank lines (including lines
    containing only whitespace) as separators. Each paragraph is stripped of
    leading and trailing whitespace, and empty results are discarded.

    Args:
        text: Input text to split into paragraphs.

    Returns:
        A list of non-empty paragraph strings.
    """
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def is_alpha_tokens(tokens: list[str], keep_alpha: bool = True) -> list[str]:
    """Filter tokens based on whether they match the alpha-token pattern.

    Args:
        tokens: A list of input tokens to filter.
        keep_alpha: If True, return only tokens that match
            ``ALPHA_TOKENS``. If False, return only tokens that do not
            match ``ALPHA_TOKENS``.

    Returns:
        A list of filtered tokens, depending on the value of
        ``keep_alpha``.
    """
    return [t for t in tokens if bool(ALPHA_TOKENS.fullmatch(t)) == keep_alpha]


def split_sentences_by_paragraph(paragraphs: list, batch_size: int = 1000) -> list[list[str]]:
    result = []
    for doc in nlp.pipe(paragraphs, batch_size=batch_size):
        result.extend([s.text for s in doc.sents])

    return result


def get_length_stats(
    texts: list[str], tokenizer: CustomTokenizer, keep_alpha: bool = True, return_lengths: bool = False
):
    """Computes average and median text lengths in words.

    Each text is tokenized with the provided tokenizer, then filtered with
    `is_alpha_tokens`. Only non-empty tokenized texts are included in the
    statistics.

    Args:
        texts (list[str]): A list of text units, such as sentences or paragraphs.
        tokenizer (object): A tokenizer object with a `tokenize(text)` method.
        keep_alpha (bool, optional): Whether to keep only alphabetic tokens when
            calling `is_alpha_tokens`. Defaults to True.
        return_lengths (bool, optional): If True, also returns the list of
            individual text lengths. Defaults to False.

    Returns:
        tuple: If `return_lengths` is False, returns:
            (avg_len, median_len)

            If `return_lengths` is True, returns:
            (avg_len, median_len, lengths)

            - avg_len (float): Mean number of words per text.
            - median_len (float): Median number of words per text.
            - lengths (list[int], optional): Word counts for each non-empty text.

        If no valid texts are found, `avg_len` and `median_len` are `np.nan`.
    """
    lengths = []

    for text in texts:
        tokens = is_alpha_tokens(tokenizer.tokenize(text), keep_alpha=keep_alpha)
        if len(tokens) > 0:
            lengths.append(len(tokens))

    avg_len = np.mean(lengths) if lengths else np.nan
    median_len = np.median(lengths) if lengths else np.nan

    if return_lengths:
        return avg_len, median_len, lengths
    return avg_len, median_len


def text_quality_metrics(text: str) -> dict[str, float]:
    """Computes basic character-level text composition metrics.

    The function measures the total number of characters in the input text
    and the relative proportion of non-alphabetic characters, digits, and
    whitespace. These metrics are useful for detecting formatting artifacts,
    OCR noise, or other data quality issues in textual corpora.

    Args:
        text: The input text to analyze.

    Returns:
        A dictionary with the following keys:
            n_chars: Total number of characters in the text.
            non_alpha_ratio: Proportion of characters that are not alphabetic.
            digit_ratio: Proportion of characters that are digits.
            whitespace_ratio: Proportion of characters that are whitespace.
    """
    n_chars = len(text)
    n_alpha = sum(ch.isalpha() for ch in text)
    n_digit = sum(ch.isdigit() for ch in text)
    n_space = sum(ch.isspace() for ch in text)

    return {
        "n_chars": n_chars,
        "non_alpha_ratio": 1 - (n_alpha / n_chars),
        "digit_ratio": n_digit / n_chars,
        "whitespace_ratio": n_space / n_chars,
    }


def lexical_metrics(tokens_alpha: list[str]) -> dict[str, float]:
    """Computes basic lexical diversity and word-length metrics.

    The function derives vocabulary-based measures from a sequence of
    alphabetic tokens, including the number of distinct word types, the
    type-token ratio (TTR), the number and proportion of hapax legomena,
    and the average word length.

    To avoid inflating lexical diversity through purely orthographic variation,
    alphabetic tokens were normalized to lowercase before computing vocabulary-based metrics.

    Args:
        tokens_alpha: A list of alphabetic tokens from the input text.

    Returns:
        A dictionary with the following keys:
            n_types: Number of distinct token types.
            ttr: Type-token ratio, computed as the number of types divided
                by the number of tokens.
            hapax_count: Number of token types that occur exactly once.
            hapax_ratio: Proportion of hapax legomena relative to the
                number of types.
            avg_word_len: Mean token length in characters.

    Notes:
        The `ttr` value is sensitive to text length and is therefore best
        interpreted alongside more robust lexical diversity measures such
        as MATTR. If `tokens_alpha` is empty, `ttr`, `hapax_ratio`, and
        `avg_word_len` are returned as `np.nan`.
    """
    tokens_alpha = [t.lower() for t in tokens_alpha]
    n_tokens = len(tokens_alpha)
    counts = Counter(tokens_alpha)
    n_types = len(counts)
    hapax = sum(1 for _, c in counts.items() if c == 1)

    ttr = (n_types / n_tokens) if n_tokens else np.nan
    hapax_ratio = (hapax / n_types) if n_types else np.nan

    avg_word_len = np.mean([len(t) for t in tokens_alpha]) if tokens_alpha else np.nan

    return {
        "n_types": n_types,
        "ttr": ttr,
        "hapax_count": hapax,
        "hapax_ratio": hapax_ratio,
        "avg_word_len": avg_word_len,
    }


def mattr(tokens: list[str], window: int = 100) -> float:
    """Computes the Moving-Average Type-Token Ratio (MATTR).

    MATTR estimates lexical diversity by sliding a fixed-size window over
    the token sequence, computing the type-token ratio (TTR) for each
    window, and returning the mean of those values. This metric is more
    robust to text length than the standard TTR.

    Args:
        tokens: A list of input tokens.
        window: The size of the sliding window used to compute local TTR
            values. Defaults to 100.

    Returns:
        The mean TTR across all sliding windows as a float. Returns
        `np.nan` if the number of tokens is smaller than `window` or if
        `window <= 1`.
    """
    if len(tokens) < window or window <= 1:
        return np.nan
    vals = []
    for i in range(len(tokens) - window + 1):
        window_tokens = tokens[i : i + window]
        vals.append(len(set(window_tokens)) / window)
    return float(np.mean(vals)) if vals else np.nan


def punctuation_metrics(text: str) -> dict[str, float]:
    # Conteos básicos
    punct_counts = {
        "comma_count": text.count(","),
        "period_count": text.count("."),
        "semicolon_count": text.count(";"),
        "colon_count": text.count(":"),
        "exclam_count": text.count("!"),
        "question_count": text.count("?"),
        "ellipsis_count": len(re.findall(r"\.\.\.", text)),
        "quote_count": text.count('"'),
        "dialog_dash_count": len(re.findall(r"\s*—\s*", text)),
    }
    return punct_counts


def stopword_metrics(tokens_alpha: list[str], stopword_set: set) -> dict[str, float]:
    """Computes the proportion of stopwords among alphabetic tokens.

    The function lowercases all alphabetic tokens, counts how many of them
    belong to the provided stopword set, and returns the relative frequency
    of stopwords in the text.

    Args:
        tokens_alpha: A list of alphabetic tokens extracted from the text.
        stopword_set: A set of stopwords used to identify function-word
            tokens.

    Returns:
        A dictionary with the following key:
            stopword_ratio: Proportion of alphabetic tokens that are
                stopwords.

    Raises:
        ZeroDivisionError: If `tokens_alpha` is empty.
    """
    tokens_alpha = [t.lower() for t in tokens_alpha]
    n = len(tokens_alpha)
    sw = sum(1 for t in tokens_alpha if t in stopword_set)

    return {"stopword_ratio": sw / n}


def compute_novel_metrics(clean_text: str, stopword_set: set = None) -> dict[str, float]:

    stopwords = stopword_set if stopword_set else STOP_WORDS

    norm_text = normalize_text_for_tokenization(fix_gutenberg_linebreaks(clean_text))

    tokenizer = CustomTokenizer()
    tokens_all = tokenizer.tokenize(norm_text)
    tokens_alpha = is_alpha_tokens(tokens_all, keep_alpha=True)
    tokens_not_alpha = is_alpha_tokens(tokens_all, keep_alpha=False)

    paragraphs = split_paragraphs(norm_text)
    sentences = split_sentences_by_paragraph(paragraphs)

    avg_sentence_len, median_sentence_len = get_length_stats(sentences, tokenizer)
    avg_paragraph_len, median_paragraph_len = get_length_stats(paragraphs, tokenizer)

    out = {}
    out.update(text_quality_metrics(norm_text))
    out.update(lexical_metrics(tokens_alpha))
    out.update(stopword_metrics(tokens_alpha, stopwords))
    out.update(punctuation_metrics(norm_text))

    out["n_tokens_all"] = len(tokens_all)
    out["n_tokens_alpha"] = len(tokens_alpha)
    out["n_tokens_not_alpha"] = len(tokens_not_alpha)
    out["n_sentences"] = len(sentences)
    out["n_paragraphs"] = len(paragraphs)
    out["avg_sentence_len"] = avg_sentence_len
    out["median_sentence_len"] = median_sentence_len
    out["avg_paragraph_len"] = avg_paragraph_len
    out["median_paragraph_len"] = median_paragraph_len
    out["mattr_100"] = mattr(tokens_alpha, window=100)

    # Ratios por 1.000 palabras
    for col in [
        "comma_count",
        "period_count",
        "semicolon_count",
        "colon_count",
        "exclam_count",
        "question_count",
        "ellipsis_count",
        "quote_count",
        "dialog_dash_count",
    ]:
        out[f"{col}_per_1000"] = out[col] * 1000 / out["n_tokens_alpha"]

    return out


if __name__ == "__main__":

    text = """
    "Not till the words are said which make us man and wife," declared
Carleton Roberts. "Unless"--and here his perfect courtesy manifested
itself even in this crisis of life and death--"you feel it your duty to
carry what assistance you can to the saving of your frightened flock."

"God must save my flock," said the minister with a solemn glance upward.
"I am where my duty places me." And calmly as though the pews were
filled with guests and joy attended the ceremony instead of apprehended
doom, he proceeded with the rite.

"Wilt thou have this man...."

The glad "I will" leaped bravely from Ermentrude's lips; but it was lost
in loud calls and shrieks from without, mingled with that sound--terrible
to all who hear--impossible to describe--of the might of the hills made
audible in this down-rushing mass, now halting, now gathering fresh
momentum, but coming--always coming, till its voice, but now a threat,
swells into thunder in which all human cries are lost, and only from the
movement of the minister's lips can this couple see that the words which
make them one are being spoken.
    """

    text_n = normalize_text_for_tokenization(fix_gutenberg_linebreaks(text))

    print(text_n)

    print(punctuation_metrics(text_n))
