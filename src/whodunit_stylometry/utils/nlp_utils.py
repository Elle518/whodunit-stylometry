"""Exploratory Data Analysis (EDA) utilities for the corpus."""

import re
import statistics
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import spacy
from nltk import ngrams
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.tokenizer import Tokenizer
from spacy.util import compile_infix_regex, compile_prefix_regex, compile_suffix_regex

from whodunit_stylometry.utils.data_utils import read_book_text

# Matches tokens composed of Unicode letters, optionally containing
# internal apostrophes or periods, and allowing a single trailing period.
# It excludes digits, underscores, leading punctuation, and trailing
# apostrophes.
# Examples of matching tokens: "don't", "rock'n'roll", "a.m.", "hello"
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

        # Abbreviations with periods: a.m., p.m., i.e., e.g., U.S.A.
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
    text = re.sub(r"—+”", " ... ”", text)
    text = re.sub(r"-+”", " ... ”", text)
    text = re.sub(r'—+"', ' ... "', text)
    text = re.sub(r'-+"', ' ... "', text)
    text = re.sub(r"_", "", text)
    text = re.sub(r"\[", " [", text)
    text = re.sub(r"\]", "] ", text)
    text = re.sub(r"——+|--+", " — ", text)
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
    """Split each paragraph into a list of sentences using a spaCy pipeline.

    The function processes the input paragraphs in batches with ``nlp.pipe``
    and extracts sentence strings from ``doc.sents``. Leading and trailing
    whitespace is stripped, and empty sentences are discarded.

    Notes:
        This function expects a global spaCy ``nlp`` object to be available in
        scope and configured with sentence boundary detection (e.g., a parser
        or a ``sentencizer`` component).

    Args:
        paragraphs: List of paragraph texts to be segmented into sentences.
        batch_size: Number of paragraphs to process per batch in ``nlp.pipe``.
            Larger values may improve throughput at the cost of memory.

    Returns:
        A list where each element corresponds to the input paragraph at the
        same index and contains the list of sentence strings extracted from it.
    """

    result = []

    for doc in nlp.pipe(paragraphs, batch_size=batch_size):
        sentences = [s.text.strip() for s in doc.sents if s.text.strip()]
        result.append(sentences)

    return result


def get_length_stats(
    texts: list[str], tokenizer: CustomTokenizer, keep_alpha: bool = True, return_lengths: bool = False
):
    """Computes average, median and standard deviation of text lengths in words.

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
            (avg_len, median_len, std_len)

            If `return_lengths` is True, returns:
            (avg_len, median_len, std_len, lengths)

            - avg_len (float): Mean number of words per text.
            - median_len (float): Median number of words per text.
            - std_len (float): Standard deviation of word counts per text.
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
    std_len = np.std(lengths) if lengths else np.nan

    if return_lengths:
        return avg_len, median_len, std_len, lengths
    return avg_len, median_len, std_len


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
            median_word_len: Median token length in characters.
            long_word_ratio: Proportion of tokens longer than 8 characters.

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
    median_word_len = np.median([len(t) for t in tokens_alpha]) if tokens_alpha else np.nan

    return {
        "n_types": n_types,
        "ttr": ttr,
        "hapax_count": hapax,
        "hapax_ratio": hapax_ratio,
        "avg_word_len": avg_word_len,
        "median_word_len": median_word_len,
        "long_word_ratio": compute_long_word_ratio(tokens_alpha),
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


def punctuation_metrics(text: str, n_tokens: int) -> dict[str, float]:
    """Counts major punctuation marks in a text.

    The function computes absolute frequencies for a set of punctuation
    features, including commas, periods, semicolons, colons, exclamation
    marks, question marks, ellipses, quotation marks, and dialogue dashes.

    Args:
        text: The input text to analyze.
        n_tokens: The total number of tokens in the text.

    Returns:
        A dictionary with the following keys:
            comma_count: Number of commas.
            period_count: Number of periods.
            semicolon_count: Number of semicolons.
            colon_count: Number of colons.
            exclam_count: Number of exclamation marks.
            question_count: Number of question marks.
            ellipsis_count: Number of ellipsis sequences ("...").
            quote_count: Number of double quotation marks.
            dialog_dash_count: Number of dialogue dashes.
            hyphen_count: Number of hyphens.
            parenthesis_count: Total number of parentheses (both "(" and ")").
            punctuation_count_total: Total count of all the above punctuation marks.
            punctuation_ratio: Total punctuation count divided by the number of tokens.

    Notes:
        These are raw counts rather than normalized frequencies, so they
        are sensitive to text length and may also reflect editorial or
        typographic conventions.
    """

    punct_counts = {
        "comma_count": text.count(","),
        "period_count": len(re.findall(r"(?<!\.)\.(?!\.)", text)),
        "semicolon_count": text.count(";"),
        "colon_count": text.count(":"),
        "exclam_count": text.count("!"),
        "question_count": text.count("?"),
        "ellipsis_count": len(re.findall(r"\.\.\.", text)),
        "quote_count": text.count('"'),
        "dialog_dash_count": len(re.findall(r"—", text)),
        "hyphen_count": text.count("-"),
        "parenthesis_count": text.count("(") + text.count(")"),
    }

    # Total punctuation (sum of the chosen categories)
    punct_counts["punctuation_count_total"] = sum(punct_counts.values())

    # Ratio (punctuation per token)
    punct_counts["punctuation_ratio"] = punct_counts["punctuation_count_total"] / n_tokens

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
    """

    tokens_alpha = [t.lower() for t in tokens_alpha]
    n = len(tokens_alpha)
    sw = sum(1 for t in tokens_alpha if t in stopword_set)

    return {"stopword_ratio": sw / n}


def compute_long_word_ratio(tokens: list[str], min_len: int = 8) -> float:
    """Compute the proportion of tokens longer than a given character length.

    A token is considered a "long word" if its character length is strictly
    greater than ``min_len``. The ratio is computed as:

        (# tokens with len(token) > min_len) / (total # tokens)

    Args:
        tokens: List of token strings.
        min_len: Minimum character length threshold. Tokens with length
            strictly greater than this value are counted as long words.
            Defaults to 8.

    Returns:
        The proportion of long-word tokens in ``tokens``.
    """
    return sum(len(token) > min_len for token in tokens) / len(tokens)


def sentence_length_ratios(
    sentence_lengths: list[int],
    short_max: int = 10,
    long_min: int = 30,
) -> tuple[float, float]:
    """Compute short- and long-sentence ratios from sentence token lengths.

    The ratios are computed over the input list of sentence lengths (in tokens)
    as:

    - short_sentence_ratio = (# lengths < short_max) / N
    - long_sentence_ratio = (# lengths > long_min) / N

    Args:
        sentence_lengths: List of sentence lengths measured in tokens.
        short_max: Upper bound (exclusive) for a sentence to be considered
            short. Defaults to 10.
        long_min: Lower bound (exclusive) for a sentence to be considered
            long. Defaults to 30.

    Returns:
        A tuple ``(short_sentence_ratio, long_sentence_ratio)``.
    """

    n = len(sentence_lengths)

    short_ratio = sum(length < short_max for length in sentence_lengths) / n
    long_ratio = sum(length > long_min for length in sentence_lengths) / n

    return short_ratio, long_ratio


def compute_novel_metrics(clean_text: str, stopword_set: set = None) -> dict[str, float]:
    """Compute a stylometric feature set for a normalized novel text.

    This function preprocesses the input text, tokenizes it, separates
    alphabetic and non-alphabetic tokens, splits the text into paragraphs
    and sentences, and computes a broad set of descriptive metrics. These
    include text quality indicators, token and sentence counts, sentence and
    paragraph length statistics, lexical richness measures, stopword-based
    measures, punctuation frequencies, and punctuation counts normalized per
    1,000 tokens.

    Args:
        clean_text: The input novel text to analyze.
        stopword_set: An optional set of stopwords to use for stopword-based
            metrics. If ``None``, the default ``STOP_WORDS`` set is used.

    Returns:
        A dictionary mapping metric names to float values. The output includes
        structural, lexical, stopword, and punctuation-based stylometric
        features, as well as normalized rates per 1,000 tokens for selected
        punctuation marks.
    """

    stopwords = stopword_set if stopword_set else STOP_WORDS

    tokenizer = CustomTokenizer()
    tokens_all = tokenizer.tokenize(clean_text)
    tokens_alpha = is_alpha_tokens(tokens_all, keep_alpha=True)
    tokens_not_alpha = is_alpha_tokens(tokens_all, keep_alpha=False)

    paragraphs = split_paragraphs(clean_text)
    paragraph_sentences = split_sentences_by_paragraph(paragraphs)
    sentences = [s for para in paragraph_sentences for s in para]
    sentence_counts_per_paragraph = [len(sents) for sents in paragraph_sentences]

    avg_sentences_per_paragraph = sum(sentence_counts_per_paragraph) / len(sentence_counts_per_paragraph)

    median_sentences_per_paragraph = statistics.median(sentence_counts_per_paragraph)

    avg_sentence_len, median_sentence_len, std_sentence_len, sentence_lenghts = get_length_stats(
        sentences, tokenizer, return_lengths=True
    )
    short_sentence_ratio, long_sentence_ratio = sentence_length_ratios(sentence_lenghts)
    avg_paragraph_len, median_paragraph_len, std_paragraph_len = get_length_stats(paragraphs, tokenizer)

    out = {}
    out.update(text_quality_metrics(clean_text))

    out["n_tokens_all"] = len(tokens_all)
    out["n_tokens_alpha"] = len(tokens_alpha)
    out["n_tokens_not_alpha"] = len(tokens_not_alpha)
    out["non_alpha_token_ratio"] = len(tokens_not_alpha) / len(tokens_all)
    out["n_sentences"] = len(sentences)
    out["n_paragraphs"] = len(paragraphs)
    out["avg_sentence_len"] = avg_sentence_len
    out["median_sentence_len"] = median_sentence_len
    out["std_sentence_len"] = std_sentence_len
    out["sentences_per_1000_tokens"] = len(sentences) * 1000 / len(tokens_all)
    out["short_sentence_ratio"] = short_sentence_ratio
    out["long_sentence_ratio"] = long_sentence_ratio
    out["avg_paragraph_len"] = avg_paragraph_len
    out["median_paragraph_len"] = median_paragraph_len
    out["std_paragraph_len"] = std_paragraph_len
    out["paragraphs_per_1000_tokens"] = len(paragraphs) * 1000 / len(tokens_all)
    out["avg_sentences_per_paragraph"] = avg_sentences_per_paragraph
    out["median_sentences_per_paragraph"] = median_sentences_per_paragraph

    out.update(lexical_metrics(tokens_alpha))
    out["mattr_100"] = mattr(tokens_alpha, window=100)

    out.update(stopword_metrics(tokens_alpha, stopwords))

    out.update(punctuation_metrics(clean_text, n_tokens=len(tokens_all)))

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
        "hyphen_count",
        "parenthesis_count",
    ]:
        out[f"{col}_per_1000"] = out[col] * 1000 / out["n_tokens_all"]

    return out


def get_tokens_alpha_by_author(df: pd.DataFrame, is_lower: bool = True) -> dict[str, list[str]]:
    """Collect alphabetic tokens from book texts grouped by author.

    For each author in the input DataFrame, this function reads the text of all
    associated books, normalizes the content for tokenization, tokenizes it,
    filters the tokens to keep only alphabetic ones, and optionally lowercases
    them. It returns a dictionary mapping each author to the aggregated list of
    tokens from all of their books.

    Args:
        df: A DataFrame containing at least the columns ``"author"`` and
            ``"path"``. Each row represents a book, where ``"author"``
            identifies the author and ``"path"`` points to the text file.
        is_lower: Whether to convert alphabetic tokens to lowercase before
            adding them to the output. Defaults to ``True``.

    Returns:
        A dictionary where each key is an author name and each value is a list
        of alphabetic tokens aggregated from all books associated with that
        author.
    """

    tokenizer = CustomTokenizer()

    out = {}
    for author, sub in df.groupby("author"):
        toks = []
        for _, row in sub.iterrows():
            text = read_book_text(Path(row["path"]))
            tokens_all = tokenizer.tokenize(text)
            tokens_alpha = is_alpha_tokens(tokens_all, keep_alpha=True)
            if is_lower:
                tokens_alpha = [t.lower() for t in tokens_alpha]
            toks.extend(tokens_alpha)
        out[author] = toks

    return out


def compute_word_frecuencies(
    alpha_tokens: dict, stopword_set: set = None, remove_stopwords: bool = False
) -> dict[str, float]:
    """Computes word frequencies for each author from tokenized texts.

    The function iterates over a dictionary whose keys are author identifiers
    and whose values are token lists. For each author, it builds a `Counter`
    with the frequency of each token. Optionally, it can exclude tokens that
    appear in a provided stopword set before counting.

    Args:
        alpha_tokens (dict): A dictionary mapping each author to a list of tokens
            associated with that author's text or corpus.
        stopword_set (set, optional): A set of stopwords to remove before counting. Defaults
            to None.
        remove_stopwords (bool, optional): Whether to filter out stopwords before computing
            frequencies. Defaults to False.

    Returns:
        A dictionary mapping each author to a `Counter` containing token
        frequencies.
    """

    counters = {}

    for author, toks in alpha_tokens.items():
        c = Counter()
        if remove_stopwords and stopword_set is not None:
            toks = [t for t in toks if t not in stopword_set]
        c.update(toks)
        counters[author] = c

    return counters


def top_n_words(counter: Counter, n: int = 20) -> pd.DataFrame:
    """Returns the `n` most frequent tokens as a pandas DataFrame.

    The function extracts the most common items from a `Counter` object and
    converts them into a DataFrame with two columns: one for the token and
    another for its frequency.

    Args:
        counter: A `Counter` containing token frequencies.
        n: The number of most frequent tokens to return. Defaults to 20.

    Returns:
        A pandas DataFrame with columns `"token"` and `"freq"`, sorted by
        descending frequency.
    """
    data = counter.most_common(n)
    return pd.DataFrame(data, columns=["token", "freq"])


def top_ngrams(tokens: list[str], n: int = 2, top_k: int = 20, stopword_set: set = None) -> list[tuple[str, int]]:
    """Return the most frequent n-grams from a token sequence.

    This function optionally removes stopwords from the input token list,
    generates n-grams of size ``n``, counts their frequencies, and returns
    the ``top_k`` most common n-grams as pairs of joined n-gram text and
    frequency.

    Args:
        tokens: A list of input tokens from which to generate n-grams.
        n: The size of the n-grams to generate. For example, ``2`` for
            bigrams and ``3`` for trigrams. Defaults to ``2``.
        top_k: The maximum number of most frequent n-grams to return.
            Defaults to ``20``.
        stopword_set: An optional set of stopwords to exclude from the token
            sequence before generating n-grams. Defaults to ``None``.

    Returns:
        A list of tuples where each tuple contains:
            - The n-gram as a space-joined string.
            - Its frequency as an integer.

        The list is sorted in descending order of frequency.
    """

    if stopword_set is not None:
        tokens = [t for t in tokens if t not in stopword_set]

    ng = ngrams(tokens, n)
    c = Counter(ng)

    return [(" ".join(t), f) for t, f in c.most_common(top_k)]


def compute_ngrams_frecuencies(
    alpha_tokens: dict, stopword_set: set = None, ngram_n: int = 2, top_k: int = 20
) -> dict[str, Counter]:
    """Compute the most frequent n-grams for each author.

    This function takes a dictionary mapping authors to token lists and computes
    the top ``k`` most frequent n-grams for each author using ``top_ngrams``.
    A stopword set can be provided to exclude n-grams containing stopwords,
    depending on the behavior of ``top_ngrams``.

    Args:
        alpha_tokens: A dictionary where each key is an author name and each
            value is a list of tokens associated with that author.
        stopword_set: An optional set of stopwords used when computing n-grams.
            Defaults to ``None``.
        ngram_n: The size of the n-grams to compute. For example, ``2`` for
            bigrams and ``3`` for trigrams. Defaults to ``2``.
        top_k: The maximum number of most frequent n-grams to return per
            author. Defaults to ``20``.

    Returns:
        A dictionary mapping each author to a ``Counter`` containing the most
        frequent n-grams for that author's token list.
    """

    out = {}

    for author, toks in alpha_tokens.items():
        out[author] = top_ngrams(toks, n=ngram_n, top_k=top_k, stopword_set=stopword_set)

    return out


def build_mfw_features(tokens_by_file, function_words, top_n=100):
    function_words = set(function_words)

    # 2. frecuencia global de function words en train
    global_fw_counts = Counter()
    for tokens in tokens_by_file.values():
        global_fw_counts.update(tok for tok in tokens if tok in function_words)

    # 3. seleccionar MFW
    mfw = [word for word, _ in global_fw_counts.most_common(top_n)]

    # 4. matriz de features por obra
    rows = []
    for file_name, tokens in tokens_by_file.items():
        counts = Counter(tokens)
        total_tokens = len(tokens)

        row = {"file_name": file_name}
        for word in mfw:
            row[f"fw_{word}"] = counts[word] / total_tokens if total_tokens > 0 else 0.0

        rows.append(row)

    train_mfw_df = pd.DataFrame(rows)

    return mfw, train_mfw_df


def transform_with_mfw(tokens_by_file, mfw):
    rows = []

    for file_name, tokens in tokens_by_file.items():
        counts = Counter(tokens)
        total_tokens = len(tokens)

        row = {"file_name": file_name}
        for word in mfw:
            row[f"fw_{word}"] = counts[word] / total_tokens if total_tokens > 0 else 0.0

        rows.append(row)

    return pd.DataFrame(rows)


def tokenize_text(text: str, tokenizer: CustomTokenizer, lowercase: bool = True) -> list[str]:
    """Normalize and tokenize text, keeping only alphabetic tokens.

    Args:
        text: Text to normalize and tokenize.
        tokenizer: Tokenizer used to split the normalized text into tokens.
        lowercase: Whether to lowercase the returned tokens.

    Returns:
        A list of alphabetic tokens. Tokens are lowercased when ``lowercase`` is
        ``True``.
    """
    tokens = is_alpha_tokens(tokenizer.tokenize(text), keep_alpha=True)
    if lowercase:
        return [token.lower() for token in tokens]
    return tokens


def add_tokens(df: pd.DataFrame, lowercase: bool = True) -> pd.DataFrame:
    """Attach token lists and lexical statistics to a corpus table.

    Args:
        df: Corpus table containing a ``text`` column with the text to tokenize.
        lowercase: Whether to lowercase text during tokenization.

    Returns:
        A copy of ``df`` with four additional columns:
            - ``tokens``: Token list for each text.
            - ``token_count``: Number of tokens in each token list.
            - ``type_count``: Number of unique tokens in each token list.
            - ``ttr``: Type-token ratio, computed as ``type_count / token_count``.
              Empty token lists receive ``np.nan``.
    """
    tokenizer = CustomTokenizer()
    out = df.copy()
    out["tokens"] = [tokenize_text(text, tokenizer=tokenizer, lowercase=lowercase) for text in out["text"]]
    out["token_count"] = out["tokens"].map(len)
    out["type_count"] = out["tokens"].map(lambda tokens: len(set(tokens)))
    out["ttr"] = out.apply(
        lambda row: row["type_count"] / row["token_count"] if row["token_count"] else np.nan,
        axis=1,
    )
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
    print(punctuation_metrics(text_n, n_tokens=100))
