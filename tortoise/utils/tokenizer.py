import os
import re

import inflect
import torch
from tokenizers import Tokenizer


# Regular expression matching whitespace:
from unidecode import unidecode
from typing import List, Sequence


_whitespace_re = re.compile(r'\s+')


# List of (regular expression, replacement) pairs for abbreviations:
_abbreviations = [(re.compile('\\b%s\\.' % x[0], re.IGNORECASE), x[1]) for x in [
  ('mrs', 'misess'),
  ('mr', 'mister'),
  ('dr', 'doctor'),
  ('st', 'saint'),
  ('co', 'company'),
  ('jr', 'junior'),
  ('maj', 'major'),
  ('gen', 'general'),
  ('drs', 'doctors'),
  ('rev', 'reverend'),
  ('lt', 'lieutenant'),
  ('hon', 'honorable'),
  ('sgt', 'sergeant'),
  ('capt', 'captain'),
  ('esq', 'esquire'),
  ('ltd', 'limited'),
  ('col', 'colonel'),
  ('ft', 'fort'),
]]


def expand_abbreviations(text):
  for regex, replacement in _abbreviations:
    text = re.sub(regex, replacement, text)
  return text


_inflect = inflect.engine()
_comma_number_re = re.compile(r'([0-9][0-9\,]+[0-9])')
_decimal_number_re = re.compile(r'([0-9]+\.[0-9]+)')
_pounds_re = re.compile(r'£([0-9\,]*[0-9]+)')
_dollars_re = re.compile(r'\$([0-9\.\,]*[0-9]+)')
_ordinal_re = re.compile(r'[0-9]+(st|nd|rd|th)')
_number_re = re.compile(r'[0-9]+')
_over36digits_re = re.compile(r"\d{37,}")


def _remove_commas(m):
  return m.group(1).replace(',', '')


def _expand_decimal_point(m):
  return m.group(1).replace('.', ' point ')


def _expand_dollars(m):
  match = m.group(1)
  parts = match.split('.')
  if len(parts) > 2:
    return match + ' dollars'  # Unexpected format
  dollars = int(parts[0]) if parts[0] else 0
  cents = int(parts[1]) if len(parts) > 1 and parts[1] else 0
  if dollars and cents:
    dollar_unit = 'dollar' if dollars == 1 else 'dollars'
    cent_unit = 'cent' if cents == 1 else 'cents'
    return '%s %s, %s %s' % (dollars, dollar_unit, cents, cent_unit)
  elif dollars:
    dollar_unit = 'dollar' if dollars == 1 else 'dollars'
    return '%s %s' % (dollars, dollar_unit)
  elif cents:
    cent_unit = 'cent' if cents == 1 else 'cents'
    return '%s %s' % (cents, cent_unit)
  else:
    return 'zero dollars'


def _expand_ordinal(m):
  return _inflect.number_to_words(m.group(0))

def _split_long_number(m):
    s = m.group(0)
    return " ".join(s[i:i+36] for i in range(0, len(s), 36))

def _expand_number(m):
  num = int(m.group(0))
  if num > 1000 and num < 3000:
    if num == 2000:
      return 'two thousand'
    elif num > 2000 and num < 2010:
      return 'two thousand ' + _inflect.number_to_words(num % 100)
    elif num % 100 == 0:
      return _inflect.number_to_words(num // 100) + ' hundred'
    else:
      return _inflect.number_to_words(num, andword='', zero='oh', group=2).replace(', ', ' ')
  else:
    return _inflect.number_to_words(num, andword='')


def normalize_numbers(text):
  text = re.sub(_comma_number_re, _remove_commas, text)
  text = re.sub(_pounds_re, r'\1 pounds', text)
  text = re.sub(_dollars_re, _expand_dollars, text)
  text = re.sub(_decimal_number_re, _expand_decimal_point, text)
  text = re.sub(_ordinal_re, _expand_ordinal, text)
  text = re.sub(_over36digits_re, _split_long_number, text)
  text = re.sub(_number_re, _expand_number, text)
  return text


def expand_numbers(text):
  return normalize_numbers(text)


def lowercase(text):
  return text.lower()


def collapse_whitespace(text):
  return re.sub(_whitespace_re, ' ', text)


def convert_to_ascii(text):
  return unidecode(text)


def basic_cleaners(text):
  '''Basic pipeline that lowercases and collapses whitespace without transliteration.'''
  text = lowercase(text)
  text = collapse_whitespace(text)
  return text


def transliteration_cleaners(text):
  '''Pipeline for non-English text that transliterate to ASCII.'''
  text = convert_to_ascii(text)
  text = lowercase(text)
  text = collapse_whitespace(text)
  return text


def english_cleaners(text):
  '''Pipeline for English text, including number and abbreviation expansion.'''
  text = convert_to_ascii(text)
  text = lowercase(text)
  text = expand_numbers(text)
  text = expand_abbreviations(text)
  text = collapse_whitespace(text)
  text = text.replace('"', '')
  return text


def lev_distance(s1, s2):
  if len(s1) > len(s2):
    s1, s2 = s2, s1

  distances = range(len(s1) + 1)
  for i2, c2 in enumerate(s2):
    distances_ = [i2 + 1]
    for i1, c1 in enumerate(s1):
      if c1 == c2:
        distances_.append(distances[i1])
      else:
        distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
    distances = distances_
  return distances[-1]


DEFAULT_VOCAB_FILE = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../data/tokenizer.json')


class VoiceBpeTokenizer:
    def __init__(self, vocab_file=None, use_basic_cleaners=False):
        self.tokenizer = Tokenizer.from_file(
          DEFAULT_VOCAB_FILE if vocab_file is None else vocab_file
        )
        self.space_token_index = self.tokenizer.get_vocab()["[SPACE]"]
        if use_basic_cleaners:
            self.preprocess_text = basic_cleaners
        else:
            self.preprocess_text = english_cleaners

    def encode(self, txt):
        txt = self.preprocess_text(txt)
        txt = txt.replace(' ', '[SPACE]')
        return self.tokenizer.encode(txt).ids

    def encode_chunks(self, txt: str, max_chunk_tokens: int) -> Sequence[Sequence[int]]:
        return split_ids_on_spaces(
          self.encode(txt), space_token_index=self.space_token_index, max_chunk_length=max_chunk_tokens)
        

    def decode(self, seq):
        if isinstance(seq, torch.Tensor):
            seq = seq.cpu().numpy()
        txt = self.tokenizer.decode(seq, skip_special_tokens=False).replace(' ', '')
        txt = txt.replace('[SPACE]', ' ')
        txt = txt.replace('[STOP]', '')
        txt = txt.replace('[UNK]', '')
        return txt


def split_ids_on_spaces(
    ids: List[int],
    space_token_index: int,
    max_chunk_length: int,
) -> List[List[int]]:
    """
    WARNING: VIBE-CODED
    Split a sequence of ids into contiguous chunks of up to max_chunk_length.

    Chunks are split on space_token_index whenever possible and every chunk
    except the last ideally ends with space_token_index. If a contiguous span
    has no space_token_index and is longer than max_chunk_length, it is split
    evenly over arbitrary positions (this is the only case where non-space
    splits are allowed).
    """
    if max_chunk_length <= 0:
        raise ValueError("max_chunk_length must be positive")

    n = len(ids)
    if n == 0:
        return []

    spaces = [i for i, t in enumerate(ids) if t == space_token_index]
    result: List[List[int]] = []
    prev_idx = -1  # last index included in previous chunk
    space_pos = 0  # pointer into spaces list

    while prev_idx < n - 1:
        max_end = min(prev_idx + max_chunk_length, n - 1)

        # Advance to first space strictly after prev_idx.
        while space_pos < len(spaces) and spaces[space_pos] <= prev_idx:
            space_pos += 1

        # Try to end the chunk at the last space within (prev_idx, max_end].
        last_space = None
        j = space_pos
        while j < len(spaces) and spaces[j] <= max_end:
            last_space = spaces[j]
            j += 1

        if last_space is not None:
            start = prev_idx + 1
            end = last_space
            result.append(ids[start : end + 1])
            prev_idx = end
            space_pos = j
            continue

        # No space within the allowed window.
        next_space = spaces[space_pos] if space_pos < len(spaces) else None

        if next_space is None:
            # Remaining tail has no spaces at all.
            tail_start = prev_idx + 1
            tail_len = n - tail_start

            if tail_len <= max_chunk_length:
                result.append(ids[tail_start:])
                break

            # Split the tail evenly into k chunks, all <= max_chunk_length.
            k = (tail_len + max_chunk_length - 1) // max_chunk_length
            base = tail_len // k
            rem = tail_len % k
            offset = tail_start
            for i in range(k):
                size = base + (1 if i < rem else 0)
                result.append(ids[offset : offset + size])
                offset += size
            break

        # There is a space further on, but it lies beyond max_chunk_length.
        # The run [run_start, run_end) contains no spaces.
        run_start = prev_idx + 1
        run_end = next_space  # exclusive
        run_len = run_end - run_start

        if run_len <= max_chunk_length:
            # Short run with no spaces; must be a non-space chunk.
            result.append(ids[run_start:run_end])
            prev_idx = run_end - 1
        else:
            # Long run with no spaces: split evenly, ignoring space boundaries.
            k = (run_len + max_chunk_length - 1) // max_chunk_length
            base = run_len // k
            rem = run_len % k
            offset = run_start
            for i in range(k):
                size = base + (1 if i < rem else 0)
                result.append(ids[offset : offset + size])
                offset += size
            prev_idx = run_end - 1

    return result
