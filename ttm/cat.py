#!/usr/bin/env python3

from getopt import getopt
from .types import *
import os, re

def _normalize(doc):
    doc = re.sub(r'-\n', '', doc)
    doc = re.sub(r'\s+', ' ', doc)
    return doc

def _split_doc(path, filename, window, step):
    """
    Split the text file at 'path/filename' into documents of up to
    'window' tokens each. The resulting iterator yields tuples of
    the following format: (doc_id, tokens) where tokens is a list
    of strings.
    """
    if window % step != 0:
        raise Exception('window must be divisible by step')
    with open(os.path.join(path, filename)) as f:
        text = _normalize(f.read())
    tokens = text.split()
    i = 0
    n = 1
    while (i-step)+window < len(tokens):
        yield (f'{filename}:{n}', tokens[i:i+window])
        i+=step
        n+=1

def docs(path, window=300, step=150):
    """
    Read the corpus at a given 'path' and return an iterator over
    documents with up to 'window' tokens each. Large textfiles in
    'path' will be split into smaller documents of 'window' size.
    The resulting iterator yields tuples, of the following format:
    (doc_id, n_tokens, n_chars, content).
    """
    for filename in os.listdir(path):
        for doc_id, tokens in _split_doc(path, filename, window, step):
            n_tokens = len(tokens)
            content = " ".join(tokens)
            n_chars = len(content)
            yield (doc_id, n_tokens, n_chars, content)

def psq_pairs(path, window=300, step=150):
    """
    Read the corpus at a given 'path' and return an iterator over
    documents that follow each other. This data is used for evaluating
    models. The resulting iterator yields tuples of (a, b) where 'a'
    and 'b' are document ids where 'b' represents the page immediately
    following 'a' in the same plain text file with no overlap.
    """
    for filename in os.listdir(path):
        last_ids = []
        for doc_id, _tokens in _split_doc(path, filename, window, step):
            last_ids.append(doc_id)
            if len(last_ids) > window // step:
                yield (last_ids.pop(0), doc_id)

_cli_help="""
Usage: ttm [OPT]... cat [COMMAND-OPTION]... TYPE DIR

Positional Arguments
    TYPE    Either 'docs' or 'psq-pairs'. If TYPE is 'docs', a tsv file
            containing the document id and content in respective columns
            is produced. All whitespace is normalized to a single ASCII
            space and hyphenated words (i. e. words containing '-\\n')
            are joined.
            If TYPE is 'psq-pairs', the document ids of consecutive pages
            are provided as a tsv file with no header row, one document
            id per column, and one pair of adjacent pages per row.
    DIR     Directory containing the corpus. The corpus may be split
            across multiple plain text files. The directory is assumed
            to contain nothing but those plain text files.

Command Options
    -w N, --window N
            Split text files in DIR into documents of up to N tokens.
            Default: 300.
    -s N, --step N
            Slide the window forward N tokens for a new window. This allows
            to create overlapping document representations. Overlapping
            documents are never returned as sequential pages. In order for
            this to work, the window needs to be divisible by step, so that
            some number of steps will add up to a window. Default: Half the
            specified window size.
    -h, --help
            Print this help message and exit.
""".lstrip()

def _cli(argv, infile, outfile):
    opts, args = getopt(argv, 'hw:s:', ['help', 'window=', 'step='])
    short2long = { '-h': '--help', '-w': '--window', '-s': '--step' }
    opts = { short2long.get(k, k).lstrip('-'): v for k, v in opts }
    if 'help' in opts:
        raise HelpRequested(_cli_help)
    elif len(args) != 2:
        argstring = ' '.join(args)
        raise CliError('Unable to parse positional arguments to ttm cat: '\
                      f"expected 'TYPE DIR' but got '{argstring}'")
    elif not os.path.isdir(args[1]):
        raise CliError(f"Directory '{args[1]}' not found")
    window = int(opts.get('window', 300))
    if 'step' in opts:
        step = int(opts.get('step'))
    elif window % 2 == 0:
        step = window // 2
    else:
        raise CliError('The step was not specified and should default to '
                      f'half the window size,\nbut {window} is not an even '
                       'number.')
    if window % step != 0:
        w, s = window, step
        low_w, high_w = (w//s)*s, ((w//s)+1)*s
        sug_w = low_w if w - low_w < high_w - w else high_w
        low_s, high_s = s, s
        while w % low_s != 0 and w % high_s != 0 \
                    and low_s > s - .2*w and high_s < s + .2*w:
            low_s, high_s = low_s - 1, high_s - 1
        if w % high_s == 0:
            or_step = f' or a step of {high_s}'
        elif w % low_s == 0:
            or_step = f' or a step of {low_s}'
        else:
            or_step = ''
        raise CliError(f'Window must be divisible by step, but {w} % {s} '
                       f'is {w%s}. You may want to\nconsider a window of '
                       f'{sug_w}{or_step}.')
    if args[0] == 'docs':
        print('id', 'n_tokens', 'n_chars', 'content', sep='\t', file=outfile)
        for doc_id, n_tokens, n_chars, content in docs(args[1],
                                                window=window, step=step):
            if '\t' in doc_id or '\n' in doc_id:
                raise CliError(f"Document id '{doc_id}' contains an "\
                                'invalid character (tab or newline)')
            print(doc_id, n_tokens, n_chars, content, sep='\t', file=outfile)
    elif args[0] == 'psq-pairs':
        for a, b in psq_pairs(args[1], window=window, step=step):
            if '\t' in a or '\n' in a:
                raise CliError(f"Document id '{a}' contains an invalid "\
                                'character (tab or newline)')
            if '\t' in b or '\n' in b:
                raise CliError(f"Document id '{b}' contains an invalid "\
                                'character (tab or newline)')
            print(a, b, sep='\t', file=outfile)
    else:
        raise CliError("TYPE must be either 'docs' or 'psq-pairs', not "\
                      f"'{args[0]}'")
