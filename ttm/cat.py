#!/usr/bin/env python3

from getopt import gnu_getopt
from .types import *
import os, re

def _normalize(doc):
    doc = re.sub(r'-\n', '', doc)
    doc = re.sub(r'\s+', ' ', doc)
    return doc

def _read_files(*paths):
    for p in paths:
        with open(p) as f:
            filename = os.path.basename(p)
            text = _normalize(f.read())
            yield (filename, None, text)

def _split_doc(filename, text, window, step):
    """
    Split the text into documents of up to 'window' tokens each.
    The resulting iterator yields tuples of the following format:
    (doc_id, tokens) where tokens is a list of strings.
    """
    if window % step != 0:
        raise Exception('window must be divisible by step')
    tokens = text.split()
    i = 0
    n = 1
    while (i-step)+window < len(tokens):
        yield (f'{filename}:{n}', tokens[i:i+window])
        i+=step
        n+=1

def docs(files, window=300, step=150):
    """
    Read the corpus from given files and return an iterator over
    documents with up to 'window' tokens each. Large textfiles will
    be split into smaller documents of 'window' size.  The resulting
    iterator yields tuples of (doc_id, n_tokens, n_chars, content).
    """
    for filename, label, text in files:
        for doc_id, tokens in _split_doc(filename, text, window, step):
            n_tokens = len(tokens)
            content = " ".join(tokens)
            n_chars = len(content)
            yield (doc_id, label, n_tokens, n_chars, content)

def psq_pairs(files, window=300, step=150):
    """
    Read the corpus from given files and return an iterator over
    documents that follow each other. This data is used for evaluating
    models. The resulting iterator yields tuples of (a, b) where 'a'
    and 'b' are document ids where 'b' represents the page immediately
    following 'a' in the same plain text file with no overlap.
    """
    for filename, _label, text in files:
        last_ids = []
        for doc_id, _tokens in _split_doc(filename, text, window, step):
            last_ids.append(doc_id)
            if len(last_ids) > window // step:
                yield (last_ids.pop(0), doc_id)

def calculate_window(window, step):
    window = 300 if window == None else int(window)
    step = window if step == None else int(step)
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
    return window, step

_cli_help="""
Usage: ttm [OPT]... cat [COMMAND-OPTION]... FILE...

Print a tsv file containing the document id and content in respective
columns. All whitespace is normalized to a single ASCII space and
hyphenated words (i. e. words containing '-\\n') are joined. Note that
the file names (excluding any directories leading up to the file) must
be unique. Also note that filenames must not contain tab or newline
characters.

Command Options
    -w N, --window N
            Split text files in DIR into documents of up to N tokens.
            Default: 300.
    -s N, --step N
            Slide the window forward N tokens for a new window. This allows
            to create overlapping document representations. Overlapping
            documents are never returned as sequential pages when using the
            '--psq-pairs' option. In order for this to work, the window
            needs to be divisible by step, so that some number of steps
            will add up to a window. Default: The same as the specified
            window size (i. e. non-overlapping windows).
    --psq-pairs
            Rather than printing the documents, print a list of pages
            following one another. This list is a tsv file with no header
            row. The file contains one id per cell and one pair of ids
            of adjacent pages per row.
    -h, --help
            Print this help message and exit.
""".lstrip()

def _cli(argv, infile, outfile):
    opts, args = gnu_getopt(argv, 'hw:s:', ['help', 'window=', 'step=',
                                            'psq-pairs'])
    short2long = { '-h': '--help', '-w': '--window', '-s': '--step' }
    opts = { short2long.get(k, k).lstrip('-'): v for k, v in opts }
    if 'help' in opts:
        raise HelpRequested(_cli_help)
    elif len(args) < 1:
        raise CliError('At least one FILE must be specified for ttm cat')
    window, step = calculate_window(opts.get('window', None),
                                    opts.get('step', None))
    name2path = dict()
    for path in args:
        name = os.path.basename(path)
        if '\t' in name or '\n' in name:
            raise CliError(f"Document id '{name}' for file '{path}' "
                            'contains an invalid character (tab or newline)')
        elif name in name2path.keys():
            raise CliError(f"Duplicate document id '{name}' for "
                           f"'{name2path[name]}' and '{path}'")
        elif not os.path.isfile(path):
            raise CliError(f"File '{path}' does not exist or is not "
                            'a regular file')
        else:
            name2path[name] = path
    del name2path
    if 'psq-pairs' in opts:
        for a, b in psq_pairs(_read_files(*args), window=window, step=step):
            print(a, b, sep='\t', file=outfile)
    else:
        print('id', 'n_tokens', 'n_chars', 'content', sep='\t', file=outfile)
        for doc_id, _label, n_tokens, n_chars, content \
         in docs(_read_files(*args), window=window, step=step):
            print(doc_id, n_tokens, n_chars, content, sep='\t', file=outfile)
