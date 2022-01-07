#!/usr/bin/env python3

from getopt import getopt
from common_types import *
import os, re

def _normalize(doc):
    doc = re.sub(r'-\n', '', doc)
    doc = re.sub(r'\s+', ' ', doc)
    return doc

def _split_doc(path, filename, window):
    """
    Split the text file at 'path/filename' into documents of up to
    'window' tokens each. The resulting iterator yields tuples of
    the following format: (doc_id, tokens) where tokens is a list
    of strings.
    """
    with open(os.path.join(path, filename)) as f:
        text = _normalize(f.read())
    tokens = text.split()
    i = 0
    while i*window < len(tokens):
        yield (f'{filename}:{i+1}', tokens[i*window:(i+1)*window])
        i+=1

def docs(path, window=300):
    """
    Read the corpus at a given 'path' and return an iterator over
    documents with up to 'window' tokens each. Large textfiles in
    'path' will be split into smaller documents of 'window' size.
    The resulting iterator yields tuples, of the following format:
    (doc_id, n_tokens, n_chars, content).
    """
    for filename in os.listdir(path):
        for doc_id, tokens in _split_doc(path, filename, window):
            n_tokens = len(tokens)
            content = " ".join(tokens)
            n_chars = len(content)
            yield (doc_id, n_tokens, n_chars, content)

def psq_pairs(path, window=300):
    """
    Read the corpus at a given 'path' and return an iterator over
    documents that follow each other. This data is used for evaluating
    models. The resulting iterator yields tuples of (a, b) where 'a'
    and 'b' are document ids where 'b' represents the page immediately
    following 'a' in the same plain text file with no overlap.
    """
    for filename in os.listdir(path):
        last_id = None
        for doc_id, _tokens in _split_doc(path, filename, window):
            if last_id != None: yield (last_id, doc_id)
            last_id = doc_id

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
            Split text files in DIR into documents of up to N tokens
            (default: 300).
    -h, --help
            Print this help message and exit.
""".lstrip()

def _cli(argv, infile, outfile):
    opts, args = getopt(argv, 'hw:s:', ['help','window='])
    short2long = { '-h': '--help', '-w': '--window' }
    opts = { short2long.get(k, k).lstrip('-'): v for k, v in opts }
    if 'help' in opts:
        raise HelpRequested(_cli_help)
    elif len(args) != 2:
        argstring = ' '.join(args)
        raise CliError('Unable to parse positional arguments to ttm cat: '\
                      f"expected 'TYPE DIR' but got '{argstring}'")
    elif not os.path.isdir(args[1]):
        raise CliError(f"Directory '{args[1]}' not found")
    if 'window' in opts:
        opts['window'] = int(opts['window'])
    if args[0] == 'docs':
        print('id', 'n_tokens', 'n_chars', 'content', sep='\t', file=outfile)
        for doc_id, n_tokens, n_chars, content in docs(args[1], **opts):
            if '\t' in doc_id or '\n' in doc_id:
                raise CliError(f"Document id '{doc_id}' contains an "\
                                'invalid character (tab or newline)')
            print(doc_id, n_tokens, n_chars, content, sep='\t', file=outfile)
    elif args[0] == 'psq-pairs':
        for a, b in psq_pairs(args[1], **opts):
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
