#!/usr/bin/env python3

from getopt import gnu_getopt
from .types import *
import re

def _normalize(doc):
    doc = re.sub(r'\s+', ' ', doc)
    return doc

def docs():
    """
    Load the 20 newsgroups dataset via sklearn and return an iterator
    over all documents. The resulting iterator yields tuples of the
    following format: (doc_id, n_tokens, n_chars, content)
    """
    from sklearn.datasets import fetch_20newsgroups
    groups = fetch_20newsgroups(subset='all', shuffle=False,
                                remove=('headers','footers','quotes'))
    docs = ( _normalize(d) for d in groups.data )
    labels = ( groups.target_names[i] for i in groups.target )
    label_is = { label: 0 for label in groups.target_names }
    for content, group_label in zip(docs, labels):
        label_is[group_label] += 1
        doc_id = f'{group_label}:{label_is[group_label]}'
        n_chars = len(content)
        n_tokens = len(content.split())
        yield (doc_id, n_tokens, n_chars, content)

_cli_help="""
Usage: ttm [OPT]... 20cat [--help]

Command Options
    -h, --help      Print this help message and exit.
""".lstrip()

def _cli(argv, infile, outfile):
    opts, cmd = gnu_getopt(argv, 'h', ['help'])
    short2long = { '-h': '--help' }
    opts = { short2long.get(k, k).lstrip('-'): v for k, v in opts }
    if 'help' in opts:
        raise HelpRequested(_cli_help)
    print('id', 'n_tokens', 'n_chars', 'content', sep='\t', file=outfile)
    for doc_id, n_tokens, n_chars, content in docs():
        if '\t' in doc_id or '\n' in doc_id:
            raise CliError(f"Document id '{doc_id}' contains an invalid "\
                            'character (tab or newline)')
        print(doc_id, n_tokens, n_chars, content, sep='\t', file=outfile)
