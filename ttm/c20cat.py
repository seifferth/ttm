#!/usr/bin/env python3

from getopt import gnu_getopt
from .types import *
import re

def _normalize(doc):
    doc = re.sub(r'\s+', ' ', doc)
    return doc.strip()

def docs(min_chars=1, min_tokens=1, max_chars=None, max_tokens=None):
    """
    Load the 20 newsgroups dataset via sklearn and return an iterator
    over all documents. The resulting iterator yields tuples of the
    following format: (doc_id, newsgroup, n_tokens, n_chars, content)
    """
    from sklearn.datasets import fetch_20newsgroups
    groups = fetch_20newsgroups(subset='all', shuffle=False,
                                remove=('headers','footers','quotes'))
    docs = filter(bool, ( _normalize(d) for d in groups.data ))
    labels = ( groups.target_names[i] for i in groups.target )
    msg_number = { label: 0 for label in groups.target_names }
    for content, group_label in zip(docs, labels):
        msg_number[group_label] += 1
        doc_id = f'{group_label}.message-{msg_number[group_label]:03d}'
        n_chars = len(content)
        n_tokens = len(content.split())
        if n_chars < min_chars or n_tokens < min_tokens : continue
        if (max_chars != None and n_chars > max_chars) or \
           (max_tokens != None and n_tokens > max_tokens): continue
        yield (doc_id, group_label, n_tokens, n_chars, content)

_cli_help="""
Usage: ttm [OPT]... 20cat [COMMAND-OPTION]...

Command Options
    --min-chars N
            Skip documents with less than N characters.
    --min-tokens N
            Skip documents with less than N tokens.
    --max-chars N
            Skip documents with more than N characters.
    --max-tokens N
            Skip documents with more than N tokens.
    -h, --help
            Print this help message and exit.
""".lstrip()

def _cli(argv, infile, outfile):
    opts, cmd = gnu_getopt(argv, 'h', ['help',
        'min-chars=', 'min-tokens=', 'max-chars=', 'max-tokens='])
    if cmd: raise CliError('ttm 20cat does not support any positional '\
                          f'arguments but received {len(cmd)}')
    short2long = { '-h': '--help' }
    opts = { short2long.get(k, k).lstrip('-'): v for k, v in opts }
    if 'help' in opts:
        raise HelpRequested(_cli_help)
    print('id', 'newsgroup', 'n_tokens', 'n_chars', 'content',
                                            sep='\t', file=outfile)
    for doc_id, newsgroup, n_tokens, n_chars, content \
     in docs(min_chars=int(opts.get('min-chars', 1)),
             min_tokens=int(opts.get('min-tokens', 1)),
             max_chars=int(opts.get('max-chars', 1)),
             max_tokens=int(opts.get('max-tokens', 1))):
        if '\t' in doc_id or '\n' in doc_id:
            raise CliError(f"Document id '{doc_id}' contains an invalid "\
                            'character (tab or newline)')
        print(doc_id, newsgroup, n_tokens, n_chars, content,
                                            sep='\t', file=outfile)
