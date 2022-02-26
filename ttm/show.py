#!/usr/bin/env python3

from getopt import getopt
from .types import *
from .eval import cluster_distribution

def book(infile: InputFile, cluster_order: list, book: str, res=15) -> str:
    infile.ensure_loaded()
    page_clusters = dict()
    for doc, c in zip(infile.column('id'), infile.column('cluster')):
        b, p = doc.split(':'); p=int(p)
        if b != book: continue
        page_clusters[p] = c
    cluster_freq = { c: 0 for c in set(page_clusters.values()) }
    for p, c in page_clusters.items():
        cluster_freq[c] += 1
    header = f'  {"":>9}   ' + \
                ''.join([f'{c[:2]:>2} ' for c in cluster_order])
    result = [ header ]
    pages = sorted(page_clusters.keys())
    while pages:
        fst, lst = None, None
        clusters_found = { c: 0 for c in cluster_order }
        for i in range(min(len(pages), res)):
            p = pages.pop(0)
            if fst == None: fst = p
            clusters_found[page_clusters[p]] += 1
        lst = p
        line = [ f'  {str(fst)+"-"+str(lst):>9}  |' ]
        for c in cluster_order:
            if clusters_found[c]/res == 0:     line.append('   ')
            elif clusters_found[c]/res <= 1/6: line.append(' - ')
            elif clusters_found[c]/res <= 2/6: line.append(' + ')
            elif clusters_found[c]/res <= 3/6: line.append('-+ ')
            elif clusters_found[c]/res <= 4/6: line.append('-+-')
            elif clusters_found[c]/res <= 5/6: line.append('++-')
            else:                              line.append('+++')
        result.append(''.join(line)+'|')
    result.append(header)
    return '\n'.join(result)

_cli_help="""
Usage: ttm [OPT]... show SUBCOMMAND [SUBCOMMAND-ARGUMENT]...

Subcommands
    book [--res=N] REGEX...
        Display an ascii-art rendering of the clusters found in one or more
        books. The REGEX is used to determine which books are displayed. The
        --res parameter specifies how many pages are summarized in any given
        line.
""".lstrip()

def _book(argv, infile):
    from getopt import gnu_getopt
    import re
    from textwrap import fill, indent
    opts, bookexp = gnu_getopt(argv, '', ['res='])
    opts = { k.lstrip('-'): int(v) for k, v in opts }
    if len(bookexp) == 0:
        raise CliError("No REGEX specified for 'ttm show book'")
    cluster_dist = cluster_distribution(infile.column('cluster'))
    cluster_order = [ c for _, c in sorted([(f, c) for c, f
                        in cluster_dist.items()], reverse=True) ]
    all_books = { d.split(':')[0] for d in infile.column('id') }
    blocks = []
    for exp in bookexp:
        for b in sorted(all_books):
            if re.search(exp, b):
                all_books.remove(b)
                graph = book(infile, cluster_order, b,
                             res=opts.get('res', 15))
                title = indent(fill(b, width=38), '  ')
                blocks.append(f'{title}\n\n{graph}\n\n')
    full_length = sum([ len(b.splitlines()) for b in blocks ])
    blocks_length = 0; i = 0
    for _ in range(len(blocks)):
        blocks_length += len(blocks[i].splitlines()); i+=1
        if blocks_length > full_length / 2: break
    col_a = ''.join(blocks[:i]).splitlines()
    col_b = ''.join(blocks[i:]).splitlines()
    for i in range(max(map(len, [col_a, col_b]))):
        print(f'{col_a[i] if i < len(col_a) else "":<40}', end='')
        print(f'{col_b[i] if i < len(col_b) else "":<40}', end='')
        print()

def _cli(argv, infile, outfile):
    opts, subcmd = getopt(argv, 'h', ['help'])
    short2long = { '-h': '--help' }
    opts = { short2long.get(k, k).lstrip('-'): v for k, v in opts }
    if 'help' in opts:
        raise HelpRequested(_cli_help)
    elif len(subcmd) < 1:
        raise CliError('No SUBCOMMAND provided to ttm show')
    elif subcmd[0] == 'book': _book(subcmd[1:], infile)
    else:
        raise CliError(f"Unknown SUBCOMMAND for ttm show: '{subcmd[0]}'")
