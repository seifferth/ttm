#!/usr/bin/env python3

from getopt import getopt
from .types import *
from .eval import cluster_distribution

def book(infile: InputFile, cluster_order: list, book: str, res=30) -> str:
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

def clusters(clusters) -> str:
    """
    Create a markdown table showing the cluster size. The argument can
    be either a Column with cluster IDs or a dict mapping cluster ids
    to absolute page counts.
    """
    t = []
    if type(clusters) == Column:
        csize = cluster_distribution(clusters, absolute=True)
    elif type(clusters) == dict:
        csize = clusters
    else:
        raise Exception("Argument 'clusters' must be of type 'Column' or "
                       f"'dict', not '{type(clusters)}'")
    w_cluster = max(9, max([len(str(x)) for x in csize.keys()]))
    w_pages = max(8, max([len(str(x)) for x in csize.values()]))
    t.append(f'  {"Cluster".rjust(w_cluster)}   {"Pages".rjust(w_pages)}'
             f'   {"Size (%)".rjust(9)}   Histogram')
    t.append(f'  {w_cluster*"-"}   {w_pages*"-"}   {9*"-"}   {35*"-"}--')
    total = sum(csize.values())
    histscale = 35/(max(csize.values())/total)
    for c in sorted(csize, key=lambda c: csize[c], reverse=True):
        relsize = csize[c]/total
        t.append(f'  {str(c).rjust(w_cluster)}'
                 f'   {str(csize[c]).rjust(w_pages)}   {100*relsize:9.2f}'
                  '   ' + round(relsize*histscale)*'*')
    return '\n'.join(t)

def desc(infile, clusters=None) -> dict:
    cdesc = { c: None for c in clusters }
    infile.ensure_loaded()
    for cluster, tfidf_words, pure_docs in zip(
                infile.column('cluster'), infile.column('tfidf_words'),
                infile.column('pure_docs')):
        if cdesc[cluster] == None:
            cdesc[cluster] = { 'tfidf_words': tfidf_words,
                               'pure_docs': pure_docs }
        if None not in cdesc.values(): break
    return cdesc

_cli_help="""
Usage: ttm [OPT]... show SUBCOMMAND [SUBCOMMAND-ARGUMENT]...

Subcommands
    book [--res=N|--cols=N] REGEX...
        Display an ascii-art rendering of the clusters found in one or more
        books. The REGEX is used to determine which books are displayed. The
        --res parameter specifies how many pages are summarized in any given
        line. The --cols parameter can be used to specify the desired number
        of columns for displaying multiple books.
    clusters [REGEX]...
        Print a markdown table showing clusters and cluster sizes. If a REGEX
        is specified, only books matching this expression will be considered
        when counting cluster sizes. This allows to create tables describing
        parts of the corpus for manual comparison.
    desc
        Deduplicate the cluster descriptions produced with 'ttm desc' and
        print them to stdout in markdown format.
""".lstrip()

def _book_filter(infile: InputFile, bookexp: list):
    import re
    all_books = { d.split(':')[0] for d in infile.column('id') }
    for exp in bookexp:
        for b in sorted(all_books):
            if re.search(exp, b):
                all_books.remove(b)
                yield b

def _book(argv, infile):
    from getopt import gnu_getopt
    from textwrap import fill, indent
    from shutil import get_terminal_size
    opts, bookexp = gnu_getopt(argv, '', ['res=', 'cols='])
    opts = { k.lstrip('-'): int(v) for k, v in opts }
    if len(bookexp) == 0:
        raise CliError("No REGEX specified for 'ttm show book'")
    cluster_dist = cluster_distribution(infile.column('cluster'))
    cluster_order = [ c for _, c in sorted([(f, c) for c, f
                        in cluster_dist.items()], reverse=True) ]
    blocks = []
    for b in _book_filter(infile, bookexp):
        graph = book(infile, cluster_order, b, res=opts.get('res', 30))
        title = indent(fill(b, width=37), '  ')
        blocks.append(f'{title}\n\n{graph}\n\n\n')
    full_length = sum([ len(b.splitlines()) for b in blocks ])
    max_line = max((max(map(len, b.splitlines())) for b in blocks))
    col_width = max(40, max_line)
    n_cols = opts.get('cols', (get_terminal_size().columns+1) // col_width)
    cols = [ [] for _ in range(n_cols) ]
    blocks_length = 0; i = 0; n = 0
    while i < len(blocks):
        blocks_length += len(blocks[i].splitlines()); i+=1
        if blocks_length >= full_length / n_cols:
            cols[n], blocks = blocks[:i], blocks[i:]
            lastlen = len(cols[n][-1].splitlines())
            if blocks and blocks_length - lastlen/2 > full_length / n_cols:
                blocks.insert(0, cols[n].pop())
            blocks_length = 0; i = 0; n+=1
    if blocks: cols[-1].extend(blocks)      # Catch trailing block
    cols = [ ''.join(c).splitlines() for c in cols ]
    print()
    for i in range(max(map(len, cols))):
        line = []
        for c in cols:
            line.append(format(c[i] if i < len(c) else "", str(col_width)))
        print(''.join(line).rstrip())

def _clusters(argv, infile):
    from textwrap import fill, indent
    if argv:
        books = list(_book_filter(infile, argv))
        col = infile.column('cluster').filter('id',
                                        lambda x: x.split(':')[0] in books)
        if len(books) == 0:
            raise ExpectedRuntimeError(f'No matches for ' + \
                    (f"'{argv[0]}'" if len(argv) == 1 else f'any of {argv}'))
        elif len(books) == 1:
            caption = 'Overview of clusters and cluster sizes for ' + \
                      books[0]
        elif len(books) < 10:
            caption = 'Overview of clusters and cluster sizes for ' + \
                      ', '.join(books[:-1]) + ' and ' + books[-1]
        elif len(argv) == 1:
            caption = 'Overview of clusters and cluster sizes for ' + \
                     f"{len(books)} books matching '{argv[0]}'"
        else:
            caption = 'Overview of clusters and cluster sizes for ' + \
                     f'{len(books)} books that matched at least one ' + \
                     f'of the following regular expressions: {argv}'
    else:
        col = infile.column('cluster')
        caption = 'Overview of all clusters and cluster sizes'
    print('\n' + clusters(col))
    print('\n' + indent(fill(f': {caption}', width=75), '  '), end='\n\n')

def _desc(argv, infile):
    from textwrap import fill, indent
    import re
    if argv: raise CliError("'ttm show desc' takes no further arguments, "
                           f'but {len(argv)} arguments were supplied')
    csize = cluster_distribution(infile.column('cluster'), absolute=True)
    print('\n' + clusters(csize))
    print('\n  : Overview of clusters and cluster sizes', end='\n\n\n')
    cs = sorted(csize, key=lambda c: csize[c], reverse=True)
    cdesc = desc(infile, clusters=cs)
    for c in cs:
        print(f'# Cluster {c}\n## tfidf_words', end='\n\n')
        print(fill(cdesc[c]['tfidf_words'], width=78), end='\n\n')
        print('## pure_docs', end='\n\n')
        docs = re.sub(r'( \(1?[0-9]?[0-9]\.[0-9][0-9] %\)), ', r'\1\n',
                      cdesc[c]['pure_docs'])
        print(indent(docs, '- '), end='\n\n')

def _cli(argv, infile, outfile):
    opts, subcmd = getopt(argv, 'h', ['help'])
    short2long = { '-h': '--help' }
    opts = { short2long.get(k, k).lstrip('-'): v for k, v in opts }
    if 'help' in opts:
        raise HelpRequested(_cli_help)
    elif len(subcmd) < 1:
        raise CliError('No SUBCOMMAND provided to ttm show')
    elif subcmd[0] == 'book': _book(subcmd[1:], infile)
    elif subcmd[0] == 'desc': _desc(subcmd[1:], infile)
    elif subcmd[0] == 'clusters': _clusters(subcmd[1:], infile)
    else:
        raise CliError(f"Unknown SUBCOMMAND for ttm show: '{subcmd[0]}'")
