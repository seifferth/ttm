#!/usr/bin/env python3

from getopt import getopt, gnu_getopt
from .types import *
import sys, json

def id(vectors):
    return vectors

def umap(vectors, components=5, neighbors=15, metric='cosine',
                  min_dist=.1):
    from umap import UMAP
    print('Applying UMAP ...', end='', file=sys.stderr, flush=True)
    result = UMAP(
        n_neighbors = neighbors,
        n_components = components,
        metric = metric,
        min_dist = min_dist,
    ).fit(vectors.matrix())
    print(' done', end='\n', file=sys.stderr, flush=True)
    return result.embedding_.tolist()

def lda(vectors, components=5, max_epochs=10, shift=False):
    from sklearn.decomposition import LatentDirichletAllocation
    import numpy as np
    matrix = vectors.matrix()
    if shift:
        rows, cols = matrix.shape
        for c in range(cols):
            colmin = matrix[:,c].min()
            if colmin >= 0:
                print(f'\rSkipping column {c+1}/{cols} ...',
                      end='', file=sys.stderr, flush=True)
            else:
                print(f'\rShifting column {c+1}/{cols} ...',
                      end='', file=sys.stderr, flush=True)
                for r in range(rows):
                    matrix[r,c] -= colmin
        print(' done', end='\n', file=sys.stderr, flush=True)
    print('Applying LDA ...', end='', file=sys.stderr, flush=True)
    result = LatentDirichletAllocation(
            max_iter=max_epochs,
            n_components=components,
        ).fit_transform(matrix)
    print(' done', end='\n', file=sys.stderr, flush=True)
    return result.tolist()

def svd(vectors, components=5):
    from sklearn.decomposition import TruncatedSVD
    print('Applying SVD ...', end='', file=sys.stderr, flush=True)
    result = TruncatedSVD(
            n_components = 5,
            algorithm = 'arpack',   # Should produce deterministic results
        ).fit_transform(vectors.matrix(dtype=float))
    print(' done', end='\n', file=sys.stderr, flush=True)
    return result.tolist()

_cli_help="""
Usage: ttm [OPT]... redim [--help] METHOD [ARG]...

Methods
    id          Use the identity function to map 'highdim' vectors to
                'lowdim'. This will make verbatim copies of the vectors
                and provides a convenient way to disable dimensionality
                reduction. 'id' takes no further arguments.
    svd         Use the sklearn implementation of Truncated Singular Value
                Decomposition.
    lda         Use the sklearn implementation of Blei et al.'s well known
                Latent Dirichlet Allocation.
    umap        Use McInnes' and Healy's Uniform Manifold Approximation
                and Projection Algorithm (and python implementation)
                for dimensionality reduction. This is the same method as
                used in Dimo Angelov's Top2Vec and Maarten Grootendorst's
                BERTopic. The default arguments are the ones used by Dimo
                Angelov. Maarten Grootendorst's BERTopic uses almost the
                same values but sets --min-dist to 0.

Arguments for 'svd'
    --components N      Number of dimensions in the 'lowdim' vector.
                        Default: 5.

Arguments for 'lda'
    --components N      Number of dimensions in the 'lowdim' vector.
                        Default: 5.
    --max-epochs N      Maximum number of training epochs. Default: 10.
    --shift             LDA only works for inputs that do not contain
                        negative entries. Since many embedding methods
                        produce negative values in some vectors, such
                        vectors need to be adjusted to serve as input
                        to LDA. If --shift is specified, every column in
                        the input data that contains negative numbers is
                        shifted upwards, so that the lowest entry has a
                        value of zero. All entries in any given column
                        are shifted upwards by the same amount.

Arguments for 'umap'
    --components N      Number of dimensions in the 'lowdim' vector.
                        Default: 5.
    --neighbors N       Local neighborhood size used in manifold
                        approximation. The UMAP documentation suggests
                        using a value between 2 and 100. For further
                        elaboration see 'pydoc umap.UMAP'. Default: 15.
    --metric METRIC     A number of metrics are supported, such as
                        'euclidean', 'manhattan' or 'cosine'. For a full
                        list of supported metrics see 'pydoc umap.UMAP'.
                        Default: 'cosine'.
    --min-dist N        Minimum distance between points in 'lowdim'. See
                        'pydoc umap.UMAP' for more details. Default: 0.1.
""".lstrip()

def _cli(argv, infile, outfile):
    opts, args = getopt(argv, 'h', ['help'])
    short2long = { '-h': '--help' }
    opts = { short2long.get(k, k).lstrip('-'): v for k, v in opts }
    if 'help' in opts:
        raise HelpRequested(_cli_help)
    elif len(args) == 0:
        raise CliError('No METHOD specified for ttm redim')
    def fail_on_rest(rest):
        if rest:
            rest = ' '.join(rest)
            raise CliError(f"Unsupported command line argument '{rest}'")
    if args[0] == 'id':
        fail_on_rest(args[1:])
        method, method_args = id, {}
    elif args[0] == 'svd':
        svd_opts, rest = gnu_getopt(args[1:], '', ['components='])
        fail_on_rest(rest)
        svd_opts = { k.lstrip('-'): int(v) for k, v in svd_opts }
        method, method_args = svd, svd_opts
    elif args[0] == 'lda':
        lda_opts, rest = gnu_getopt(args[1:], '', ['components=',
                                    'max-epochs=', 'shift'])
        fail_on_rest(rest)
        lda_opts = { k.lstrip('-').replace('-', '_'): v
                     for k, v in lda_opts }
        for k in ['components', 'max_epochs']:
            if k in lda_opts: lda_opts[k] = int(v)
        for k in ['shift']:
            if k in lda_opts: lda_opts[k] = True
        method, method_args = lda, lda_opts
    elif args[0] == 'umap':
        umap_opts, rest = gnu_getopt(args[1:], '', ['components=',
                                     'neighbors=', 'metric=', 'min-dist='])
        fail_on_rest(rest)
        umap_opts = { k.lstrip('-').replace('-', '_'): v
                      for k, v in umap_opts }
        for k in ['components', 'neighbors']:
            if k in umap_opts: umap_opts[k] = int(umap_opts[k])
        for k in ['min_dist']:
            if k in umap_opts: umap_opts[k] = float(umap_opts[k])
        method, method_args = umap, umap_opts
    else:
        raise CliError(f"Unknown ttm redim METHOD '{args[0]}'")
    # Apply dimensionality reduction
    highdim = infile.column('highdim', map_f=json.loads)
    lowdim = method(highdim, **method_args)
    # Copy result into outfile
    infile.ensure_loaded()
    input_lines = iter(infile.strip('lowdim'))
    print(f'{next(input_lines)}\t{"lowdim"}', file=outfile)
    for line, v in zip(input_lines, lowdim):
        print(f'{line}\t{json.dumps(v)}', file=outfile)
