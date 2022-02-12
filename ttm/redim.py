#!/usr/bin/env python3

from getopt import getopt
from .types import *
import sys, json

def id(vectors):
    return list(vectors)

def umap(vectors, components=5, neighbors=15, metric='cosine',
                  min_dist=.1):
    from umap import UMAP
    model = UMAP(
        n_neighbors = neighbors,
        n_components = components,
        metric = metric,
        min_dist = min_dist,
    ).fit(list(vectors))
    return model.embedding_.tolist()

def lda(vectors, components=5, max_epochs=10):
    from sklearn.decomposition import LatentDirichletAllocation
    return LatentDirichletAllocation(
            max_iter=max_epochs,
            n_components=components,
        ).fit_transform(list(vectors)).tolist()

def svd(vectors, components=5):
    from sklearn.decomposition import TruncatedSVD
    return TruncatedSVD(
            n_components = 5,
            algorithm = 'arpack',   # Should produce deterministic results
        ).fit_transform(list(vectors)).tolist()

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
        svd_opts, rest = getopt(args[1:], '', ['components='])
        fail_on_rest(rest)
        svd_opts = { k.lstrip('-'): int(v) for k, v in svd_opts }
        method, method_args = svd, svd_opts
    elif args[0] == 'lda':
        lda_opts, rest = getopt(args[1:], '', ['components=',
                                'max-epochs='])
        fail_on_rest(rest)
        lda_opts = { k.lstrip('-').replace('-', '_'): int(v)
                     for k, v in lda_opts }
        method, method_args = lda, lda_opts
    elif args[0] == 'umap':
        umap_opts, rest = getopt(args[1:], '', ['components=',
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
    print(f'Applying {method.__name__} for dimensionality reduction',
          file=sys.stderr)
    lowdim = method(highdim, **method_args)
    # Copy result into outfile
    input_lines = iter(infile.strip('lowdim'))
    print(f'{next(input_lines)}\t{"lowdim"}', file=outfile)
    for i, line in enumerate(input_lines):
        print(f'{line}\t{json.dumps(lowdim[i])}', file=outfile)
