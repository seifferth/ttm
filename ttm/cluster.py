#!/usr/bin/env python3

from getopt import getopt
from common_types import *
import sys, json

def argmax(vectors):
    from numpy import argmax
    return [ int(argmax(v)) for v in vectors ]

def aggl(vectors, clusters=10, affinity='euclidean', linkage='ward'):
    from sklearn.cluster import AgglomerativeClustering
    return AgglomerativeClustering(
            n_clusters = clusters,
            affinity = affinity,
            linkage = linkage,
        ).fit_predict(list(vectors)).tolist()

def kmeans(vectors, clusters=10, init='k-means++'):
    from sklearn.cluster import KMeans
    return KMeans(
            n_clusters = clusters,
            init = init,
            algorithm = 'full',
        ).fit_predict(list(vectors)).tolist()

def hdbscan(vectors, metric='euclidean', cluster_selection_method='eom',
                     min_cluster_size=15):
    from hdbscan import HDBSCAN
    return HDBSCAN(
        metric = 'euclidean',
        cluster_selection_method = 'eom',
        min_cluster_size = min_cluster_size,
    ).fit(list(vectors)).labels_.tolist()

_cli_help="""
Usage: ttm [OPT]... cluster [--help] METHOD [ARGS]...

Methods
    argmax      Use the argmax on 'lowdim' to assign a single topic to
                each document. Albeit rather primitive, this method is
                often used in combination with 'lda' for dimensionality
                reduction and 'bow' for document embeddings to perform
                topic modelling. 'argmax' takes no further arguments.
    aggl        Use simple agglomerative clustering, relying on the
                sklearn implementation.
    kmeans      Use the sklearn implementation of the well-known k-means
                Algorithm to cluster documents.
    hdbscan     Use McInnes' et al.'s HDBSCAN implementation to cluster
                documents. This is the clustering algorithm used in
                both Dimo Angelov's Top2Vec and Maarten Grootendorst's
                BERTopic. The default parameters are the ones used
                in Top2Vec. Maarten Grootendorst uses almost the same
                parameters as Angelov, but sets --min-cluster-size to
                10 rather than 15.

Arguments for aggl
    --clusters N        Number of clusters to produce
    --affinity METRIC   Use METRIC as a distance measure between points.
                        Can be any one of 'euclidean', 'l1', 'l2',
                        'manhattan', or 'cosine'. Default: 'euclidean'.
    --linkage LINKAGE   Which criterion to use for linkage. Can be any
                        one of 'ward', 'complete', 'average', or 'single'.
                        Default: 'ward'. Note that 'ward' linkage only
                        works with the 'euclidean' affinity metric.
                        For more information about 'affinity' and 'linkage'
                        see 'pydoc sklearn.cluster.AgglomerativeClustering'.

Arguments for kmeans
    --clusters N        Number of clusters to produce.
    --init METHOD       Initialization method for k-means. Can be either
                        'k-means++' or 'random'. Default: 'k-means++'. See
                        'pydoc sklearn.cluster.KMeans' for further details.

Arguments for hdbscan
    --metric METRIC
                Default: 'euclidean'. For a full list of supported metrics
                see 'pydoc sklearn.metrics.pairwise.pairwise_distances'.
    --cluster-selection-method METHOD
                METHOD can be either 'eom' or 'leaf. Default: 'eom'.
    --min-cluster-size N
                Number of minimum documents per cluster. Clusters with less
                than N documents are assigned to a non-cluster with id '-1'.
                Default: 15. For more details see 'pydoc hdbscan.HDBSCAN'.
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
            rest = '\n'.join(rest)
            raise CliError(f"Unsupported command line argument '{rest}'")
    if args[0] == 'argmax':
        fail_on_rest(args[1:])
        method, method_args = argmax, {}
    elif args[0] == 'aggl':
        aggl_opts, rest = getopt(args[1:], '', ['clusters=', 'affinity=',
                                 'linkage='])
        fail_on_rest(rest)
        aggl_opts = { k.lstrip('-'): v for k, v in aggl_opts }
        for k in ['clusters']:
            if k in aggl_opts: aggl_opts[k] = int(aggl_opts[k])
        method, method_args = aggl, aggl_opts
    elif args[0] == 'kmeans':
        kmeans_opts, rest = getopt(args[1:], '', ['clusters=', 'init='])
        fail_on_rest(rest)
        kmeans_opts = { k.lstrip('-'): v for k, v in kmeans_opts }
        for k in ['clusters']:
            if k in kmeans_opts: kmeans_opts[k] = int(kmeans_opts[k])
        method, method_args = kmeans, kmeans_opts
    elif args[0] == 'hdbscan':
        hdbscan_opts, rest = getopt(args[1:], '',
            ['metric=', 'cluster-selection-method=', 'min-cluster-size='])
        fail_on_rest(rest)
        hdbscan_opts = { k.lstrip('-').replace('-', '_'): v
                         for k, v in hdbscan_opts }
        for k in ['min_cluster_size']:
            if k in hdbscan_opts: hdbscan_opts[k] = int(hdbscan_opts[k])
        method, method_args = hdbscan, hdbscan_opts
    else:
        raise CliError(f"Unknown ttm cluster METHOD '{args[0]}'")
    # Apply clustering
    lowdim = infile.column('lowdim', map_f=json.loads)
    print(f'Clustering document vectors with {method.__name__}',
          file=sys.stderr)
    cluster = method(lowdim, **method_args)
    # Copy result into outfile
    input_lines = iter(infile)
    print(f'{next(input_lines)}\t{"cluster"}', file=outfile)
    for i, line in enumerate(input_lines):
        print(f'{line}\t{cluster[i]}', file=outfile)
