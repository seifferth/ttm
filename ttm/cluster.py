#!/usr/bin/env python3

from getopt import getopt
from .types import *
import sys, json

# math is not used directly, but it can be convenient for the 'eval'
# function that is used with the 'random' clustering method.
import math

def argmax(vectors):
    from numpy import argmax
    return ( int(argmax(v)) for v in vectors )

def aggl(vectors, clusters=10, affinity='euclidean', linkage='ward'):
    from sklearn.cluster import AgglomerativeClustering
    return AgglomerativeClustering(
            n_clusters = clusters,
            affinity = affinity,
            linkage = linkage,
        ).fit_predict(vectors.matrix()).tolist()

def kmeans(vectors, clusters=10, init='k-means++'):
    from sklearn.cluster import KMeans
    return KMeans(
            n_clusters = clusters,
            init = init,
            algorithm = 'full',
        ).fit_predict(vectors.matrix()).tolist()

def hdbscan(vectors, metric='euclidean', cluster_selection_method='eom',
                     min_cluster_size=15):
    from hdbscan import HDBSCAN
    return HDBSCAN(
        metric = 'euclidean',
        cluster_selection_method = 'eom',
        min_cluster_size = min_cluster_size,
    ).fit(vectors.matrix()).labels_.tolist()

def random(vectors, clusters=10, weights=None, function=None):
    from random import choices
    k = 0
    for _ in vectors:
        k += 1
    if weights == None: weights = [ 1 for _ in range(clusters) ]
    if function != None:
        weights = [ w * function(i+1) for i, w in enumerate(weights) ]
    return choices(range(clusters), weights=weights, k=k)

_cli_help="""
Usage: ttm [OPT]... cluster [COMMAND-OPTION]... METHOD [ARGS]...

Command Options
    --split CLUSTER
            Rather than clustering all documents, apply the clustering
            method to split an existing CLUSTER into subclusters.
    -h, --help
            Print this help message and exit.

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
    random      Assign a random cluster id to each document. This
                intentionally produces nonsensical clusters. It is meant
                as a convenient way of getting a feel for how far the
                results produced by other means differ from random noise.

Arguments for aggl
    --clusters N        Number of clusters to produce. Default: 10.
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
    --clusters N        Number of clusters to produce. Default: 10.
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

Arguments for random
    --clusters N        Number of clusters to produce. Default: 10.
    --weights WEIGHTS   The weights to use for producing the clusters,
                        specified as a list of comma-separated floating
                        point numbers. If both --clusters and --weights
                        are specified, their numbers must match. If
                        --weights is specified but --clusters is missing,
                        the number of clusters to create will be inferred
                        from the number of weights.
    --function FUNC     Use FUNC to assign cluster sizes. This can be used
                        to specify a probability distribution that differs
                        from equally-sized clusters. FUNC is applied to
                        natural numbers starting with 1. It is evaluated
                        using python's 'eval' function. The number that FUNC
                        is applied to is available as 'x'. The python math
                        module can be accessed via that namespace (i. e. by
                        using 'math.MATH_FUNCTION', e. g. 'math.sqrt(x)').
                        If both --function and --weights are specified, the
                        weights are applied on top of the function's output.
""".lstrip()

def _cli(argv, infile, outfile):
    opts, args = getopt(argv, 'h', ['help', 'split='])
    short2long = { '-h': '--help' }
    opts = { short2long.get(k, k).lstrip('-'): v for k, v in opts }
    if 'help' in opts:
        raise HelpRequested(_cli_help)
    elif len(args) == 0:
        raise CliError('No METHOD specified for ttm cluster')
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
    elif args[0] == 'random':
        random_opts, rest = getopt(args[1:], '',
            ['clusters=', 'weights=', 'function='])
        fail_on_rest(rest)
        random_opts = { k.lstrip('-'): v for k, v in random_opts }
        for k in ['clusters']:
            if k in random_opts: random_opts[k] = int(random_opts[k])
        if 'weights' in random_opts:
            random_opts['weights'] = [ float(x.strip()) for x in
                                       random_opts['weights'].split(',') ]
            if 'clusters' not in random_opts:
                random_opts['clusters'] = len(random_opts['weights'])
            if len(random_opts['weights']) != random_opts['clusters']:
                n_weights = len(random_opts['weights'])
                n_clusters = random_opts['clusters']
                raise CliError('Expected the number of weights to match the '
                               f'number of clusters, but found {n_weights} '
                               f'weights for {n_clusters} clusters')
        if 'function' in random_opts:
            random_dist_function = random_opts['function']
            random_opts['function'] = lambda x: eval(random_dist_function)
        method, method_args = random, random_opts
    else:
        raise CliError(f"Unknown ttm cluster METHOD '{args[0]}'")
    # Apply clustering
    lowdim = infile.column('lowdim', map_f=json.loads)
    if 'split' in opts:
        split = opts['split']
        print(f"Splitting cluster '{split}' with {method.__name__}",
              file=sys.stderr)
        try:
            subclusters = iter(method(
                lowdim.filter('cluster', lambda c: c == split),
                **method_args
            ))
        except ColumnNotFound as e:
            raise CliError(f"Unable to split cluster '{split}': " +
                             str(e)) from e
        except EmptyColumnError as e:
            raise CliError(f"Cluster '{split}' does not exist") from e
        cluster = ( f'{c}.{next(subclusters)}' if c == split else c
                    for c in infile.column('cluster') )
    else:
        print(f'Clustering document vectors with {method.__name__}',
              file=sys.stderr)
        cluster = method(lowdim, **method_args)
    # Copy result into outfile
    input_lines = iter(infile.strip('cluster'))
    print(f'{next(input_lines)}\t{"cluster"}', file=outfile)
    for line, cluster_id in zip(input_lines, cluster):
        print(f'{line}\t{cluster_id}', file=outfile)
