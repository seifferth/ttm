#!/usr/bin/env python3

from getopt import getopt
from common_types import *
import json

def extract_X_y(infile: InputFile) -> tuple:
    X = list(infile.column('lowdim', map_f=json.loads))
    y = list(infile.column('cluster'))
    return (X, y)

def calinski_harabasz(X, y) -> float:
    from sklearn.metrics import calinski_harabasz_score
    return calinski_harabasz_score(X, y)

def davies_bouldin(X, y) -> float:
    from sklearn.metrics import davies_bouldin_score
    return davies_bouldin_score(X, y)

def silhouette(X, y, metric='euclidean', sample_size=.2) -> tuple:
    from sklearn.metrics import silhouette_score
    samples = round(len(X) * sample_size)
    score = silhouette_score(X, y, metric=metric, sample_size=samples)
    return (score, samples)

def cluster_distribution(cluster: Column) -> dict:
    counts = dict()
    for c in cluster:
        if c not in counts: counts[c] = 0
        counts[c] += 1
    total = sum(counts.values())
    counts = { k: v/total for k, v in
               sorted(counts.items(), key=lambda x: x[1], reverse=True) }
    return counts

def bucket_probability(cluster_distribution: dict) -> float:
    """
    Random bucket score. Given a set of specifically sized buckets, and
    assuming that pages are randomly sorted into buckets (respecting the
    bucket sizes), what is the possibility that any two pages appear in
    the same bucket.
    """
    return sum(( x**2 for x in cluster_distribution.values() ))

def psq_count(infile: InputFile, psq_pairs: PsqPairs) -> float:
    """
    Given a list of pages following one another, calculate how many of these
    pairs of pages can be found in the same cluster.
    """
    n_matches = 0
    n_pairs = 0
    infile.ensure_loaded()
    doc2cluster = { d: c for d, c in
                    zip(infile.column('id'), infile.column('cluster')) }
    for a, b in psq_pairs:
        if doc2cluster[a] == doc2cluster[b]: n_matches += 1
        n_pairs += 1
    return n_matches / n_pairs

def psq_score(psq_count, cluster_distribution) -> tuple:
    """
    The raw psq_count is highly sensitive to the number and size
    of clusters. If the majority of all pages are asigned to the
    same cluster, this leads to a higher psq_count without indicating
    better quality. The psq_score takes the number of clusters and their
    respecitve sizes into account by calculating

        ( psq_count - p_expected ) / ( 1 - p_expected )

    The expected probability is estimated from the number and size
    of clusters by taking the sum of squares of all relative cluster
    sizes.

    The psq_score is reported as 'score' and 'zoom', where 'zoom' is
    the scale adjustment only.

        zoom = 1 / ( 1 - p_expected )

    The zoom starts at 1 and has no upper bound. The value increases both
    with decreasing numbers of clusters and with increasing differences
    in the respective cluster sizes. The zoom should not be directly
    relevant to evaluation, but high values may indicate undesirable
    cluster sizes. As a rule of thumb, a zoom value above 3 may be
    considered suspicious.

    This adjustment for chance is basically the same that
    sklearn applies in sklearn.metrics.adjusted_rand_score or
    sklearn.metrics.cohen_kappa_score for instance.
    """
    psq_expected = bucket_probability(cluster_distribution)
    psq_observed = psq_count
    score = (psq_observed - psq_expected) / (1 - psq_expected)
    zoom = 1 / (1 - psq_expected)
    return (score, zoom)

_cli_help="""
Usage: ttm [OPT]... eval [COMMAND-OPTION]... FILE [FILE]...

'ttm eval' takes one or more file names as positional arguments.
These files are supposed to contain a corpus processed by ttm. On
the fly decompression works just like it does with the '-i' option.
The evaluation is done for each of the specified files.

Evaluation Metrics
    cluster-distribution
        Relative cluster size for each cluster. If --format is 'text',
        the cluster sizes are reported in percent and visualized with a
        histogram. If --format is 'tsv', the relative cluster sizes are
        included as a json-serialized dictionary.

    calinski-harabasz
        The Caliński-Harabasz Score is a metric for measuring the degree
        of separation between clusters proposed by T. Caliński and
        J. Harabasz in 1974. Higher values indicate better separation.
        See 'pydoc sklearn.metrics.calinski_harabasz_score' for further
        information.

    davies-bouldin
        The Davies-Bouldin Score is a metric for measuring the degree
        of separation between clusters proposed by D. L. Davies and
        D. W. Bouldin in 1979. Smaller values indicate better separation,
        with 0 being the lower bound for this metric. For further
        information see 'pydoc sklearn.metrics.davies_bouldin_score'.

    silhouette
        The Silhouette Coefficient is a metric for measuring the degree
        of separation between clusters proposed by P. J. Rousseeuw in
        1987. Possible values range from -1 to +1 with higher values
        indicating better separation. A value of 0 indicates that clusters
        are not separated at all. Values below 0 indicate that there is
        less cohesion within clusters than in the unclustered dataset. See
        'pydoc sklearn.metrics.silhouette_score' for further information.

    psq-count
        Given a list of pages following one another, calculate how many
        of these pairs of pages can be found in the same cluster.

    psq-score
        Psq-count adjusted for chance. Reasonable values range from 0
        to 1, with higher values indicating better clusters. The lower
        bound of this metric depends on the 'zoom' part and may grow to
        be any negative number. Values below 0 may be discarded entirely,
        however, as they indicate a clustering with worse than random
        cluster assignment. The 'zoom' is not directly relevant to
        interpreting this metric, but high zoom values indicate a large
        variation in cluster sizes. As a rule of thumb, a zoom above 3
        may be considered suspicious and should be investigated further
        by taking a closer look at the cluster-distribution.

Command Options
    -f FORMAT, --format FORMAT
                Either 'text' or 'tsv'. The 'text' format makes use of
                indentation and is more human readable than the tabular
                'tsv'. The 'tsv' format may provide a more convenient
                starting point for further analysis. Default: 'text'.
    --silhouette-metric METRIC
                Distance metric used for calculating the silhouette
                coefficient. For a full list of supported metrics see
                'pydoc sklearn.metrics.pairwise.pairwise_distances'.
                Default: 'euclidean'.
    --silhouette-sample-size N
                The relative number of samples to draw from the data when
                calculating the silhouette coefficient. Default: 0.2.
    --psq-pairs FILE
                Header-less tsv-file containing document id pairs
                representing consecutive pages. The file must contain
                exactly one pair of pages per line, separated by a
                single tab character. This input is required to calculate
                the psq-score.
    -h, --help  Print this help message and exit.
""".lstrip()

def _print_text(model_name: str, cluster_distribution: dict,
                calinski_harabasz: float, davies_bouldin: float,
                silhouette: tuple, psq_count: float, psq_score: tuple):
    print(f'Evaluation results for {model_name}')
    for cluster, quota in cluster_distribution.items():
        print(3*' ', f'{cluster:>5}    {100*quota:6.2f} %    ',
              round(quota*50)*'*')
    if calinski_harabasz == None:
        print(f'  calinski-harabasz  undefined')
    else:
        print(f'  calinski-harabasz     {calinski_harabasz:<.4f}')
    if davies_bouldin == None:
        print(f'  davies-bouldin     undefined')
    else:
        print(f'  davies-bouldin        {davies_bouldin:<.4f}')
    if silhouette == None:
        print(f'  silhouette         undefined')
    else:
        print(f'  silhouette           {silhouette[0]:>7.4f}  '
              f'({silhouette[1]} samples)')
    if psq_count == None:
        print(f'  psq-count                N/A')
        print(f'  psq-score                N/A')
    else:
        print(f'  psq-count            {psq_count:>7.4f}')
        if psq_score == None:
            print(f'  psq-score          undefined')
        else:
            print(f'  psq-score            {psq_score[0]:>7.4f}  '
                  f'(zoom {psq_score[1]:.2f})')
    print()

def _print_tsv_header():
    row = [ 'model_name', 'psq_score', 'psq_score_zoom', 'psq_count',
            'silhouette', 'silhouette_samples', 'davies_bouldin',
            'calinski_harabasz', 'cluster_distribution' ]
    print(*row, sep='\t', end='\n')
def _print_tsv(model_name: str, cluster_distribution: dict,
               calinski_harabasz: float, davies_bouldin: float,
               silhouette: tuple, psq_count: float, psq_score: tuple):
    if calinski_harabasz == None: calinski_harabasz = 'undefined'
    if davies_bouldin == None: davies_bouldin = 'undefined'
    if silhouette == None: silhouette = ('undefined', 'undefined')
    if psq_count == None:
        psq_count = 'N/A'
        psq_score = ('N/A', 'N/A')
    elif psq_score == None:
        psq_score = 'undefined'
    row = [ model_name, psq_score[0], psq_score[1], psq_count,
            silhouette[0], silhouette[1], davies_bouldin,
            calinski_harabasz, json.dumps(cluster_distribution) ]
    print(*row, sep='\t', end='\n')

def _cli(argv, infile, outfile):
    opts, filenames = getopt(argv, 'hf:', ['help', 'format=',
            'silhouette-metric=', 'silhouette-sample-size=', 'psq-pairs='])
    short2long = { '-h': '--help', '-f': '--format' }
    opts = { short2long.get(k, k).lstrip('-').replace('-', '_'): v
             for k, v in opts }
    if 'help' in opts:
        raise HelpRequested(_cli_help)
    if len(filenames) < 1:
        raise CliError('At least one FILE argument is required '
                       'for ttm eval')
    files = [ InputFile(f) for f in filenames ]
    if not 'format' in opts: opts['format'] = 'text'
    if 'psq_pairs' in opts:
        opts['psq_pairs'] = PsqPairs(opts['psq_pairs'])
    silhouette_opts = dict()
    for k, v in opts.items():
        if k.startswith('silhouette_'):
            k = k.replace('silhouette_', '')
            if k == 'sample_size': v = float(v)
            silhouette_opts[k] = v
    if opts['format'] == 'tsv': _print_tsv_header()
    for f, name in zip(files, filenames):
        result = { 'model_name': name }
        result['cluster_distribution'] = \
                            cluster_distribution(f.column('cluster'))
        X, y = extract_X_y(f)
        if len(result['cluster_distribution']) > 1:
            result['calinski_harabasz'] = calinski_harabasz(X, y)
            result['davies_bouldin'] = davies_bouldin(X, y)
            result['silhouette'] = silhouette(X, y, **silhouette_opts)
        else:
            result['calinski_harabasz'] = None
            result['davies_bouldin'] = None
            result['silhouette'] = None
        if 'psq_pairs' in opts:
            result['psq_count'] = psq_count(f, opts['psq_pairs'])
            if len(result['cluster_distribution']) > 1:
                result['psq_score'] = psq_score(result['psq_count'],
                                        result['cluster_distribution'])
            else:
                result['psq_score'] = None
        else:
            result['psq_count'] = None
            result['psq_score'] = None
        if opts['format'] == 'text':
            _print_text(**result)
        elif opts['format'] == 'tsv':
            _print_tsv(**result)
        else:
            raise CliError("Unknown FORMAT '{}'".format(opts['format']))
