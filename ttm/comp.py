#!/usr/bin/env python3

from getopt import getopt
from .types import *
from .eval import cluster_distribution, bucket_probability
from itertools import combinations

def _cluster2doc(f: InputFile) -> dict:
    cluster2doc = { c: [] for c in set(f.column('cluster')) }
    for doc, cluster in zip(f.column('id'), f.column('cluster')):
        cluster2doc[cluster].append(doc)
    return cluster2doc
def _cluster_pairs(f: InputFile):
    for _cluster, docs in _cluster2doc(f).items():
        for c in combinations(sorted(docs), 2):
            yield c
def _kappa(f: InputFile, g: InputFile) -> tuple:
    ids = list(f.column('id'))
    U = list(combinations(ids, 2))
    F = set(_cluster_pairs(f))
    G = set(_cluster_pairs(g))
    #
    p_o = 1 - ( len(F.symmetric_difference(G)) / len(U) )
    p_F = bucket_probability(cluster_distribution(f.column('cluster')))
    p_G = bucket_probability(cluster_distribution(g.column('cluster')))
    p_e = p_F * p_G + (1 - p_F) * (1 - p_G)
    #
    kappa = (p_o - p_e) / (1 - p_e)
    zoom = 1 / (1 - p_e)
    return (kappa, zoom)
def _kappa(f: InputFile, g: InputFile) -> tuple:
    """
    Memory-optimized version for large kappas (also seems to run faster)
    """
    n_ids = sum((1 for _ in f.column('id')))
    len_U = (n_ids**2 - n_ids) / 2    # Matrix of id x id minus diagonal and
                                      # with the mirrored halfs deduplicated
    F = _cluster_pairs(f)
    G = _cluster_pairs(g)
    diff_F_G = set(F)
    for pair in G:               # Memory-efficient symmetric set difference
        if pair in diff_F_G:
            diff_F_G.remove(pair)
        else:
            diff_F_G.add(pair)
    #
    p_o = 1 - ( len(diff_F_G) / len_U )
    p_F = bucket_probability(cluster_distribution(f.column('cluster')))
    p_G = bucket_probability(cluster_distribution(g.column('cluster')))
    p_e = p_F * p_G + (1 - p_F) * (1 - p_G)
    #
    kappa = (p_o - p_e) / (1 - p_e)
    zoom = 1 / (1 - p_e)
    return (kappa, zoom)
def avg_kappa(*infiles: InputFile) -> float:
    kappas, zooms = [], []
    for f, g in combinations(infiles, 2):
        k, z = _kappa(f, g)
        kappas.append(k)
        zooms.append(z)
    avg_k = sum(kappas)/len(kappas)
    dev_k = (sum( (k - avg_k)**2 for k in kappas) / len(kappas))**.5
    avg_z = sum(zooms)/len(zooms)
    dev_z = (sum( (z - avg_z)**2 for z in zooms) / len(zooms))**.5
    return (avg_k, dev_k, avg_z, dev_z)

_cli_help="""
Usage: ttm [OPT]... comp [COMMAND-OPTION]... FILE FILE...

'ttm comp' takes two or more file names as positional arguments.
These files are supposed to contain a corpus processed by ttm. On
the fly decompression works just like it does with the '-i' option.

Evaluation Metrics
    avg-kappa   Averaged overlap between any two models, reported as
                mean average with standard deviation. For all possible
                pairs of any two pages, the agreement on whether the
                pages belong together or not is counted. (I. e. if both
                models assigned both pages to the same cluster, they
                are said to agree. If both models assigned both pages
                to different clusters, they are also said to agree. If
                one model assigned both pages to the same cluster, but
                the other model assigned them to different clusters, the
                models are said to be in disagreement.) The agreement is
                adjusted for chance following Jacob Cohen's well known
                suggestion from 1960.

Command Options
    -h, --help  Print this help message and exit.
""".lstrip()

def _cli(argv, infile, outfile):
    opts, filenames = getopt(argv, 'h', ['help'])
    short2long = { '-h': '--help' }
    opts = { short2long.get(k, k).lstrip('-'): v for k, v in opts }
    if 'help' in opts:
        raise HelpRequested(_cli_help)
    if len(filenames) < 2:
        raise CliError('At least two FILE arguments are required '
                       'for ttm comp')
    files = [ InputFile(f) for f in filenames ]
    print('models              ', end='', flush=True)
    print(f'{len(files)}    {filenames}')
    print('avg-kappa           ', end='', flush=True)
    avg_k, dev_k, avg_z, dev_z = avg_kappa(*files)
    print(f'{avg_k:.4f} \u00B1{dev_k:.4f}  '
          f'(zoom {avg_z:.2f} \u00B1{dev_z:.2f})')
