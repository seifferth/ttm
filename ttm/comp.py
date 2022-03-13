#!/usr/bin/env python3

from getopt import gnu_getopt
from .types import *
from .eval import cluster_distribution, bucket_probability
from itertools import combinations

def _cluster2doc(f: InputFile, sample: set) -> dict:
    cluster2doc = { c: [] for c in set(f.column('cluster')) }
    for doc, cluster in zip(f.column('id'), f.column('cluster')):
        if doc in sample: cluster2doc[cluster].append(doc)
    return cluster2doc
def _cluster_pairs(f: InputFile, sample: set):
    for _cluster, docs in _cluster2doc(f, sample).items():
        for c in combinations(sorted(docs), 2):
            yield c
def _kappa(f: InputFile, g: InputFile, sample: set) -> tuple:
    U = list(combinations(sample, 2))
    F = set(_cluster_pairs(f, sample=sample))
    G = set(_cluster_pairs(g, sample=sample))
    #
    p_o = 1 - ( len(F.symmetric_difference(G)) / len(U) )
    p_F = bucket_probability(cluster_distribution(f.column('cluster')))
    p_G = bucket_probability(cluster_distribution(g.column('cluster')))
    p_e = p_F * p_G + (1 - p_F) * (1 - p_G)
    #
    kappa = (p_o - p_e) / (1 - p_e)
    zoom = 1 / (1 - p_e)
    return (kappa, zoom)
def _kappa(f: InputFile, g: InputFile, sample: set) -> tuple:
    """
    Memory-optimized version for large kappas (also seems to run faster)
    """
    n_ids = len(sample)
    len_U = (n_ids**2 - n_ids) / 2    # Matrix of id x id minus diagonal and
                                      # with the mirrored halfs deduplicated
    F = _cluster_pairs(f, sample=sample)
    G = _cluster_pairs(g, sample=sample)
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
def avg_kappa(*infiles: InputFile, sample_size: float=1.,
              n_samples: int=None) -> tuple:
    import random
    docs = None
    for f in infiles:
        if docs == None:
            docs, last_filename = sorted(f.column('id')), f.filename
        else:
            if sorted(f.column('id')) != docs: raise ExpectedRuntimeError(
                'Found differences in document ids between '
               f"'{last_filename}' and '{f.filename}'")
    kappas, zooms = [], []
    if n_samples == None: n_samples = round(sample_size*len(docs))
    for f, g in combinations(infiles, 2):
        k, z = _kappa(f, g, sample=set(random.sample(docs, n_samples)))
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
    --sample-size N
                The relative sample size to draw from the data when
                calculating avg-kappa. This value must lie between 0
                and 1. Default: 1.
    -h, --help
                Print this help message and exit.
""".lstrip()

def _cli(argv, infile, outfile):
    opts, filenames = gnu_getopt(argv, 'h', ['help', 'sample-size='])
    short2long = { '-h': '--help' }
    opts = { short2long.get(k, k).lstrip('-').replace('-', '_'): v
             for k, v in opts }
    if 'help' in opts:
        raise HelpRequested(_cli_help)
    if len(filenames) < 2:
        raise CliError('At least two FILE arguments are required '
                       'for ttm comp')
    for k in ['sample_size']:
        if k not in opts:
            opts[k] = 1.
        else:
            opts[k] = float(opts[k])
            if opts[k] < 0 or opts[k] > 1:
                raise CliError('--sample-size must lie between 0 and 1')
    files = [ InputFile(f) for f in filenames ]
    print('models              ', end='', flush=True)
    print(f'{len(files)}    {filenames}')
    print(f'sample-size         {opts["sample_size"]}  ', end='', flush=True)
    n_samples = round(opts['sample_size']*len(files[0].column('id')))
    print(f'({n_samples} samples)')
    print('avg-kappa           ', end='', flush=True)
    avg_k, dev_k, avg_z, dev_z = avg_kappa(*files, n_samples=n_samples)
    print(f'{avg_k:.4f} \u00B1{dev_k:.4f}  '
          f'(zoom {avg_z:.2f} \u00B1{dev_z:.2f})')
