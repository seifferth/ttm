#!/usr/bin/env python3

from getopt import getopt
from common_types import *

def tfidf(docs, topics, limit=10):
    import numpy as np
    from sklearn.feature_extraction.text import CountVectorizer
    # Count per document term frequencies
    count = CountVectorizer().fit(docs)
    docs = count.transform(docs)
    # Join per document term frequencies into per topic term frequencies
    topic2id = { t: i for i, t in enumerate(set(topics)) }
    tf = np.zeros((len(topic2id.keys()), docs.get_shape()[1]), int)
    for d, t in zip(docs, topics):
        tf[topic2id[t]] += d.toarray()[0]
    # Normalize term frequencies by document length
    tf = (tf.T / tf.sum(axis=1)).T
    # Calculate logarithmically scaled idf
    idf = np.log( len(tf) / np.where(tf > 0, 1, 0).sum(axis=0) )
    tfidf = tf * idf
    # Extract most significant terms per topic from topic-tfidf-matrix
    result = { t: [] for t in topic2id.keys() }
    for t, i in topic2id.items():
        maxwords = tfidf[i].argsort()[::-1]
        for i in maxwords[:limit]:
            doc_v = np.zeros(len(tf[0]))
            doc_v[i] = 1
            token = count.inverse_transform([doc_v])[0][0]
            result[t].append(token)
    return result

_cli_help="""
Usage: ttm [OPT]... desc [--help] METHOD [ARGS]...

Methods
    tfidf       Use tfidf on document clusters to find the most important
                terms per cluster.

Arguments for tfidf
    --limit N       Include only the N most significant words for each
                    cluster. Default: 10.
""".lstrip()

def _cli(argv, infile, outfile):
    opts, args = getopt(argv, 'h', ['help'])
    short2long = { '-h': '--help' }
    opts = { short2long.get(k, k).lstrip('-'): v for k, v in opts }
    if 'help' in opts:
        raise HelpRequested(_cli_help)
    elif len(args) == 0:
        raise CliError('No METHOD specified for ttm desc')
    def fail_on_rest(rest):
        if rest:
            rest = '\n'.join(rest)
            raise CliError(f"Unsupported command line argument '{rest}'")
    if args[0] == 'tfidf':
        tfidf_opts, rest = getopt(args[1:], '', ['limit='])
        fail_on_rest(rest)
        tfidf_opts = { k.lstrip('-'): int(v) for k, v in tfidf_opts }
        method, method_args = tfidf, tfidf_opts
    # Create topic descriptions
    docs = infile.column('content')
    topics = infile.column('cluster')
    topic_desc = method(docs, topics, **method_args)
    # Copy result into outfile
    input_lines = iter(infile)
    print(f'{next(input_lines)}\t{"desc"}', file=outfile)
    for line, cluster in zip(input_lines, topics):
        print(f'{line}\t{", ".join(topic_desc[cluster])}', file=outfile)
