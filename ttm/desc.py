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

def pure_docs(docs, topics, limit=5, cutoff=.5):
    tcount = { t: 0 for t in topics }
    doc_ts = { doc_id.rsplit(':', 1)[0]: tcount.copy() for doc_id in docs }
    for d, t in zip(docs, topics):
        d = d.rsplit(':', 1)[0]
        doc_ts[d][t] += 1
    for d in doc_ts:
        total = sum(doc_ts[d].values())
        for t in doc_ts[d]:
            doc_ts[d][t] /= total
    ts = { t: [] for t in tcount }
    for t in ts:
        for d in doc_ts:
            ts[t].append((doc_ts[d][t], d))
        ts[t].sort(reverse=True)
        if limit != None and limit > 0: ts[t] = ts[t][:limit]
        ts[t] = [ f'{d[1]} ({100*d[0]:.2f} %)' for d in ts[t]
                  if cutoff != None and d[0] >= cutoff ]
    return ts

_cli_help="""
Usage: ttm [OPT]... desc [--help] METHOD [ARGS]...

Methods
    tfidf       Use tfidf on document clusters to find the most important
                terms per cluster.
    pure-docs   Find documents that have a large part of their pages
                assigned to a given topic.

Arguments for tfidf
    --limit N       Include only the N most significant words for each
                    cluster. Default: 10.

Arguments for pure-docs
    --limit N       Include only the N purest docs for each cluster.
                    Default: 5.
    --cutoff N      Include only docs with at least the N'th of their
                    pages assigned to the given topic. Default: 0.5.
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
        docs = infile.column('content')
    elif args[0] == 'pure-docs':
        pure_docs_opts, rest = getopt(args[1:], '', ['limit=', 'cutoff='])
        fail_on_rest(rest)
        pure_docs_opts = { k.lstrip('-'): v for k, v in pure_docs_opts }
        for k in ['limit']:
            if k in pure_docs_opts:
                pure_docs_opts[k] = int(pure_docs_opts[k])
        for k in ['cutoff']:
            if k in pure_docs_opts:
                pure_docs_opts[k] = float(pure_docs_opts[k])
        method, method_args = pure_docs, pure_docs_opts
        docs = infile.column('id')
    # Create topic descriptions
    topics = infile.column('cluster')
    topic_desc = method(docs, topics, **method_args)
    # Copy result into outfile
    input_lines = iter(infile.strip('desc'))
    print(f'{next(input_lines)}\t{"desc"}', file=outfile)
    for line, cluster in zip(input_lines, topics):
        print(f'{line}\t{", ".join(topic_desc[cluster])}', file=outfile)
