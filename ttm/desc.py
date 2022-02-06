#!/usr/bin/env python3

from getopt import getopt
from common_types import *

def tfidf_words(infile: InputFile, limit=10, min_df=.1, max_df=1.):
    docs = infile.column('content')
    topics = infile.column('cluster')
    import numpy as np
    from sklearn.feature_extraction.text import CountVectorizer
    # Count per document term frequencies
    count = CountVectorizer(min_df=min_df, max_df=max_df).fit(docs)
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

def pure_docs(infile, limit=5, cutoff=.5):
    docs = infile.column('id')
    topics = infile.column('cluster')
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
Usage: ttm [OPT]... desc [OPTION]...

Description Methods
    tfidf-words
        Use tfidf on document clusters to find the most important terms
        per cluster.

    pure-docs
        Find documents that have a large part of their pages assigned
        to a given topic.

Command Options
    --tfidf-words-limit N
                Include only the N most significant words for each cluster.
                Default: 10.
    --tfidf-words-min-df N
                Only consider words that appear on at least N pages. N
                is a percentage of pages and must lie between 0 and 1.
                Default: 0.1.
    --tfidf-words-max-df N
                Only consider words that appear up to N pages. N is
                a percentage of pages and must lie between 0 and 1.
                Default: 1.
    --pure-docs-limit N
                Include only the N purest docs for each cluster. Default: 5.
    --pure-docs-cutoff N
                Only include docs with at least N of their pages assigned
                to a given topic. N is a percentage of pages and must lie
                between 0 and 1. Default: 0.5.
""".lstrip()

def _cli(argv, infile, outfile):
    opts, rest = getopt(argv, 'h', ['help', 'tfidf-words-limit=',
                        'tfidf-words-min-df=', 'tfidf-words-max-df=',
                        'pure-docs-limit=', 'pure-docs-cutoff='])
    short2long = { '-h': '--help' }
    opts = { short2long.get(k, k).lstrip('-').replace('-', '_'): v
             for k, v in opts }
    if 'help' in opts:
        raise HelpRequested(_cli_help)
    elif len(rest) > 0:
        rest = ' '.join(rest)
        raise CliError(f"Unsupported command line argument '{rest}'")
    tfidf_words_opts = dict()
    for k, v in opts.items():
        if k.startswith('tfidf_words_'):
            k = k.replace('tfidf_words_', '')
            if k == 'limit': v = int(v)
            if k == 'min_df': v = float(v)
            if k == 'max_df': v = float(v)
            tfidf_words_opts[k] = v
    pure_docs_opts = dict()
    for k, v in opts.items():
        if k.startswith('pure_docs_'):
            k = k.replace('pure_docs_', '')
            if k == 'limit': v = int(v)
            if k == 'cutoff': v = float(v)
            pure_docs_opts[k] = v
    desc = dict()
    desc['tfidf_words'] = tfidf_words(infile, **tfidf_words_opts)
    desc['pure_docs'] = pure_docs(infile, **pure_docs_opts)
    input_lines = iter(infile)
    desc_headers = '\t'.join(desc.keys())
    print(f'{next(input_lines)}\t{desc_headers}', file=outfile)
    for line, cluster in zip(input_lines, infile.column('cluster')):
        topic_descs = [ ', '.join(desc[method][cluster]) for method in desc ]
        topic_descs = '\t'.join(topic_descs)
        print(f'{line}\t{topic_descs}', file=outfile)
