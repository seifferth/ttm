#!/usr/bin/env python3

from getopt import getopt
from common_types import *
import sys, json

def bow(docs, min_df=.2, max_df=.5):
    from sklearn.feature_extraction.text import CountVectorizer
    vs = CountVectorizer(min_df=min_df, max_df=max_df).fit_transform(docs)
    return [ v.tolist() for v in vs.toarray() ]

def tfidf(docs, min_df=.2, max_df=.5):
    from sklearn.feature_extraction.text import TfidfVectorizer
    vs = TfidfVectorizer(min_df=min_df, max_df=max_df).fit_transform(docs)
    return [ v.tolist() for v in vs.toarray() ]

def doc2vec(docs, vector_size=300, min_count=50, window=15, sample=1e-5,
            negative=0, hs=1, epochs=40, dm=0, dbow_words=1, store_model=None):
    from gensim.models.doc2vec import Doc2Vec, TaggedDocument
    from gensim.utils import simple_preprocess
    import logging
    logging.basicConfig(format='[doc2vec] %(levelname)s: %(message)s',
                        level=logging.INFO)
    class PreprocessedDocs():
        def __init__(self, docs):
            self.docs = docs
        def __iter__(self):
            for i, doc in enumerate(self.docs):
                doc = simple_preprocess(doc, deacc=True)
                yield TaggedDocument(doc, [i])
    model = Doc2Vec(documents = PreprocessedDocs(docs),
                    vector_size = vector_size,
                    min_count = min_count,
                    window = window,
                    sample = sample,
                    negative = negative,
                    hs = hs,
                    epochs = epochs,
                    dm = dm,
                    dbow_words = dbow_words)
    if store_model != None: model.save(store_model)
    del logging
    vs = [ model.dv[i].tolist() for i, _ in enumerate(docs) ]
    return vs

def bert(docs, model='all-MiniLM-L6-v2'):
    """
    Produce BERT sentence transformer embeddings using the same default
    configuration as Maarten Grootendorst does with BERTopic.
    """
    from sentence_transformers import SentenceTransformer
    print(f"[bert] Loading model '{model}'", file=sys.stderr)
    embedding_model = SentenceTransformer(model)
    dociter = iter(docs)
    embeddings = []
    batch = 0
    while True:
        batch += 1
        docs = []
        for i in range(1000):
            try:
                docs.append(next(dociter))
            except StopIteration:
                break
        if len(docs) == 0: break
        print(f'[bert] Embedding batch {batch} with {len(docs)} docs',
              file=sys.stderr)
        embeddings.extend(
            embedding_model.encode(docs, show_progress_bar=True)
        )
    return [ v.tolist() for v in embeddings ]

_cli_help="""
Usage: ttm [OPT]... embed [COMMAND-OPTION]... [METHOD [ARG]...]...

'ttm embed' takes one or more embedding methods as arguments. Each
of those embedding methods supports a different set of arguments that
can be used to adjust the operation of that one method. It is possible
to specify any number of embedding methods, each followed by optional
arguments. It is also possible to specify the same method multiple times,
possibly specifying different arguments each time.

All specified methods are individually applied to the input data. The
resulting embeddings are then concatenated in the same order the methods
are specified on the command line. Additionally, this command may be used
to concatenate document vectors from multiple files (see the '--append'
and '--include' options described below). If '--include' is specified,
specifying embedding METHODs becomes optional.

Methods
    bow         Create simple Bag of Words vectors.
    tfidf       Create Tf-Idf-weighted Bag of Words vectors.
    doc2vec     Train doc2vec embeddings, relying on the gensim doc2vec
                implementation. These embeddings are also used in Dimo
                Angelov's Top2Vec. The default values specified below are
                the same as used by Top2Vec when run with the Top2Vec
                'learn' settings. The Top2Vec 'deep-learn' settings
                are mostly the same, with 'epochs' set to 400 rather
                than 40. Top2Vec's 'fast-learn' uses 40 epochs, an 'hs'
                setting of 1 and a 'negative' setting of 5.
    bert        Create BERT embeddings using the sentence_transformers
                package. These embeddings are also used by Maarten
                Grootendorst's BERTopic.

Command Options
    -a, --append
            Append the resulting vectors to an existing 'highdim' column
            in the input file.
    --include FILE
            Also append the document vectors found in the 'highdim' column
            of FILE to the resulting vectors. This option can be specified
            multiple times to concatenate vectors from different files. The
            row order of the included FILE may differ from the input file.
            There must be an 'id' and a 'highdim' column in the included
            FILE, and each document id found in the input file must also
            be present in the included FILE.
    -h, --help
            Print this help message and exit.

Arguments for 'bow' and 'tfidf'
    --min-df N      Ignore tokens that appear in less than N documents.
                    N is a percentage of documents and must lie between
                    0 and 1. Default: 0.2.
    --max-df N      Ignore tokens that appear in more than N documents.
                    N is a percentage of documents and must lie between
                    0 and 1. Default: 0.5.

Arguments for 'doc2vec'
    --vector-size N     int         default:   300
    --min-count N       int         default:    50
    --window N          int         default:    15
    --sample N          float       default:  1e-5
    --negative N        int         default:     0
    --hs N              int         default:     1
    --epochs N          int         default:    40
    --dm N              int         default:     0
    --dbow-words N      int         default:     1

    --store-model PATH
         Store gensim's internal model representation at a given path.
         This is not used for anything inside ttm yet, but may be used in
         a future version. It may also be useful for inspecting the model
         independently of ttm.

Arguments for 'bert'
    --model MODEL
         Name of the pre-trained language model to use.
         The default MODEL is 'all-MiniLM-L6-v2'. Detailed
         information on pretrained language models is available
         as part of the sentence_transformers documentation. See
         https://www.sbert.net/docs/pretrained_models.html for a
         complete list of available models.
""".lstrip()

def _cli(argv, infile, outfile):
    all_opts, rest = getopt(argv, 'ha', ['help', 'append', 'include='])
    short2long = { '-h': '--help', '-a': '--append' }
    opts = { short2long.get(k, k).lstrip('-'): v for k, v in all_opts }
    opts['include'] = [ InputFile(v) for k, v in all_opts
                                      if k == '--include' ]
    if 'help' in opts:
        raise HelpRequested(_cli_help)
    elif len(rest) == 0 and len(opts['include']) == 0:
        raise CliError('No METHOD specified for ttm embed')
    methods = []
    while len(rest) > 0:
        if rest[0] in ['bow', 'tfidf']:
            cmd = bow if rest[0] == 'bow' else tfidf
            cmd_opts, rest = getopt(rest[1:], '', ['min-df=', 'max-df='])
            cmd_opts = { k.lstrip('-').replace('-', '_'): float(v)
                         for k, v in cmd_opts }
            for k, v in cmd_opts.items():
                if k == 'min_df' and (v < 0 or v > 1):
                    raise CliError('--min-df must lie between 0 and 1')
                elif k == 'max_df' and (v < 0 or v > 1):
                    raise CliError('--max-df must lie between 0 and 1')
            methods.append((cmd, cmd_opts))
        elif rest[0] == 'doc2vec':
            doc2vec_opts, rest = getopt(rest[1:], '',
                    ['vector-size=', 'min-count=', 'window=', 'sample=',
                     'negative=', 'hs=', 'epochs=', 'dm=', 'dbow-words=',
                     'store-model='])
            doc2vec_opts = { k.lstrip('-').replace('-', '_'): v
                             for k, v in doc2vec_opts }
            # Parse ints
            for k in ['vector_size', 'min_count', 'window', 'negative',
                      'hs', 'epochs', 'dm', 'dbow_words']:
                if k in doc2vec_opts:
                    doc2vec_opts[k] = int(doc2vec_opts[k])
            # Parse floats
            for k in ['sample']:
                if k in doc2vec_opts:
                    doc2vec_opts[k] = float(doc2vec_opts[k])
            methods.append((doc2vec, doc2vec_opts))
        elif rest[0] == 'bert':
            bert_opts, rest = getopt(rest[1:], '', ['model='])
            bert_opts = { k.lstrip('-'): v for k, v in bert_opts }
            methods.append((bert, bert_opts))
        else:
            raise CliError(f"Unknown ttm embed METHOD '{rest[0]}'")
    embeddings = []
    if 'append' in opts:
        embeddings.append(list(infile.column('highdim', map_f=json.loads)))
    for f in opts['include']:
        f.ensure_loaded()
        vs = { d: v for d, v in
               zip(f.column('id'), f.column('highdim', map_f=json.loads)) }
        embeddings.append([vs[d] for d in infile.column('id')])
        del vs
    for m, args in methods:
        print(f'Creating {m.__name__} embeddings', file=sys.stderr)
        embeddings.append(m(infile.column('content'), **args))
    input_lines = iter(infile.strip('highdim'))
    print(f'{next(input_lines)}\t{"highdim"}', file=outfile)
    for i, line in enumerate(input_lines):
        v = []
        for vs in embeddings:
            v.extend(vs[i])
        print(f'{line}\t{json.dumps(v)}', file=outfile)
