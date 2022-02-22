#!/usr/bin/env python3

from getopt import getopt
from tqdm import tqdm
from .types import *
import sys, json

def bow(docs, min_df=.2, max_df=.5):
    from sklearn.feature_extraction.text import CountVectorizer
    print('\rCreating term-document matrix for bag of words embeddings',
          10*' ', file=sys.stderr)
    vs = CountVectorizer(min_df=min_df, max_df=max_df).fit_transform(docs)
    for v in vs:
        yield v.toarray().flatten().tolist()

def tfidf(docs, min_df=.2, max_df=.5):
    from sklearn.feature_extraction.text import TfidfVectorizer
    print('\rCreating weighted term-document matrix for tf-idf embeddings',
          10*' ', file=sys.stderr)
    vs = TfidfVectorizer(min_df=min_df, max_df=max_df).fit_transform(docs)
    for v in vs:
        yield v.toarray().flatten().tolist()

def doc2vec(docs, vector_size=300, min_count=50, window=15, sample=1e-5,
            negative=0, hs=1, epochs=40, dm=0, dbow_words=1, store_model=None):
    from gensim.models.doc2vec import Doc2Vec, TaggedDocument
    from gensim.utils import simple_preprocess
    import logging
    logging.basicConfig(format='\r[doc2vec] %(levelname)s: %(message)s',
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
    for i, _ in enumerate(docs):
        yield model.dv[i].tolist()

def bert(docs, model='bert-base-uncased', pooling='cls'):
    from flair.data import Sentence
    from flair.embeddings import TransformerDocumentEmbeddings
    model = TransformerDocumentEmbeddings(model, fine_tune=False,
                                          pooling=pooling)
    for d in docs:
        s = Sentence(d); model.embed(s)
        yield s.get_embedding().tolist()

def sbert(docs, model='all-MiniLM-L6-v2'):
    from flair.data import Sentence
    from flair.embeddings import SentenceTransformerDocumentEmbeddings
    model = SentenceTransformerDocumentEmbeddings(model)
    for d in docs:
        s = Sentence(d); model.embed(s)
        yield s.get_embedding().tolist()

def pool(docs, pooling='mean', word_embeddings=[], flair_embeddings=[]):
    from flair.data import Sentence
    from flair.embeddings import WordEmbeddings, FlairEmbeddings, \
                                 DocumentPoolEmbeddings
    embeddings = []
    for e in word_embeddings:
        embeddings.append(WordEmbeddings(e, fine_tune=False))
    for e in flair_embeddings:
        embeddings.append(FlairEmbeddings(e, fine_tune=False))
    model = DocumentPoolEmbeddings(embeddings)
    for d in docs:
        s = Sentence(d); model.embed(s)
        yield s.get_embedding().tolist()

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
    bert        Create transformer embeddings using the flair wrapper around
                pretrained models from huggingface.co.
    sbert       Create Sentence BERT embeddings using the flair wrapper
                around the sentence_transformers package. These embeddings
                are also used by Maarten Grootendorst's BERTopic.
    pool        Create document embeddings by pooling word embeddings for
                every word in the document, as proposed by Akbik et al.

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
    --highdim-only
            Include only 'id' and 'highdim' columns in output. This is not
            advisable for general use, since the 'content' column is also
            used in later steps. It may be convenient to create standalone
            embedding files that can later be added to a complete input file
            via the '--include' option, however.
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
    --pooling METHOD
         The pooling method to use for combining the transformer word
         embeddings. The pooling METHOD may be any of 'cls', 'max' or
         'mean'. Default: 'cls'.
    --model MODEL
        Name of the pre-trained language model to use. The default MODEL
        is 'bert-base-uncased'. A complete list of supported models is
        available at https://huggingface.co/models.

Arguments for 'sbert'
    --model MODEL
         Name of the pre-trained language model to use.
         The default MODEL is 'all-MiniLM-L6-v2'. Detailed
         information on pretrained language models is available
         as part of the sentence_transformers documentation. See
         https://www.sbert.net/docs/pretrained_models.html for a
         complete list of available models.

Arguments for 'pool'
    --pooling METHOD
         The pooling method to use for combining the specified word
         embeddings. The pooling METHOD may be any of 'mean', 'min' or
         'max'. Default: 'mean'.
    --word-embeddings MODEL[,MODEL]...
         Comma-separated list of pretrained word embedding models to use.
         See 'pydoc flair.embeddings.WordEmbeddings.__init__' for further
         information on supported embeddings. If this option is specified
         multiple times, the lists are concatenated.
    --flair-embeddings MODEL[,MODEL]...
         Comma-separated list of pretrained contextual string embeddings
         to use. See 'pydoc flair.embeddings.FlairEmbeddings.__init__' for
         further information on supported embeddings. If this option is
         specified multiple times, the lists are concatenated.
""".lstrip()

def _cli(argv, infile, outfile):
    all_opts, rest = getopt(argv, 'ha', ['help', 'append', 'include=',
                            'highdim-only'])
    short2long = { '-h': '--help', '-a': '--append' }
    opts = { short2long.get(k, k).lstrip('-').replace('-', '_'): v
             for k, v in all_opts }
    opts['include'] = [ v for k, v in all_opts if k == '--include']
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
            bert_opts, rest = getopt(rest[1:], '', ['model=', 'pooling='])
            bert_opts = { k.lstrip('-'): v for k, v in bert_opts }
            pooling = bert_opts.get('pooling', None)
            if pooling != None and pooling not in ['cls', 'max', 'mean']:
                raise CliError("bert --pooling METHOD must be 'cls', 'max' "
                              f"or 'mean', not '{pooling}'")
            methods.append((bert, bert_opts))
        elif rest[0] == 'sbert':
            sbert_opts, rest = getopt(rest[1:], '', ['model='])
            sbert_opts = { k.lstrip('-'): v for k, v in sbert_opts }
            methods.append((sbert, sbert_opts))
        elif rest[0] == 'pool':
            all_pool_opts, rest = getopt(rest[1:], '', ['pooling=',
                                'word-embeddings=', 'flair-embeddings='])
            pool_opts = { 'word_embeddings': [], 'flair_embeddings': [] }
            for k, v in all_pool_opts:
                if k == '--pooling':
                    if v not in {'mean','min','max'}:
                        raise CliError("pool --pooling METHOD must be "
                                      f"'mean', 'min' or 'max', not '{v}'")
                    pool_opts['pooling'] = v
                if k == '--word-embeddings':
                    pool_opts['word_embeddings'].extend(v.split(','))
                if k == '--flair-embeddings':
                    pool_opts['flair_embeddings'].extend(v.split(','))
            if not pool_opts['word_embeddings'] \
                                and not pool_opts['flair_embeddings']:
                raise CliError('At least one of --word-embeddings or '
                               '--flair-embeddings needs to be specified '
                               "for 'ttm embed pool' embeddings")
            methods.append((pool, pool_opts))
        else:
            raise CliError(f"Unknown ttm embed METHOD '{rest[0]}'")
    embeddings = []
    total_docs = len(infile.column('id'))
    if 'append' in opts:
        embeddings.append(infile.column('highdim', map_f=json.loads))
    for filename in opts['include']:
        f = InputFile(filename)
        if len(f) != len(infile):
            raise ExpectedRuntimeError(
                f"The included file '{filename}' contains {len(f)} lines, "
                f"but the main input file contains {len(infile)}"
            )
        for line, id_a, id_b in zip(range(2, total_docs+2),
                                    infile.column('id'), f.column('id')):
            if id_a == id_b: continue
            raise ExpectedRuntimeError(
                f"The row order in '{filename}' differs from the one found "
                f"in the main input file.\nLine {line}: Mismatch between "
                f"'{id_a}' (main input file) and '{id_b}' ({filename}).")
        embeddings.append(iter(f.column('highdim')))
    for m, args in methods:
        embeddings.append(m(infile.column('content'), **args))
    if 'highdim_only' in opts:
        print(f'{"id"}\t{"highdim"}', file=outfile)
        input_lines = iter(infile.column('id'))
    else:
        input_lines = iter(infile.strip('highdim'))
        print(f'{next(input_lines)}\t{"highdim"}', file=outfile)
    for line in tqdm(input_lines, 'Embedding documents', total=total_docs):
        v = []
        for e in embeddings:
            v.extend(next(e))
        print(f'{line}\t{json.dumps(v)}', file=outfile)
