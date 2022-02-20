# ttm – Tsv-based Topic Modelling CLI

This repository contains some scripts which wrap selected pieces from
a number of python libraries in order to provide a somewhat consistent
interface to topic modelling.

Topic Modelling is approached as an exercise in document clustering.
The process that maps the string representation of all documents in a
certain corpus onto cluster ids typically consists of multiple steps,
where one step's output is used as the next step's input. This means
that Topic Modelling can be nicely described using shell pipes. Since
data exploration and evaluation may benefit from access to intermediary
document representations, ttm enriches rather than substitutes the
data at each step. The data is passed between steps as a tsv file with
a single header on the first row, and a single document on each of the
rows following the header. Each observation (i. e. each step's output)
is represented by a column.

Topic Modelling with ttm consists of four main steps:

1. Document Embeddings
2. Dimensionality Reduction
3. Clustering
4. Cluster Description

There are a number of different options for each of these steps. One
example pipeline could look like this:

    ttm cat corpus_dir |
        ttm embed doc2vec |
        ttm redim umap |
        ttm cluster hdbscan |
        ttm desc > result.tsv

`corpus_dir` is assumed to be a directory containing one or more plain
text files with somewhat sensible file names. Each of these files will be
split into a number of smaller documents and assigned an id that contains
the filename and an index within that file. `ttm embed` may take more than
one embedding method as its argument. If multiple embeddings are specified
(e. g. `ttm embed doc2vec bert`), these embeddings are concatenated.

In addition to the steps described above, there is a `ttm eval` step that
can be used to calculate a number of evaluation metrics for one or more
`result.tsv` files. There is also a `ttm 20cat` step which provides the
well known 20 newsgroups dataset and can be used as a drop-in replacement
for `ttm cat`.

See `ttm --help` and `ttm COMMAND --help` to view all options available
for each step.

## Note on memory usage

While input and output redirection through shell pipes is a very
convenient way of chaining programs together, this approach does
have its drawbacks when used in combination with machine learning
algorithms. Many of the embedding methods used with ttm make multiple
passes over the dataset during training. Since stdin is not *seekable*,
to use the pythonic expression (i. e. it does not support rewinding
to the beginning), the full length of that stream must remain in
memory until the training process is finished. This will obviously
cause scaling issues for large corpora. To work around this problem,
ttm also supports reading its input from a file specified via the `-i`
command line option. Since files opened this way are *seekable*, ttm can
free memory during training when invoked with this switch. As a result,
use of the `-i` and `-o` options should be preferred to input and output
redirection when working with large corpora.

## Installing

This project uses python setuptools and can be installed with pip. To
install ttm into the current user's home directory, for instance, one
can simply invoke `python3 -m pip install --user .`. After installation,
the program is available both as `ttm` and as `python3 -m ttm`. Invoking
it as `python3 -m ttm` also works without installing and may be convenient
during development.

The wrapped python modules are imported inside of the functions using
them, rather than globally. This allows to use ttm with a subset of
dependencies (and a corresponding subset of functionality, of course).
To install ttm with partial dependencies, one would add the `--no-deps`
flag to the pip invocation. Afterwards, one would manually install the
global dependencies and a selection of functional dependencies.

### Global dependencies

The following python packages are required for ttm to run (even if it
just displays help messages):

- numpy
- scipy
- tqdm

### Functional dependencies

The following python packages are required for ttm's main functionality,
but may be overlooked if the specific functionality they introduce is
not used:

- sklearn (used in ttm.embed.bow, ttm.embed.tfidf, ttm.redim.lda,
  ttm.redim.svd, ttm.cluster.aggl, ttm.cluster.kmeans, ttm.desc, ttm.eval)
- gensim (used in ttm.embed.doc2vec)
- flair (used in ttm.embed.bert, ttm.embed.sbert, ttm.embed.pool)
- sentence-transformers (used in ttm.embed.bert, ttm.embed.sbert)
- umap-learn (used in ttm.redim.umap)
- hdbscan (used in ttm.cluster.hdbscan)

Since these dependencies are still required for ttm's basic functionality
to be available, they are installed by default. Use the `--no-deps` option
during installation to skip them.

## License

All files in this repository are made available under the terms of the
GNU General Purpose License, version 3 or later. A copy of that license
is included in the repository as `license.txt`.
