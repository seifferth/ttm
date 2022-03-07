#!/usr/bin/env python3

import sys, gzip, bz2, lzma
import numpy as np
from scipy.sparse import csr_matrix

class HelpRequested(Exception):
    pass
class CliError(Exception):
    pass
class ExpectedRuntimeError(Exception):
    pass
class ColumnNotFound(Exception):
    pass
class EmptyColumnError(Exception):
    pass

def _open(filename, direction):
    """
    Wrapper around a number of file opening functions that takes the
    filename into account to handle automagic on the file compression
    if the filename ends in '.gz', '.bz2', or '.xz'.
    """
    if direction not in ['in','out']:
        raise Exception("Direction must be one of 'in' or 'out', "\
                       f"not '{direction}'")
    mode = 'rt' if direction == 'in' else 'wt'
    if filename == '-':
        return sys.stdin if direction == 'in' else sys.stdout
    elif filename.endswith('.gz'):
        return gzip.open(filename, mode)
    elif filename.endswith('.bz2'):
        return bz2.open(filename, mode)
    elif filename.endswith('.xz'):
        return lzma.open(filename, mode)
    else:
        return open(filename, mode)

class OutputFile():
    """
    Simple output file type. OutputFile supports special syntax for stdout
    (filename -) and does on-the-fly data compression if the filename ends
    with '.gz', '.bz2', or '.xz'.
    """
    def __init__(self, filename):
        self.file = _open(filename, 'out')
    def write(self, content):
        return self.file.write(content)

class CachingFileReader():
    """
    Iterable over all lines in a file. This class works both for files
    stored on disk and for stdandard input (filename -). If the corpus
    is read from stdin, each line will be recorded in an in-memory
    cache when first seen, and played back from that cache for further
    iterations. Note that this causes the entire corpus to live in memory,
    which could cause problems with large corpora. If memory usage is
    an issue, make sure to load files from disk.

    CachingFileReader is smart about filenames and performs on-the-fly data
    decompression if the filename ends with '.gz', '.bz2', or '.xz'.
    """
    def __init__(self, filename):
        self.cache = list()
        self.cache_complete = False
        self.filename = filename
        self.file_accessed = False
        self._len = None
        self.f = _open(filename, 'in')
        self.regular_file = self.f != sys.stdin and self.f.seekable()
        if self.regular_file: self.f.close()
    def __iter__(self):
        if self.regular_file:
            with _open(self.filename, 'in') as f:
                for line in f:
                    line = line.rstrip('\n').rstrip('\r')
                    yield line
        elif self.cache_complete:
            for line in self.cache:
                yield line
        elif not self.file_accessed:
            self.file_accessed = True
            for line in self.f:
                line = line.rstrip('\n').rstrip('\r')
                self.cache.append(line)
                yield line
            self.cache_complete = True
        else:
            raise Exception('Second iteration over input started before '\
                            'cache was fully populated during first one')
    def ensure_loaded(self):
        _ = len(self)   # len iterates over all lines (unless it already has)
    def __len__(self):
        if self._len == None: self._len = sum((1 for _ in self))
        return self._len

class InputFile():
    """
    Iterable over all lines in a tsv file. If strip_columns (list of strings)
    is specified to the constructor, those columns will be excluded from the
    iterator.
    """
    def __init__(self, filename, strip_columns=[], file_reader=None):
        if file_reader != None:
            self.file_reader = file_reader
        else:
            self.file_reader = CachingFileReader(filename)
        self.strip_columns = strip_columns
    def __iter__(self):
        if not self.strip_columns:
            for line in self.file_reader:
                yield line
        else:
            lines = iter(self.file_reader)
            try:
                header = next(lines).split('\t')
            except StopIteration as e:
                raise ExpectedRuntimeError('Input file is empty') from e
            yield '\t'.join([ header[i] for i in range(len(header))
                              if header[i] not in self.strip_columns ])
            for line in lines:
                line = line.split('\t')
                yield '\t'.join([ line[i] for i in range(len(line))
                                  if header[i] not in self.strip_columns ])
    def ensure_loaded(self):
        self.file_reader.ensure_loaded()
    def strip(self, column: str):
        """
        Return a new InputFile with a single column removed.
        """
        return InputFile(filename = self.file_reader.filename,
                         file_reader = self.file_reader,
                         strip_columns = [ column ] + self.strip_columns)
    def column(self, column: str, map_f=lambda x: x):
        """
        Return an iterable over the contents found in a specified column
        of this file. If map_f is supplied, that function will be applied
        to every row. This can be used to deserialize embedded data
        formats, such as json.
        """
        return Column(corpus=self, column=column, map_f=map_f)
    def __len__(self):
        return len(self.file_reader)

class Column():
    """
    Iterable over one individual column of an input file, not including
    the header row. An instance of this class can be constructed by
    calling the `column` method on a givem `InputFile`. If map_f is
    supplied, that function will be applied to every row. This can be
    used to deserialize embedded data formats, such as json.

    The filter_by and filters arguments can be used to select values
    based on the value of any column in the corpus. filter_by is a list
    of column names while filters is a list of functions that are applied
    to those columns to determine inclusion. Any column in the corpus can
    be used to filter contents.
    """
    def __init__(self, corpus: InputFile, column: str, map_f=lambda x: x,
                 filter_by=[], filters=[]):
        self.corpus = corpus
        self.column = column
        self.map_f = map_f
        self._len = None
        self.filter_by = filter_by
        self.filters = filters
        if len(filter_by) != len(filters): raise Exception("'filter' and "
            "'filter_by' must have the same length, but have a length of "
           f'{len(filter_by)} and {len(filters)} respectively')
    def __iter__(self):
        lines = iter(self.corpus)
        try:
            header = next(lines).split('\t')
            if self.column not in header:
                raise ColumnNotFound(f"Column '{self.column}' does "
                                      'not exist in the input file')
            for c in self.filter_by:
                if c not in header:
                    raise ColumnNotFound(f"Column '{c}' does "
                                         'not exist in the input file')
            i_col = header.index(self.column)
            i_filters = [ header.index(c) for c in self.filter_by ]
        except StopIteration as e:
            raise ExpectedRuntimeError('Input file is empty') from e
        for line in lines:
            line = line.split('\t')
            for j, f in enumerate(self.filters):
                if not f(line[i_filters[j]]): break
            else:
                yield self.map_f(line[i_col])
    def filter(self, column: str, f):
        return Column(self.corpus, self.column, self.map_f,
                      [column] + self.filter_by, [f] + self.filters)
    def __len__(self):
        if self._len == None: self._len = sum((1 for _ in self))
        return self._len
    def ensure_loaded(self):
        self.corpus.ensure_loaded()
    def peek(self):
        """Return the first row from this column"""
        self.ensure_loaded()
        return next(iter(self))
    def matrix(self, dtype=None):
        """
        Return the data either as a scipy.sparse.csr_matrix or as a
        numpy.ndarray, depending on how dense the first few rows of
        the data appear to be.
        """
        if len(self) == 0: raise EmptyColumnError()
        entries, zeros = 0, 0
        for _, row in zip(range(10), self):
            entries += len(row)
            zeros   += sum((1 for cell in row if cell == 0))
        return self.sparse_matrix(dtype=dtype) if zeros/entries > .5 else \
               self.dense_matrix(dtype=dtype)
    def dense_matrix(self, dtype=None):
        """
        Return the data as a numpy.ndarray. Note that this only works if
        map_f deserializes the data to a numerical format supported by
        numpy. Lists of ints or floats work fine, for instance.
        """
        if len(self) == 0: raise EmptyColumnError()
        n_rows, n_cols = len(self), len(self.peek())
        if dtype == None: dtype = type(self.peek()[0])
        m = np.ndarray((n_rows, n_cols), dtype=dtype)
        for i, v in enumerate(iter(self)):
            m[i] = v
        return m
    def sparse_matrix(self, dtype=None):
        """
        Return the data as a scipy.sparse.csr_matrix. Note that this only
        works if map_f deserializes the data to a numerical format supported
        by scipy. Lists of ints or floats work fine, for instance.
        """
        if len(self) == 0: raise EmptyColumnError()
        row, col, data = [], [], []
        for i, v in enumerate(iter(self)):
            for j, cell in enumerate(v):
                if cell == 0: continue
                row.append(i); col.append(j); data.append(cell)
        n_rows, n_cols = len(self), len(self.peek())
        if dtype == None: dtype = type(self.peek()[0])
        return csr_matrix((data, (row, col)), shape=(n_rows, n_cols),
                          dtype=dtype)

class PsqPairs():
    """
    Iterable over a file in psq-pairs format. The interface is similar
    to InputFile, but the iterator yields tuples of form (a, b) where
    b is the page following a.
    """
    def __init__(self, filename):
        self.file_reader = CachingFileReader(filename)
    def __iter__(self):
        for line in self.file_reader:
            a, b = line.split('\t')
            yield (a, b)
