#!/usr/bin/env python3

import sys, gzip, bz2, lzma

class HelpRequested(Exception):
    pass
class CliError(Exception):
    pass
class ExpectedRuntimeError(Exception):
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
    """
    def __init__(self, corpus: InputFile, column: str, map_f=lambda x: x):
        self.corpus = corpus
        self.column = column
        self.map_f = map_f
    def __iter__(self):
        lines = iter(self.corpus)
        try:
            i_col = next(lines).split('\t').index(self.column)
        except StopIteration as e:
            raise ExpectedRuntimeError('Input file is empty') from e
        for line in lines:
            yield self.map_f(line.split('\t')[i_col])
    def __len__(self):
        return len(self.corpus) - 1

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
