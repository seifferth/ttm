#!/usr/bin/env python3

import sys, gzip, bz2, lzma

class HelpRequested(Exception):
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
    if filename.endswith('.gz'):
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
        self.file = sys.stdout if filename == '-' \
                    else _open(filename, 'out')
    def write(self, content):
        return self.file.write(content)

class InputFile():
    """
    Iterable over all lines in a corpus tsv file. This class works both
    for files stored on disk and for stdandard input (filename -). If
    the corpus is read from stdin, each line will be recorded in an
    in-memory cache when first seen, and played back from that cache for
    further iterations. Note that this causes the entire corpus to live
    in memory, which could cause problems with large corpora. If memory
    usage is an issue, make sure to load the corpus from disk.

    InputFile is smart about filenames and performs on-the-fly data
    decompression if the filename ends with '.gz', '.bz2', or '.xz'.
    """
    def __init__(self, filename):
        self.cache = list()
        self.cache_complete = False
        self.filename = filename
        self.file_accessed = False
    def __iter__(self):
        if self.filename != '-':
            with _open(self.filename, 'in') as f:
                for line in f:
                    line = line.rstrip('\n').rstrip('\r')
                    yield line
        elif self.cache_complete:
            for line in self.cache:
                yield line
        elif not self.file_accessed:
            self.file_accessed = True
            for line in sys.stdin:
                line = line.rstrip('\n').rstrip('\r')
                self.cache.append(line)
                yield line
            self.cache_complete = True
        else:
            raise Exception('Second iteration over input started before '\
                            'cache was fully populated during first one')
    def column(self, column: str, map_f=lambda x: x):
        """
        Return an iterable over the contents found in a specified column
        of this file. If map_f is supplied, that function will be applied
        to every row. This can be used to deserialize embedded data
        formats, such as json.
        """
        return Column(corpus=self, column=column, map_f=map_f)

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
        i_col = next(lines).split('\t').index(self.column)
        for line in lines:
            yield self.map_f(line.split('\t')[i_col])
