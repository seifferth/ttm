#!/usr/bin/env python3

class HelpRequested(Exception):
    pass

class Corpus():
    """
    Iterable over all lines in a corpus tsv file. This class works for
    both seekable and non-seekable files. If the corpus is read from
    a non-seekable file, each line will be recorded in an in-memory
    cache when first seen, and played back from that cache for further
    iterations. Note that this causes the entire corpus to live in memory,
    which could cause problems with large corpora. If memory usage is
    an issue, make sure to load the corpus from a seekable file.
    """
    def __init__(self, file):
        self.cache = list()
        self.cache_complete = False
        self.file = file
        self.file_accessed = False
    def __iter__(self):
        if self.file.seekable():
            self.file.seek(0)
            for line in self.file:
                line = line.rstrip('\n').rstrip('\r')
                yield line
        elif self.cache_complete:
            for line in self.cache:
                yield line
        elif not self.file_accessed:
            self.file_accessed = True
            for line in self.file:
                line = line.rstrip('\n').rstrip('\r')
                self.cache.append(line)
                yield line
            self.cache_complete = True
        else:
            raise Exception('Second iteration over corpus started before '\
                            'cache was fully populated during first one')
    def column(self, column: str):
        """
        Return an iterable over the contents found in a specified column
        of this corpus.
        """
        return Column(corpus=self, column=column)

class Column():
    """
    Iterable over one individual column of a corpus, not including the
    header row. An instance of this class can be constructed by calling
    the `column` method on a givem `Corpus`.
    """
    def __init__(self, corpus: Corpus, column: str):
        self.corpus = corpus
        self.column = column
    def __iter__(self):
        lines = iter(self.corpus)
        i_col = next(lines).split('\t').index(self.column)
        for line in lines:
            yield line.split('\t')[i_col]
