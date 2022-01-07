#!/usr/bin/env python3

import sys, gzip, bz2, lzma
from getopt import getopt, GetoptError
from common_types import *
import cat, c20cat, embed, redim, cluster, desc, eval as ttm_eval

_cli_help="""
Usage: ttm [GLOBAL-OPTION]... COMMAND [COMMAND-OPTION]...

See 'ttm COMMAND --help' for subcommand-specific help messages.

Global Options
    -i FILE, --input FILE
                  Read input from FILE (default: stdin). If the filename
                  extension is one of '.gz', '.bz2' or '.xz', the input
                  will be decompressed on the fly.
    -o FILE, --output FILE
                  Write output to FILE (default: stdout). If the filename
                  extension is one of '.gz', '.bz2' or '.xz', the output
                  will be compressed on the fly.
    -h, --help    Print this help message and exit.

Commands
    cat           Copy the corpus into 'id' and 'content' columns.
    20cat         Copy the 20 newsgroups dataset into 'id' and 'content'
                  columns.
    embed         Create document embeddings from the 'content' column and
                  store them as 'highdim'.
    redim         Perform dimensionality reduction on the 'highdim' column
                  and store the resulting vectors as 'lowdim'.
    cluster       Cluster the vectors in 'lowdim' and store the resulting
                  cluster ids as 'cluster'.
    desc          Create cluster descriptions based on the 'cluster' and
                  'content' columns and store the resulting word lists
                  as 'desc'.
    eval          Evaluate topic models based on 'cluster' and 'desc'
                  columns.
""".lstrip()

def _open(filename, direction):
    """
    Wrapper around a number of file opening functions that takes the
    filename into account to handle a number of special cases, such
    as compressed files and special syntax for stdin and stdout.
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

def cli(argv):
    """
    Run the ttm cli for a given list of arguments. This function will
    not kill the python interpreter. If the user requests a help message,
    a HelpRequested Exception will be thrown. This Exception's string
    representation contains the requested help message. All other
    Exceptions indicate errors.
    """
    opts, cmd = getopt(argv, 'i:o:h', ['input=', 'output=', 'help'])
    short2long = { '-i': '--input', '-o': '--output', '-h': '--help' }
    opts = { short2long.get(k, k).lstrip('-'): v for k, v in opts }
    # Default values
    if 'input' not in opts: opts['input'] = '-'
    if 'output' not in opts: opts['output'] = '-'
    # Option processing
    if 'help' in opts:
        raise HelpRequested(_cli_help)
    infile = Corpus(_open(opts['input'], 'in'))
    outfile = _open(opts['output'], 'out')
    c = { 'cat': cat._cli, '20cat': c20cat._cli, 'embed': embed._cli,
          'redim': redim._cli, 'cluster': cluster._cli, 'desc': desc._cli,
          'eval': ttm_eval._cli }
    if len(cmd) == 0:
        raise Exception('No COMMAND specified for ttm')
    elif cmd[0] in c:
        c[cmd[0]](argv=cmd[1:], infile=infile, outfile=outfile)
    else:
        raise Exception(f"Unknown ttm COMMAND '{cmd[0]}'")

if __name__ == "__main__":
    try:
        cli(sys.argv[1:])
    except HelpRequested as h:
        print(h, file=sys.stdout)
        exit(0)
    except GetoptError as e:
        print(e, file=sys.stderr)
        exit(1)
    except Exception as e:
        print(e, file=sys.stderr)
        exit(1)
