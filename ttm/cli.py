#!/usr/bin/env python3

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

def cli(argv):
    """
    Run the ttm cli for a given list of arguments. This function will
    not kill the python interpreter. If the user requests a help message,
    a HelpRequested Exception will be thrown. This Exception's string
    representation contains the requested help message. All other
    Exceptions indicate errors. For errors related to command line
    argument parsing and validation, the CliError class is used.
    """
    opts, cmd = getopt(argv, 'i:o:h', ['input=', 'output=', 'help'])
    short2long = { '-i': '--input', '-o': '--output', '-h': '--help' }
    opts = { short2long.get(k, k).lstrip('-'): v for k, v in opts }
    # Default values
    if 'input' not in opts: opts['input'] = '-'
    if 'output' not in opts: opts['output'] = '-'
    # Option processing
    c = { 'cat': cat, '20cat': c20cat, 'embed': embed, 'redim': redim,
          'cluster': cluster, 'desc': desc, 'eval': ttm_eval }
    if 'help' in opts:
        if not cmd:
            raise HelpRequested(_cli_help)
        elif cmd and cmd[0] in c:
            raise HelpRequested(c[cmd[0]]._cli_help)
        else:
            raise CliError('Unable to display help message for unknown '\
                          f"Unknown ttm COMMAND '{cmd[0]}'")
    if len(cmd) == 0:
        raise CliError('No COMMAND specified for ttm')
    elif cmd[0] in c:
        infile = InputFile(opts['input'])
        outfile = OutputFile(opts['output'])
        c[cmd[0]]._cli(argv=cmd[1:], infile=infile, outfile=outfile)
    else:
        raise CliError(f"Unknown ttm COMMAND '{cmd[0]}'")

if __name__ == "__main__":
    try:
        cli(sys.argv[1:])
    except HelpRequested as h:
        print(h, file=sys.stdout)
        exit(0)
    except BrokenPipeError as e:
        exit(0)
    except GetoptError as e:
        print(e, file=sys.stderr)
        exit(1)
    except CliError as e:
        print(e, file=sys.stderr)
        exit(1)
