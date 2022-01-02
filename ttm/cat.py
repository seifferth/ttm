#!/usr/bin/env python3

from getopt import getopt
from common_types import *

_cli_help="""
Usage: ttm [OPT]... cat [COMMAND-OPTION]... TYPE DIR

Positional Arguments
    TODO

Command Options
    TODO
""".lstrip()

def _cli(argv, infile, outfile):
    opts, cmd = getopt(argv, 'h', ['help'])
    short2long = { '-h': '--help' }
    opts = { short2long.get(k, k): v for k, v in opts }
    if '--help' in opts:
        raise HelpRequested(_cli_help)
    raise Exception('ttm cat not implemented yet')

