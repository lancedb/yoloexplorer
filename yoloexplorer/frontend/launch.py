from collections import defaultdict as _defaultdict
from streamlit.delta_generator import DeltaGenerator as _DeltaGenerator
from typing import Dict as _Dict
import streamlit as _st
import sys as _sys


# EXPORT
st = _st
ST: _Dict[str, _DeltaGenerator] = _defaultdict(_st.empty)


# VARS
_IS_WRAPPED = False


def run_streamlit(python_file):
    """
    IMPORTANT: import this function from a seperate file!
    Use this function by placing a call to it at the top of your
    check for __main__
    >>> from run_streamlit import run_streamlit, st, ST
    >>>
    >>> if __name__ == '__main__':
    >>>     run_streamlit(__file__)
    >>>     st.title('Hello')
    >>>     ST['info'].text('World!')
    """

    # do not wrap if streamlit is already running!
    global _IS_WRAPPED
    if _IS_WRAPPED:
        return True

    # ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ #

    import click
    import streamlit.cli

    @click.group()
    def run_streamlit():
        pass

    # For some reason I cant get streamlit to work without this subcommand?
    @run_streamlit.command('streamlit')
    @streamlit.cli.configurator_options
    def run_streamlit_subcommand(**kwargs):
        global _IS_WRAPPED
        _IS_WRAPPED = True
        streamlit.cli._apply_config_options_from_cli(kwargs)
        streamlit.cli._main_run(python_file, args)

    # ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ #

    # swap out arguments and keep arguments, first is the filename
    args = _sys.argv[1:]
    _sys.argv = [_sys.argv[0], 'streamlit']

    # run streamlit
    run_streamlit()
    exit(0)