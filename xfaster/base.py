"""
Logger classes
"""

from __future__ import absolute_import
from __future__ import print_function
import sys
import datetime as dt
import inspect
from configparser import RawConfigParser as rcp, DEFAULTSECT

__all__ = ["Logger", "XFasterConfig", "get_func_defaults", "extract_func_kwargs"]


class Logger(object):
    """Basic prioritized logger, extended from M. Hasselfield"""

    def __init__(self, verbosity=0, indent=True, logfile=None, **kwargs):
        self.timestamp = kwargs.pop("timestamp", True)
        self.prefix = kwargs.pop("prefix", None)
        self.levels = kwargs.pop("levels", {})
        kwargs.update(verbosity=self.get_level(kwargs.get("verbosity")))
        self.v = verbosity
        self.indent = indent
        self.set_logfile(logfile)

    def get_level(self, v):
        if v is None:
            return 0
        v = self.levels.get(v, v)
        if not isinstance(v, int):
            raise ValueError("Unrecognized logging level {}".format(v))
        return v

    def set_verbosity(self, level):
        """
        Change the verbosity level of the logger.
        """
        level = self.get_level(level)
        self.v = level

    set_verbose = set_verbosity

    def set_logfile(self, logfile=None):
        """
        Change the location where logs are written.  If logfile is None,
        log to STDOUT.
        """
        if hasattr(self, "logfile") and self.logfile != sys.stdout:
            self.logfile.close()
        if logfile is None:
            self.logfile = sys.stdout
        else:
            self.logfile = open(logfile, "a", 1)

    def format(self, s, level=0):
        """
        Format the input for writing to the logfile.
        """
        level = self.get_level(level)
        if self.prefix:
            s = "{}{}".format(self.prefix, s)
        else:
            s = "{}".format(s)
        if self.timestamp:
            stamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S%Z")
            s = "[ {} ]  {}".format(stamp, s)
        s = str(s)
        if self.indent:
            s = " " * level + s
        s += "\n"
        return s

    def write(self, s, level=0):
        level = self.get_level(level)
        if level <= self.v:
            self.logfile.write(self.format(s, level))

    def __call__(self, *args, **kwargs):
        """
        Log a message.

        Arguments
        ---------
        msg : string
            The message to log.
        level : int, optional
            The verbosity level of the message.  If at or below the set level,
            the message will be logged.
        """
        return self.write(*args, **kwargs)


class XFasterConfig(rcp):
    """
    ConfigParser subclass for storing command line options and config.
    """

    def __init__(self, defaults=None, default_sec="Uncategorized"):
        """
        Class that tracks command-line options for storage to disk.

        Arguments
        ---------
        defaults : dict
            Dictionary of overall configuration values.
            Eg: locals() at beginning of function, or vars(args) from argparse
        default_sec : string, optional
            The name of the default section in the configuration file.
        """
        rcp.__init__(self)  # supder doesn't work. old-style class?
        self.default_sec = default_sec
        if defaults is not None:
            self.update(defaults)

    def update(self, options, section=None):
        """
        Update configuration options with a dictionary. Behaves like
        dict.update() for specified section but also clears options of the same
        name from the default section.

        Arguments
        ---------
        options : dict
            The options to update
        section : string, optional
            Name of section to update. Default: self.default_sec
        """
        if section is None:
            section = self.default_sec
        if not self.has_section(section):
            self.add_section(section)
        # change kwargs to be like any other options
        kw = options.pop("kwargs", None)
        if isinstance(kw, dict):
            options.update(kw)
        for k, v in sorted(options.items()):
            self.remove_option(self.default_sec, k)
            self.set(section, k, str(v))

    def write(self, fp=None):
        """
        Write an .ini-format representation of the configuration state.
        Keys are stored alphabetically.

        Arguments
        ---------
        fp : file object
            If None, write to `sys.stdout`.
        """
        if fp is None:
            import sys

            fp = sys.stdout
        if self._defaults:
            fp.write("[%s]\n" % DEFAULTSECT)
            for (key, value) in sorted(self._defaults.items()):
                fp.write("%s = %s\n" % (key, str(value).replace("\n", "\n\t")))
            fp.write("\n")
        for section in self._sections:
            fp.write("[%s]\n" % section)
            for (key, value) in sorted(self._sections[section].items()):
                if key == "__name__":
                    continue
                if (value is not None) or (self._optcre == self.OPTCRE):
                    key = " = ".join((key, str(value).replace("\n", "\n\t")))
                fp.write("%s\n" % (key))
            fp.write("\n")


def get_func_defaults(func):
    """
    Return a dictionary containing the default values for each keyword
    argument of the given function

    Arguments
    ---------
    func : function or callable
        This function's keyword arguments will be extracted.

    Returns
    -------
    dict of kwargs and their default values
    """
    spec = inspect.getargspec(func)
    from collections import OrderedDict

    return OrderedDict(zip(spec.args[-len(spec.defaults) :], spec.defaults))


def extract_func_kwargs(func, kwargs, pop=False, others_ok=True, warn=False):
    """
    Extract arguments for a given function from a kwargs dictionary

    Arguments
    ---------
    func : function or callable
        This function's keyword arguments will be extracted.
    kwargs : dict
        Dictionary of keyword arguments from which to extract.
        NOTE: pass the kwargs dict itself, not **kwargs
    pop : bool, optional
        Whether to pop matching arguments from kwargs.
    others_ok : bool
        If False, an exception will be raised when kwargs contains keys
        that are not keyword arguments of func.
    warn : bool
        If True, a warning is issued when kwargs contains keys that are not
        keyword arguments of func.  Use with `others_ok=True`.

    Returns
    -------
    Dict of items from kwargs for which func has matching keyword arguments
    """
    spec = inspect.getargspec(func)
    func_args = set(spec.args[-len(spec.defaults) :])
    ret = {}
    for k in list(kwargs.keys()):
        if k in func_args:
            if pop:
                ret[k] = kwargs.pop(k)
            else:
                ret[k] = kwargs.get(k)
        elif not others_ok:
            msg = "Found invalid keyword argument: {}".format(k)
            raise TypeError(msg)
    if warn and kwargs:
        s = ", ".join(kwargs.keys())
        warn("Ignoring invalid keyword arguments: {}".format(s), Warning)
    return ret
