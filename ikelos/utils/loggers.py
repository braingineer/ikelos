import logging
import baal

members = ["treecut", "hlfdebug", "nlg", "csp", "trees"]
levels = level = {"debug": logging.DEBUG, "warning":logging.WARNING,
                  "info": logging.INFO, "error":logging.ERROR,
                  "critical":logging.CRITICAL}

def shell_log(loggername="", level="debug"):
    logger = logging.getLogger(loggername)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(levels[level])
        logger.addHandler(ch)
        logger.setLevel(levels[level])
    return logger


def file_log(loggername, filename, level="debug"):
    logger = logging.getLogger(loggername)
    fh = logging.FileHandler(filename)
    fh.setLevel(levels[level])
    logger.addHandler(fh)
    logger.setLevel(levels[level])

def duallog(loggername, shell_level="info", file_loc="logs/", disable=False):
    logger = logging.getLogger(loggername)
    logger.setLevel(logging.DEBUG)
    if not logger.handlers and not disable:
        print(file_loc)
        if baal.utils.general.ensure_dir(file_loc+("/" if file_loc[-1] != "/" else "")):
            print("Created directory for logger at {}".format(file_loc))
        fh = logging.FileHandler("{}/{}.debug.log".format(file_loc, loggername))
        fh.setLevel(logging.DEBUG)
        sh = logging.StreamHandler()
        sh.setLevel(levels[shell_level])
        logger.addHandler(fh)
        logger.addHandler(sh)
    return logger

def turn_on(name, level="debug", shell=True, filename=None):
    if shell:
        shell_log(name, level)
    elif filename:
        file_log(name, filename, level)

def get(name, level='debug', turn_on=True):
    if turn_on:
        return shell_log(name, level)
    else:
        logger = logging.getLogger(name)
        logger.addHandler(NullHandler())
        return logger


def set_level(name, level):
    logging.getLogger(name).setLevel(levels[level])


class NullHandler(logging.Handler):
    def emit(self, record):
        pass
