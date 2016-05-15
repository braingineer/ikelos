import logging
import os

def duallog(loggername, shell_level="info", file_loc="logs/", disable=False):

    levels = level = {"debug": logging.DEBUG, "warning":logging.WARNING,
                      "info": logging.INFO, "error":logging.ERROR,
                      "critical":logging.CRITICAL}
    logger = logging.getLogger(loggername)
    logger.setLevel(logging.DEBUG)
    if not logger.handlers and not disable:
        if not os.path.exists(file_loc+("/" if file_loc[-1] != "/" else "")):
            os.makedirs(d)
        fh = logging.FileHandler("{}/{}.debug.log".format(file_loc, loggername))
        fh.setLevel(logging.DEBUG)
        sh = logging.StreamHandler()
        sh.setLevel(levels[shell_level])
        logger.addHandler(fh)
        logger.addHandler(sh)
    return logger
