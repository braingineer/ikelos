import logging
import os

def duallog(loggername, shell_level="info", file_loc="logs/", disable=False):

    levels = level = {"debug": logging.DEBUG, "warning":logging.WARNING,
                      "info": logging.INFO, "error":logging.ERROR,
                      "critical":logging.CRITICAL}
    logger = logging.getLogger(loggername)
    logger.setLevel(logging.DEBUG)
    if not logger.handlers and not disable:
        # have to have the '/' for dir checks. 
        safe_loc = file_loc+("/" if file_loc[-1] != "/" else "")
        if not os.path.exists(safe_loc):
            os.makedirs(safe_loc)
        fh = logging.FileHandler("{}/{}.debug.log".format(file_loc, loggername))
        fh.setLevel(logging.DEBUG)
        sh = logging.StreamHandler()
        sh.setLevel(levels[shell_level])
        logger.addHandler(fh)
        logger.addHandler(sh)
    return logger
