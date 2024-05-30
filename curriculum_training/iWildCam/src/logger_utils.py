import logging
import sys


def get_logger(l_name: str, l_file: str = 'logger.out'):
    """

    :param l_name: logger name
    :param l_file: logger filename
    :return:
    """
    # Create a logger
    logger = logging.getLogger(l_name)

    # Configure the logger (optional)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - [%(levelname)s] %(message)s', "%Y-%m-%d %H:%M:%S")

    # logging to console
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.DEBUG)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    # logging to file
    fh = logging.FileHandler(l_file)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger
