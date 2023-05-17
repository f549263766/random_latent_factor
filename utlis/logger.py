import logging

from mmcv.utils import get_logger


def get_root_logger(log_file=None, log_level=logging.INFO):
    """Get root logger.
    Args:
        log_file (str, optional): File path of log. Defaults to None.
        log_level (int, optional): The level of logger.
            Defaults to :obj:`logging.INFO`.
    Returns:
        :obj:`logging.Logger`: The obtained logger
    """
    return get_logger('elm', log_file, log_level)
