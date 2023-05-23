import logging

from mmengine.logging import MMLogger


def get_root_logger(log_file=None, log_level=logging.INFO):
    """Get root logger.
    Args:
        log_file (str, optional): File path of log. Defaults to None.
        log_level (int, optional): The level of logger.
            Defaults to :obj:`logging.INFO`.
    Returns:
        :obj:`logging.Logger`: The obtained logger
    """

    return MMLogger(name='pty', log_file=log_file, log_level =log_level)
