import logging

class Log:
    @staticmethod
    def set(log_name):
        logger = logging.getLogger(log_name)
        if logger.hasHandlers():
            logger.handlers.clear()
        c_handler = logging.StreamHandler()
        c_format = logging.Formatter('%(name)s[%(levelname)s]: %(message)s')
        c_handler.setFormatter(c_format)
        logger.addHandler(c_handler)
        logger.setLevel(logging.DEBUG)
        return logger