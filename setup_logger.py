import logging

class CustomColorFormatter(logging.Formatter):
    COLORS = {
        "DEBUG": "\033[93m",  # Orange
        "INFO": "\033[92m",   # Green
    }
    RESET = "\033[0m"
    
    def format(self, record):
        message_color = self.COLORS.get(record.levelname, self.RESET)
        record.asctime = f"{self.formatTime(record, self.datefmt)}"
        formatted_message = f"{message_color}[{record.asctime}-{record.levelname}] {self.RESET}{record.msg}"        
        return formatted_message
    
def setup_logger(DEBUG_: bool = False):

    # Set up the logger
    formatter = CustomColorFormatter(datefmt='%H:%M:%S')

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logger = logging.getLogger("colored_logger")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    if DEBUG_:
        logger.debug("DEBUG: True")
    return logger