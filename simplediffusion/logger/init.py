import logging

from rich.console import Console
from rich.highlighter import ReprHighlighter
from rich.logging import RichHandler


class BetterReprHighlighter(ReprHighlighter):
    def __init__(self):
        super().__init__()
        self.highlights.append(r"(?P<number>\d+\.\d+s)")


def initialize():
    FORMAT = "%(message)s"
    logging.basicConfig(level=logging.INFO, format=FORMAT, datefmt="[%X]", handlers=[RichHandler(highlighter=BetterReprHighlighter())])
    
    # 将所有日志的 handler 设置为 RichHandler
    for name in logging.root.manager.loggerDict:
        logging.getLogger(name).handlers = []
        logging.getLogger(name).addHandler(RichHandler(highlighter=BetterReprHighlighter()))