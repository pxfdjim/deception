import sys
from pathlib import Path
import logging
#print logger
class Logger(object):
    def __init__(self, log_file="log_file.log"):
        self.terminal = sys.stdout
        if isinstance(log_file, str):
            log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        self.file = open(log_file, "a")  # 使用追加模式
        sys.stdout=self #重定向输出

    def write(self, message):
        self.terminal.write(message)
        self.file.write(message)
        self.flush()

    def flush(self):
        self.file.flush()

    def __del__(self):
        self.file.close()
        sys.stdout = self.terminal

    # me
    @staticmethod
    def setup_logger(log_path):
        """配置标准 logging 模块"""
        logger = logging.getLogger('training_log')
        logger.setLevel(logging.INFO)

        # 防止重复添加 handler
        if not logger.handlers:
            file_handler = logging.FileHandler(log_path, mode='w')
            file_handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(message)s')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        return logger