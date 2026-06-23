# import sys
# from pathlib import Path
# import time

# #print logger
# class Logger(object):
#     def __init__(self, log_file="log_file.log"):
#         self.terminal = sys.stdout
#         if isinstance(log_file, str):log_file = Path(log_file)
#         log_file.parent.mkdir(parents=True, exist_ok=True)
#         self.file = open(log_file, "a")  # 使用追加模式
#         sys.stdout=self #重定向输出

#     def write(self, message):
#         self.terminal.write(message)
#         self.file.write(message)
#         self.flush()

#     def flush(self):
#         self.file.flush()

#     def __del__(self):
#         self.file.close()
#         sys.stdout = self.terminal

#     def info(self, message):
#         message=f'{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())} - {message}'
#         print(message)


import logging
from pathlib import Path

class Logger:
    def __init__(self, log_path):
        # 使用特定名称而不是 __name__，确保每个路径对应一个logger
        self.logger_name = f"logger_{log_path}"
        self.logger = logging.getLogger(self.logger_name)
        self.logger.setLevel(logging.INFO)
        
        # 清除现有的处理器
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)
        
        # 确保日志目录存在
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 创建文件处理器
        file_handler = logging.FileHandler(log_path, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # 设置格式
        formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # 添加处理器到logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # 避免重复记录
        self.logger.propagate = False
    
    def info(self, message):
        self.logger.info(message)
    
    def error(self, message):
        self.logger.error(message)
    
    def warning(self, message):
        self.logger.warning(message)
    
    def debug(self, message):
        self.logger.debug(message)
    
    def critical(self, message):
        self.logger.critical(message)