
import os
class Logger():
    def __init__(self,filename : str, log_path = 'logs/loss/'):
        os.makedirs(log_path, exist_ok=True)
        self.log_path = log_path + filename
    def initlog(self):
        self.looger =  open(file = self.log_path,mode='a')
        return self.looger
    def close(self):
        self.looger.close()