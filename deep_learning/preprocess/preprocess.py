from deep_learning.lib.utility import *
from deep_learning.preprocess.augment import *
from deep_learning.preprocess.dataframe import *

class PreProcess:
    def __init__(self,params):
        super(PreProcess,self).__init__()
        self.logger = logging.getLogger(params['logger_name']+'.preprocess')
        self.params = params
        self.aug = Augment(self.params)
        self.df = Dataframe(self.params)
    
    def process_dataframe(self):
        self.aug.process_augment()
        self.df.process_dataframe()
        
    def run_check_augument(self):
        self.aug.run_check_augument()