from deep_learning.lib.utility import *
from deep_learning.preprocess.preprocess import *

class Visualize:
    def __init__(self,params):
        self.params = params
        self.logger = logging.getLogger(params['logger_name']+'.visualize')
        self.params = params
        self.preprocess = PreProcess(params)
        self.preprocess.process_dataframe()
    
    def visualize_dataframe(self):
        print(self.preprocess)
    
    def visualize_images(self):
        df = self.preprocess.df.train
        imageIds = []
        for c in df.Class.unique():
            imageIds.extend(np.random.choice(np.unique(df[(df['Class']==c) & (df['Label']==1)]['ImageID'].values),self.params['visualize']['sample'],replace=False))
#        outline_mask(df[df['ImageID'].isin(imageIds)].copy(),'ImageID','EncodedPixels','Class',image_path=DATA_DIR+self.params['visualize']['image_path'],save_path=DATA_DIR+self.params['visualize']['save_path'],class_names=df.Class.unique(),plot=False)
        outline_mask(df[df['ImageID'].isin(imageIds)].copy(),'ImageID','EncodedPixels','Class',image_path=DATA_DIR+self.params['visualize']['image_path'],save_path=DATA_DIR+self.params['visualize']['save_path'],class_names=df.Class.unique(),plot=False)
        