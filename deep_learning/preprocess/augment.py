from deep_learning.lib.utility import *

class Augment:
    def __init__(self,params):
        super(Augment,self).__init__()
        self.logger = logging.getLogger(params['logger_name']+'.preprocess.augment')
        self.params = params
        self.H = self.params['preprocess']['input_size'][0]
        self.W = self.params['preprocess']['input_size'][1]
        self.train_augment = None
        self.test_augment = None
        self.valid_augment = None
        
    def __get_training_augment__(self):
        train_transform = [
#            albu.CropNonEmptyMaskIfExists(self.H,self.W),
            albu.HorizontalFlip(),
            albu.VerticalFlip(),
            albu.RandomContrast(),
#            albu.Resize(self.H,self.W),
            albu.Normalize(),
            ToTensorV2()
        ]
        self.logger.debug('Training Augumentation Loaded')
        return albu.Compose(train_transform)
    
    def __get_validation_augment__(self):
        test_transform = [
#            albu.Resize(self.H,self.W),
            albu.Normalize(),
            ToTensorV2()
        ]
        self.logger.debug('Validation Augumentation Loaded')
        return albu.Compose(test_transform)
    
    def __get_testing_augment__(self):
        test_transform = [
#            albu.Resize(self.H,self.W),
            albu.Normalize(),
            ToTensorV2()
        ]
        self.logger.debug('Testing Augumentation Loaded')
        return albu.Compose(test_transform)
    
    def process_augment(self):
        self.train_augment = self.__get_training_augment__()
        self.valid_augment = self.__get_validation_augment__()
        self.test_augment  = self.__get_testing_augment__()
    
    def run_check_augument(self):
        img = cv2.imread(DATA_DIR+'lenna.bmp')
        aug = self.__get_training_augment__()
        img = aug(image=img)['image']
        img = np.transpose(img,(1,2,0))
        plt.title('Train Augument')
        plt.imshow(img)
        plt.show()
        
        img = cv2.imread(DATA_DIR+'lenna.bmp')
        aug = self.__get_validation_augment__()
        img = aug(image=img)['image']
        img = np.transpose(img,(1,2,0))
        plt.title('Valid Augument')
        plt.imshow(img)
        plt.show()
        
        img = cv2.imread(DATA_DIR+'lenna.bmp')
        aug = self.__get_testing_augment__()
        img = aug(image=img)['image']
        img = np.transpose(img,(1,2,0))
        plt.title('Test Augument')
        plt.imshow(img)
        plt.show()
        
        