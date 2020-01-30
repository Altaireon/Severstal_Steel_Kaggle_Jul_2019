from data.lib.utility import *

class _SteelDataset(Dataset):
    def __init__(self, df, mode, logger_name, path, preprocess=None):
        self.logger = logging.getLogger(logger_name+'.data.dataset')
        self.mode    = mode
        self.preprocess = preprocess
        self.df = df
        self.num_image = self.df.shape[0]//4
        self.path = DATA_DIR+path
        self.uid = df['ImageID'].unique()

    def __str__(self):
        num1 = (self.df['Class']==1).sum()
        num2 = (self.df['Class']==2).sum()
        num3 = (self.df['Class']==3).sum()
        num4 = (self.df['Class']==4).sum()
        pos1 = ((self.df['Class']==1) & (self.df['Label']==1)).sum()
        pos2 = ((self.df['Class']==2) & (self.df['Label']==1)).sum()
        pos3 = ((self.df['Class']==3) & (self.df['Label']==1)).sum()
        pos4 = ((self.df['Class']==4) & (self.df['Label']==1)).sum()
        neg1 = num1-pos1
        neg2 = num2-pos2
        neg3 = num3-pos3
        neg4 = num4-pos4

        num = self.df.shape[0]
        pos = (self.df['Label']==1).sum()
        neg = num-pos


        string  = '\n'
        string += '\tmode    = %s\n'%self.mode
        string += '\tnum_image = %8d\n'%self.num_image
        string += '\tlen       = %8d\n'%num
        if self.mode in ['train','valid']:
            string += '\t\tpos1, neg1 = %5d  %0.3f,  %5d  %0.3f\n'%(pos1,pos1/num,neg1,neg1/num)
            string += '\t\tpos2, neg2 = %5d  %0.3f,  %5d  %0.3f\n'%(pos2,pos2/num,neg2,neg2/num)
            string += '\t\tpos3, neg3 = %5d  %0.3f,  %5d  %0.3f\n'%(pos3,pos3/num,neg3,neg3/num)
            string += '\t\tpos4, neg4 = %5d  %0.3f,  %5d  %0.3f\n'%(pos4,pos4/num,neg4,neg4/num)
        return string


    def __len__(self):
        return len(self.uid)


    def __getitem__(self, index):
        image_id = self.uid[index-1]
        label = np.reshape(np.array([
            self.df.loc[self.df['ImageId_ClassId']==image_id + '_1','Label'].values[0],
            self.df.loc[self.df['ImageId_ClassId']==image_id + '_2','Label'].values[0],
            self.df.loc[self.df['ImageId_ClassId']==image_id + '_3','Label'].values[0],
            self.df.loc[self.df['ImageId_ClassId']==image_id + '_4','Label'].values[0],
        ],dtype=int),(4,1,1))
        rle = [
            self.df.loc[self.df['ImageId_ClassId']==image_id + '_1','EncodedPixels'].values[0],
            self.df.loc[self.df['ImageId_ClassId']==image_id + '_2','EncodedPixels'].values[0],
            self.df.loc[self.df['ImageId_ClassId']==image_id + '_3','EncodedPixels'].values[0],
            self.df.loc[self.df['ImageId_ClassId']==image_id + '_4','EncodedPixels'].values[0],
        ]
        image = cv2.imread(self.path+image_id)
        mask  = np.array([run_length_decode(r, height=256, width=1600, fill_value=1.0) for c,r in zip([1,2,3,4],rle)])
        mask = mask.max(0)
        if self.preprocess is None:
            return image, mask, label, image_id
        else:
            ag = self.preprocess(image=image,mask=mask)
            image = ag['image']
            mask = ag['mask'].long()
            return image, mask, label, image_id
    
class Loader():
    def __init__(self,preprocess, params):
        super(Loader,self).__init__()
        self.logger = logging.getLogger(params['logger_name']+'.data.dataloader')
        self.train_loader   = None
        self.valid_loader   = None
        self.test_loader    = None
        self.params = params
        self.preprocess = preprocess
    
    def __process_train__(self):
        train_dataset = _SteelDataset(
            mode    = 'train',
            df     = self.preprocess.df.train,
            logger_name = self.params['logger_name'],
            path = self.params['path']['train'],
            preprocess = self.preprocess.aug.train_augment
        )
        self.logger.debug(str(train_dataset))
        self.logger.debug("Setting up Train Loader with batch_size %d => STARTING",self.params['batch_size'][0])
        train_loader = DataLoader(
            train_dataset,
            #sampler     = BalanceClassSampler(train_dataset, 3*len(train_dataset)),
#            sampler    = SequentialSampler(train_dataset),
            sampler     = FiveBalanceClassSampler(train_dataset),
            batch_size  = self.params['batch_size'][0],
            drop_last   = True,
            num_workers = 1,
            pin_memory  = True,
#            collate_fn  = null_collate
        )
        self.logger.debug("Setting up Train Loader => SUCCESS")
        return train_loader
    
    def __process_valid__(self):
        valid_dataset = _SteelDataset(
            mode    = 'valid',
            df     = self.preprocess.df.valid,
            logger_name = self.params['logger_name'],
            path = self.params['path']['valid'],
            preprocess = self.preprocess.aug.valid_augment
        )
        self.logger.debug(str(valid_dataset))
        self.logger.debug("Setting up Valid Loader with batch_size %d => STARTING",self.params['batch_size'][1])
        valid_loader = DataLoader(
            valid_dataset,
            sampler     = RandomSampler(valid_dataset),
            #sampler     = RandomSampler(valid_dataset),
            batch_size  = self.params['batch_size'][1],
            drop_last   = False,
            num_workers = 1,
            pin_memory  = True,
#            collate_fn  = null_collate
        )
        self.logger.debug("Setting up Valid Loader => SUCCESS")
        return valid_loader
    
    def __process_test__(self):
        test_dataset = _SteelDataset(
            mode    = 'test',
            df     = self.preprocess.df.test,
            logger_name = self.params['logger_name'],
            path = self.params['path']['test'],
            preprocess = self.preprocess.aug.test_augment
        )
        self.logger.debug(str(test_dataset))
        self.logger.debug("Setting up Test Loader with batch_size %d => STARTING",self.params['batch_size'][2])
        test_loader = DataLoader(
            test_dataset,
            sampler    = SequentialSampler(test_dataset),
            batch_size  = self.params['batch_size'][2],
            drop_last   = False,
            num_workers = 1,
            pin_memory  = True,
#            collate_fn  = null_collate
        )
        self.logger.debug("Setting up Test Loader => SUCCESS")
        return test_loader
    
    def process_data(self):
        self.logger.debug(f'\nBatch Size:\nTrain: {self.params["batch_size"][0]}\nValid: {self.params["batch_size"][1]}\nTest: {self.params["batch_size"][2]}')
        self.train_loader = self.__process_train__()
        self.valid_loader = self.__process_valid__()
        self.test_loader  = self.__process_test__()
        self.logger.debug(f'Processing Data => SUCCESS')
        
    def run_check_loader(self):
        self.logger.info("Training Data.........")
        for img,mask,label,image_id in self.train_loader:
            self.logger.info(f'Image Shape = {img.shape}')
            assert(img.shape == (self.params['batch_size'][0],3,self.params['preprocess']['input_size'][0],self.params['preprocess']['input_size'][1]))
            self.logger.info(f'Label Shape = {label.shape}')
            assert(label.shape == (self.params['batch_size'][0],self.params['num_class'],1,1))
            self.logger.info(f'Mask Shape = {mask.shape}')
            assert(mask.shape == (self.params['batch_size'][0],self.params['preprocess']['input_size'][0],self.params['preprocess']['input_size'][1]))
            self.logger.info(f'image_id = {image_id}')
            break
#        itr = 0
#        for img,_,_,_ in tqdm.tqdm(self.train_loader):
#            itr = itr + img.shape[0]
#        assert(itr == self.train_loader.dataset.df.shape[0])
        
        self.logger.info("Validation Data.........")
        for img,mask,label,image_id in self.valid_loader:
            self.logger.info(f'Image Shape = {img.shape}')
            assert(img.shape == (self.params['batch_size'][0],3,self.params['preprocess']['input_size'][0],self.params['preprocess']['input_size'][1]))
            self.logger.info(f'Label Shape = {label.shape}')
            assert(label.shape == (self.params['batch_size'][0],self.params['num_class'],1,1))
            self.logger.info(f'Mask Shape = {mask.shape}')
            assert(mask.shape == (self.params['batch_size'][0],self.params['preprocess']['input_size'][0],self.params['preprocess']['input_size'][1]))
            self.logger.info(f'image_id = {image_id}')
            break
#        itr = 0
#        for img,_,_,_ in tqdm.tqdm(self.valid_loader):
#            itr = itr + img.shape[0]
#        assert(itr == self.valid_loader.dataset.df.shape[0])
        
        self.logger.info("Testing Data.........")
        for img,mask,label,image_id in self.test_loader:
            self.logger.info(f'Image Shape = {img.shape}')
            assert(img.shape == (self.params['batch_size'][0],3,self.params['preprocess']['input_size'][0],self.params['preprocess']['input_size'][1]))
            self.logger.info(f'Label Shape = {label.shape}')
            assert(label.shape == (self.params['batch_size'][0],self.params['num_class'],1,1))
            self.logger.info(f'Mask Shape = {mask.shape}')
            assert(mask.shape == (self.params['batch_size'][0],self.params['preprocess']['input_size'][0],self.params['preprocess']['input_size'][1]))
            self.logger.info(f'image_id = {image_id}')
            break
#        itr = 0
#        for img,_,_,_ in tqdm.tqdm(self.test_loader):
#            itr = itr + img.shape[0]
#        assert(itr == self.test_loader.dataset.df.shape[0])
        
        self.logger.info("Passed")