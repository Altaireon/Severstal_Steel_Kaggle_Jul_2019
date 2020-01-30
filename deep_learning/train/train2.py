from deep_learning.lib.utility import *

class Train2:
    def __init__(self,loader,params):
        super(Train2,self).__init__()
        self.logger = logging.getLogger(params['logger_name']+'.train')
        self.loader = loader
        self.params = params
        self.model = smp.FPN(encoder_name=self.params['model']['encoder'],encoder_weights=self.params['model']['encoder_weights'],classes=self.params['num_class'],activation=self.params['model']['activation'])
        
    def run_check_net(self):
        pass
    
    def process_train_classification(self):
        pass
    
    def process_valid_classification(self):
        pass
    
    def __clean__(self):
        out_dir = LOG_DIR + 'model-' + self.params['id'] + '/'
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)
        os.mkdir(out_dir)
        self.logger.debug(f'{out_dir} created..')
        
        model_dir = out_dir + 'checkpoint/'
        if os.path.exists(model_dir):
            shutil.rmtree(model_dir)
        os.mkdir(model_dir)
        self.logger.debug(f'{model_dir} created..')
    
    def process_train_segmentation(self):
        
        self.__clean__()
        loss = smp.utils.losses.BCEDiceLoss(eps=1.,activation=self.params['model']['activation'])
        metrics = [
            smp.utils.metrics.IoUMetric(eps=1.),
            smp.utils.metrics.FscoreMetric(eps=1.),
        ]
        savePath = LOG_DIR+'model-'+self.params['id']+'/checkpoint/'
        en_params = self.params['model']['en_params']
        de_params = self.params['model']['de_params']
        en_lr = iter(en_params['lr'].keys())
        en_ep = iter(en_params['lr'].values())
        de_lr = iter(de_params['lr'].keys())
        de_ep = iter(de_params['lr'].values())
        optimizer = torch.optim.Adam([
            {'params': self.model.decoder.parameters(), 'lr': float(next(de_lr))}, 
            {'params': self.model.encoder.parameters(), 'lr': float(next(en_lr))},  
        ])
        train_epoch = smp.utils.train.TrainEpoch(
            self.model, 
            loss=loss, 
            metrics=metrics, 
            optimizer=optimizer,
            device=self.params['device'],
            verbose=True,
        )
        
        valid_epoch = smp.utils.train.ValidEpoch(
            self.model, 
            loss=loss, 
            metrics=metrics, 
            device=self.params['device'],
            verbose=True,
        )

        max_score = 0
        
        en_iter = next(en_ep)
        de_iter = next(de_ep)
        print(savePath)
        for i in range(0, self.params['num_epochs']):
            
            self.logger.info('\nEpoch: {}'.format(i))
            train_logs = train_epoch.run(self.loader.train_loader)
            valid_logs = valid_epoch.run(self.loader.valid_loader)
            self.logger.info(train_logs)
            self.logger.info(valid_logs)
            if max_score < valid_logs['f-score']:
                max_score = valid_logs['f-score']
                torch.save(self.model, f'{savePath}_{i}_{max_score}_best_model.pth')
                self.logger.info('Model saved!')
                
            if i == de_iter-1:
                x=float(next(de_lr))
                optimizer.param_groups[0]['lr'] = x
                self.logger.info(f'Decrease decoder learning rate to {x}')
                
            if i == en_iter-1:
                x=float(next(en_lr))
                optimizer.param_groups[1]['lr'] = x
                self.logger.info(f'Decrease encoder learning rate to {x}')
        

        
    
    