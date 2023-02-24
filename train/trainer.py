class Trainer(object):
    
    def __init__(self, dataset, base_model, config):
        self.dataset = dataset
        self.base_model = base_model
        self.config = config
        self.device = self._get_device()
        self.loss = nn.CrossEntropyLoss()
        self.model_dict = {'mymodel': MyModel}
        
    def _get_decive(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Running on:", device)
        return device
    
    def _get_model(self):
        try:
            model = self.model_dict[self.base_model]
            return model
        except:
            raise ("Invalid model name. Pass one of the model dictionary")
        
    def _validate(self, epoch, net, criterion, valid_loader, best_acc):
        
        ### 생략
        
        # Save checkpoint
        model_checkpoints_folder = os.path.join('./weights', 'checkpoints')
        
        acc = 100. * correct / total 
        if acc > best_acc:
            print('Saving..')
            state = {
                'net' : net.state_dict(),
                'acc' : acc,
                'epoch' : epoch,
            }
            torch..save(state, os.path.join(model_checkpoints_folder, 'model.pyh'))
            best_acc = acc
            
        net.train()
        return valid_loss, best_acc
    
    def _load_pre_trained_weights(self, model):
        
        ### 생략
        
        return model, best_acc, start_epoch
    
    def train(self):
        train_loader, test_loader = self.dataset.get_data_loaders()
        
        model = self._get_model()
        model = model(**self.config['model'])
        model, best_acc, start_epoch = self._load_pre_trained_weights(model)
        model = model.to(self.device)
        model.train()
        
        criterion = self.loss.to(self.device)
        optimizer = optim.SGD(model.parameters(), lr=self.config['learning_rate'], momentum=0.9)
        
        # save config file
        model_checkpoints_folder = os.path.join('./weights', 'checkpoints')
        _save_config_file(model_checkpoints_folder, str(sefl.base_model))
        
        history = {}
        history['train_lass'] = []
        history['valid+loss'] = []
        
        epochs = self.config['epochs']
        for e in range(start_epoch, start_epoch+epochs):
            h = np.array([])
            
            train_loss = 0
            correct = 0
            total = 0
            
            for i, (inputs, labels) in enumerate(train_loader, 0):
                
                ### 생략
                
            history['train_loss'].append(train_loss)
            history['valid_loss'].append(valid_loss)
            
            plt.figure(figsize=(15,10))
            
            ### 생략
            
            plt.savefig('./weights/checkpoints/loss.png')
            plt.close()
            
        # copy and save trained model with config to experiments dir.
        _copy_to_experiment_dir(model_checkpoints_folder, str(self.base_model))
            