import torch.nn as nn
import torchvision.models as models

class MyModel(nn.Module):
    
    def __init__(self, base_model):
        super(MyModel, self).__init__()
        self.model_dict = {"my_model_class_1": models.resnet50(pretrained=False),
                           "my_model_class_2": models.resnet152(pretraind=False)}
        
        mymodel = self._get_basemodel(base_model)
        self.features = nn.Sequential(*list(mymodel.children()))[:-1]
        
    def _get_basemodel(self, model_name):
        try:
            model = self.model_dict[model_name]
            print("Feature extractor:", model_name)
            return model
        except:
            raise ("Invalid model name. Check the config file")
    
    def forward(self, x):
        h = self.features(x)
        h = h.squeeze()
        return h
    