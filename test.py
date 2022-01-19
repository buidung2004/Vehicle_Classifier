import efficientnet.tfkeras
from tensorflow.keras.models import load_model
class SwishActivation(Activation):
    
    def __init__(self, activation, **kwargs):
        super(SwishActivation, self).__init__(activation, **kwargs)
        self.__name__ = 'swish_act'

def swish_act(x, beta = 1):
    return (x * sigmoid(beta * x))

model = load_model('D:/UIT/Docker/my_webapp/models/model_finetune_40epoch_Efficientnet_focalloss.h5',custom_objects= {"swish_act":swish_act})