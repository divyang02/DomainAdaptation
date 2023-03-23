import torch
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
learning_rate = 2e-5
epochs = 3
# train_batch_size = 64
# valid_batch_size = 64
# test_batch_size = 128
# unlabeled_batch_size = 128

train_batch_size = 1
valid_batch_size = 1
test_batch_size = 1
unlabeled_batch_size = 2
dropout_rate = 0.3
Temp = 1
kd_alpha = 0.5
class_num = 2
bidirectional = True

#Whether to use data aug and what is the ratio
noise_injection = True
noise_injection_rate = 0.4

# Define hyperparameters
train_params = {'batch_size': train_batch_size,
                'shuffle':True,
                'num_workers':0
               }
valid_params = {'batch_size':valid_batch_size,
                'shuffle':False,
                'num_workers':0
               }
test_params = {'batch_size':test_batch_size,
                'shuffle':False,
                'num_workers':0
               }
unlabeled_params = {'batch_size':unlabeled_batch_size,
                    'shuffle':True,
                    'num_workers':0
                   }
