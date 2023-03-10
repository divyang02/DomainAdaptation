import torch
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
learning_rate = 2e-5
epochs = 5
outer_epochs = 10
inner_epochs = 3
train_batch_size = 16
valid_batch_size = 16
test_batch_size = 16
unlabeled_batch_size = 16
class_num = 2
bidirectional = True

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