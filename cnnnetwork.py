
import matplotlib.pyplot as plt
from PIL import Image

import numpy as np
import datetime

import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from torch.autograd import Variable
from collections import OrderedDict
# --------------------

# -------------------------------------------------------------------
# -------------- class Neural Network -------------------------------
# -------------------------------------------------------------------
class CNNNetwork(nn.Module):
# ------------ init --------------------------------------
    def __init__(self, model_name, train_on_gpu,e=10,flag='Train'):
        ''' initialize the model and all train/test/valid dataloaders '''
        super(CNNNetwork, self).__init__()
        print("cnn neural network bbb... ")
        print("load image data ... ", end="")
        # define transforms for the training data and testing data
        if flag=='Train':
          self.train_transforms = transforms.Compose([transforms.RandomRotation(30),
#                                        transforms.Scale(299),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

          self.test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
#                                       transforms.Scale(299),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])

          self.validation_transforms = transforms.Compose([transforms.Resize(256),
                                            transforms.CenterCrop(224),
#                                             transforms.Scale(299),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], 
                                                                 [0.229, 0.224, 0.225])])
        self.num_class=102
        self.hidden=500
        
        # pass transforms in here, then run the next cell to see how the transforms look
        # TODO: Load the datasets with ImageFolder
        if flag=='Train':
          self.data_dir = 'flower_data'
          self.train_dir = self.data_dir + '/train'
          self.valid_dir = self.data_dir + '/valid'
          self.test_dir = self.data_dir + '/test'
          self.train_data = datasets.ImageFolder(self.train_dir, transform=self.train_transforms)
          self.validation_data = datasets.ImageFolder(self.valid_dir, transform=self.validation_transforms)
          self.test_data = datasets.ImageFolder(self.test_dir ,transform = self.test_transforms)

          # TODO: Using the image datasets and the trainforms, define the dataloaders
          self.train = torch.utils.data.DataLoader(self.train_data, batch_size=64, shuffle=True)
          self.validation = torch.utils.data.DataLoader(self.validation_data, batch_size =32,shuffle = True)
          self.test = torch.utils.data.DataLoader(self.test_data, batch_size = 20, shuffle = True)
          print("done")
        
          self.class_to_idx = self.test_data.class_to_idx
        
        print("create model aaaaaa... ", end="")
        print('asdasd')
        
        if model_name == 'vgg16':
            self.model = models.vgg16(pretrained=True)

            # Freeze early layers
            for param in self.model.parameters():
                param.requires_grad = False
            self.n_inputs = self.model.classifier[6].in_features

            # Add on classifier
            self.model.classifier[6] = nn.Sequential(
                nn.Linear(self.n_inputs, 256), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(256, self.num_class), nn.LogSoftmax(dim=1))

        elif model_name == 'resnet50':
            self.model = models.resnet50(pretrained=True)

            for param in self.model.parameters():
                param.requires_grad = False

            self.n_inputs = self.model.fc.in_features
            self.model.fc = nn.Sequential(
                nn.Linear(self.n_inputs, 256), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(256, self.num_class), nn.LogSoftmax(dim=1))

        # Move to gpu and parallelize
        if train_on_gpu:
            self.model = self.model.to('cuda')
        
        
        
       
        
        # train a model with a pre-trained network
        self.criterion = nn.NLLLoss()
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=0.001 )
        
        if flag=='Train':
          self.model.class_to_idx = self.train_data.class_to_idx
          self.model.idx_to_class = {
            idx: class_
            for class_, idx in self.model.class_to_idx.items()
          }
        self.model.epochs=e
        self.model.optimizer = self.optimizer
        
            
        print("initialized.")



  



    def do_deep_learning(self, print_every,is_gpu):
        epochs=self.model.epochs
        ''' train the model based on the train-files '''
        if is_gpu:
            print("start deep-learning in -gpu- mode ... ")
        else:
            print("start deep-learning in -cpu- mode ... ")
        
        epochs = epochs
        print_every = print_every
        steps = 0
        loss_show=[]
    
        # change to cuda in case it is activated
        if is_gpu:
            self.model.cuda()
    
        self.model.train() # ---------- put model in training mode -------------------
        
        for e in range(epochs):
            running_loss = 0
            for ii, (inputs, labels) in enumerate(self.train):
                self.model.train()
                steps += 1

                inputs,labels = inputs.cuda(), labels.cuda()
                #inputs = Variable(inputs, requires_grad=True).cuda()
                #labels = Variable(labels, requires_grad=True).cuda()

                self.optimizer.zero_grad()

                # Forward and backward passes
                outputs = self.model.forward(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

                if steps % print_every == 0:
                    self.model.eval()
                    vlost = 0
                    accuracy=0


                    for ii, (inputs2,labels2) in enumerate(self.validation):
                        self.optimizer.zero_grad()

                        inputs2, labels2 =inputs2.cuda() , labels2.cuda()
                        #inputs2 = Variable(inputs2, requires_grad=True).cuda()
                        self.model.cuda()
                        with torch.no_grad():    
                            outputs = self.model.forward(inputs2)
                            vlost = self.criterion(outputs,labels2)
                            ps = torch.exp(outputs).data
                            equality = (labels2.data == ps.max(1)[1])
                            accuracy += equality.type_as(torch.FloatTensor()).mean()

                    vlost = vlost / len(self.validation)
                    accuracy = accuracy /len(self.validation)



                    print("Epoch: {}/{}... ".format(e+1, epochs),
                          "Loss: {:.4f}".format(running_loss/print_every),
                          "Validation Lost {:.4f}".format(vlost),
                           "Accuracy: {:.4f}".format(accuracy))


                    running_loss = 0
        print("-- done --")
    
    
    # implement a function for the validation pass
    def validation(self, is_gpu):
        ''' calculate the validation based on the valid-files and return the test-loss and the accuracy '''
        test_loss = 0
        accuracy = 0
        
        if is_gpu:
            self.model.cuda()
            
        for images, labels in self.validation:
            if is_gpu:
                images, labels = images.cuda(), labels.cuda()
                    
            output = self.model( images )
            test_loss += self.criterion( output, labels ).item()
    
            ps = torch.exp( output )
            equality = ( labels.data == ps.max(dim=1)[1] ) # give the highest probability
            accuracy += equality.type( torch.FloatTensor ).mean()
        
        return test_loss, accuracy
        
    
    
    def check_accuracy_on_test(self, is_gpu):
        ''' calculate the accuracy based on the test-files and print it out in percent '''
        print("calculate accuracy on test ... ", end="")
        correct = 0
        total = 0
        
        if is_gpu:
            self.model.cuda()
        
        self.model.eval() # ------------- put model in evaluation mode ----------------
        
        with torch.no_grad():
            for data in self.test:
                images, labels = data
                
                if is_gpu:
                    images, labels = images.cuda(), labels.cuda()
                        
                outputs = self.model( images )
                _, predicted = torch.max( outputs.data, 1 )
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        print("done.")
        print('accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
  
    def predict(self,image_path, model, topk=5):
        model.eval()

        
        model.cpu()

        image = process_image(image_path)
        image = image.unsqueeze(0)

        with torch.no_grad():
          
          output = model.forward(image)
          top_prob, top_labels = torch.topk(output, topk)

              # Calculate the exponentials
          top_prob = top_prob.exp()

        class_to_idx_inv = {model.class_to_idx[k]: k for k in model.class_to_idx}
        mapped_classes = list()

        for label in top_labels.numpy()[0]:
          
          mapped_classes.append(class_to_idx_inv[label])

        return top_prob.numpy()[0], mapped_classes
   
        
    def load_checkpoint(self,path):
        """Load a PyTorch model checkpoint

        Params
        --------
            path (str): saved model checkpoint. Must start with `model_name-` and end in '.pth'

        Returns
        --------
            None, save the `model` to `path`

        """

        # Get the model name
        model_name = path.split('-')[0]
        assert (model_name in ['vgg16', 'resnet50'
                               ]), "Path must have the correct model name"

        # Load in checkpoint
        checkpoint = torch.load(path, map_location='cpu')

        if model_name == 'vgg16':
            self.model = models.vgg16(pretrained=True)
            # Make sure to set parameters as not trainable
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.classifier = checkpoint['classifier']

        elif model_name == 'resnet50':
            self.model = models.resnet50(pretrained=True)
            # Make sure to set parameters as not trainable
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.fc = checkpoint['fc']

        # Load in the state dict
        self.model.load_state_dict(checkpoint['state_dict'])

        total_params = sum(p.numel() for p in self.model.parameters())
        print(f'{total_params:,} total parameters.')
        total_trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f'{total_trainable_params:,} total gradient parameters.')

        #train_on_gpu=True
        #if train_on_gpu:
        #    self.model = self.model.to('cuda')

        # Model basics
        self.model.class_to_idx = checkpoint['class_to_idx']
        self.model.idx_to_class = checkpoint['idx_to_class']
        self.model.epochs = checkpoint['epochs']

        # Optimizer
        self.optimizer = checkpoint['optimizer']
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        return self.model, self.optimizer
    
    

    def save_checkpoint(self,path='resnet50-transfer-4.pth'):
        #print(p)
        model_name = path.split('-')[0]
        assert (model_name in ['vgg16', 'resnet50'
                               ]), "Path must have the correct model name"

        # Basic details
        checkpoint = {
            'class_to_idx': self.model.class_to_idx,
            'idx_to_class': self.model.idx_to_class,
            'epochs': self.model.epochs,
        }

        # Extract the final classifier and the state dictionary
        if model_name == 'vgg16':

            checkpoint['classifier'] = self.model.classifier
            checkpoint['state_dict'] = self.model.state_dict()

        elif model_name == 'resnet50':

            checkpoint['fc'] = self.model.fc
            checkpoint['state_dict'] = self.model.state_dict()

        # Add the optimizer
        checkpoint['optimizer'] = self.model.optimizer
        checkpoint['optimizer_state_dict'] = self.model.optimizer.state_dict()

        # Save the data to the path
        torch.save(checkpoint, path)


# -------------------------------------------------------------------
# -------------- helper functions -----------------------------------
# -------------------------------------------------------------------
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    expects_means = [0.485, 0.456, 0.406]
    expects_std = [0.229, 0.224, 0.225]
           
    pil_image = Image.open(image).convert("RGB")
    
    # Any reason not to let transforms do all the work here?
    in_transforms = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(expects_means, expects_std)])
    pil_image = in_transforms(pil_image)

    return pil_image


def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

# -------------------------------------------------------------------