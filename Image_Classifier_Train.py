import argparse
import PIL
import torch
from collections import OrderedDict
from torch import nn
from torch import optim
from torchvision import datasets,transforms
from torchvision import models
def args_mapping():
	train_parser = argparse.ArgumentParser()
	train_parser.add_argument('data_dir', type=str,action="store")

	train_parser.add_argument('--save_dir',action="store",dest="save_dir",default="/home/workspace/aipnd-project/checkpoint2.pth")
	train_parser.add_argument('--arch',action="store",dest="arch",default="vgg16")
	train_parser.add_argument('--learning_rate',type=float,default=0.001,dest="learning_rate",action="store")
	train_parser.add_argument('--epochs',action="store",type=int,dest="epochs",default=5)
	train_parser.add_argument('--hidden_units',action="store",type=int,dest="hidden_units",default=610)
	train_parser.add_argument('--gpu',type=bool,action="store",dest="gpu_ena",default=False)
	train_args = train_parser.parse_args()
	print("Root Data Dir :",train_args.data_dir)
	print("Save Dir :",train_args.save_dir)
	print("Arch :",train_args.arch)
	print("Learning rate :",train_args.learning_rate)
	print("Epochs :",train_args.epochs)
	print("Hidden Units :",train_args.hidden_units)
	print("GPU/DEVICE :",train_args.gpu_ena)
	return train_args
def training_model():
	args_list=args_mapping()
	'''Mapping the training,validation and testing dataset location
	passed from the given data_dir argument and applying the transformations
	like adding randomness,resizing,cropping and normalizing the training,
	validation and test datasets. As a result forming DataLoader 
	out of it'''
	train_dir = args_list.data_dir + '/train'
	valid_dir = args_list.data_dir + '/valid'
	test_dir = args_list.data_dir + '/test'
	train_transforms=transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.RandomRotation(20, resample=PIL.Image.BILINEAR),transforms.RandomGrayscale(p=0.2),transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
	validation_transforms=transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
	'''Load datasets and apply the defined transformations'''
	image_train_datasets = datasets.ImageFolder(train_dir,transform=train_transforms)
	image_validation_datasets=datasets.ImageFolder(valid_dir,transform=validation_transforms)
	image_test_datasets=datasets.ImageFolder(test_dir,transform=validation_transforms)	
	
	dataloaders_train = torch.utils.data.DataLoader(image_train_datasets,batch_size=32,shuffle=True)
	dataloaders_valid = torch.utils.data.DataLoader(image_validation_datasets,batch_size=32,shuffle=True)
	dataloaders_test = torch.utils.data.DataLoader(image_test_datasets,batch_size=32,shuffle=True)

	model_to_train=models.__dict__[args_list.arch](pretrained=True)
	print(model_to_train.classifier.in_features)
	if(args_list.arch=='vgg16'):
		classifier= nn.Sequential(OrderedDict([
    				('first',nn.Linear(model_to_train.classifier[0].in_features,args_list.hidden_units)),
    				('first_relu',nn.ReLU()),
    				('first_drop',nn.Dropout(0.1)),
    				('final',nn.Linear(args_list.hidden_units,102)),
    				('final_log',nn.LogSoftmax(dim=1)),
        			]))	
	elif(args_list.arch=='densenet121'):
		classifier= nn.Sequential(OrderedDict([
				('first',nn.Linear(model_to_train.classifier.in_features,args_list.hidden_units)),
				('first_relu',nn.ReLU()),
    				('first_drop',nn.Dropout(0.1)),
    				('final',nn.Linear(args_list.hidden_units,102)),
				('final_log',nn.LogSoftmax(dim=1)),
        			]))	
	else:
		print("Only vgg16/densenet121 model architecture are supported. Not {}".format(args_list.arch))
	for param in model_to_train.parameters():
        	param.requires_grad=False
	model_to_train.classifier=classifier
	print(model_to_train.classifier)	
	

	if(args_list.gpu_ena==True):
		device='cuda'
	else:
		device='cpu'
	model_to_train.to(device)
	loss=nn.NLLLoss()
	optimizer=optim.Adam(model_to_train.classifier.parameters(),lr=args_list.learning_rate)
	steps=0
	running_loss=0
	print_every=40
	for e in range(args_list.epochs):
		for image,label in iter(dataloaders_train):
			model_to_train.train()
			steps += 1
			image,label=image.to(device),label.to(device)
			optimizer.zero_grad()
			output=model_to_train.forward(image)
			po=loss(output,label)
			po.backward()
			optimizer.step()
			running_loss += po.item()
			if steps % print_every == 0:
				model_to_train.eval()
				with torch.no_grad():
					validation_loss=0
					accuracy=0
					for images,labels in iter(dataloaders_valid):
						images,labels=images.to(device),labels.to(device)
						valid_output=model_to_train.forward(images)
						validation_loss+=loss(valid_output,labels).item()
						ps = torch.exp(valid_output)
						equality = (labels.data == ps.max(dim=1)[1])
						accuracy += equality.type(torch.FloatTensor).mean()


				print("Epoch : {}/{}".format(e+1,args_list.epochs),
				"Training Loss: {:.3f}".format(running_loss/40),
				"Validation Loss: {:.3f}".format(validation_loss/len(dataloaders_valid)),
				"Validation Accuracy: {:.3f}%".format(accuracy/len(dataloaders_valid)*100))
				running_loss=0
	
	if(args_list.arch == 'vgg16'):
		checkpoint = {'architecture':args_list.arch,'input_units':model_to_train.classifier[0].in_features,'output_units':102,'hidden_units':args_list.hidden_units,'state_dict': model_to_train.state_dict()}	
	else:
		checkpoint = {'architecture':args_list.arch,'input_units':model_to_train.classifier[0].in_features,'output_units':102,'hidden_units':args_list.hidden_units,'state_dict': model_to_train.state_dict()}
	print("{},{},{},{}".format(checkpoint['architecture'],checkpoint['input_units'],checkpoint['output_units'],checkpoint['hidden_units']))
	torch.save(checkpoint,args_list.save_dir)

training_model()
