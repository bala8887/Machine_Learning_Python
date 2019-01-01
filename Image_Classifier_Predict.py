import json
import argparse
from torchvision import datasets,transforms
from torch.autograd import Variable 
import numpy as np
from PIL import Image
import torch
import tensorflow as tf
from torch import nn
from collections import OrderedDict
from torchvision import models
import torch
def args_mapping():
          predict_parser = argparse.ArgumentParser()
          predict_parser.add_argument('path_to_image', type=str,action="store")
          predict_parser.add_argument('checkpoint_path', type=str,action="store")
          predict_parser.add_argument('--top_k',action="store",dest="top_k",type=int,default=3)
          predict_parser.add_argument('--category_names',action="store",dest="category_names",default="cat_to_name.json")
          predict_parser.add_argument('--gpu',type=bool,action="store",dest="gpu_ena",default=False)
          predict_args = predict_parser.parse_args()
          print("Path to Image :",predict_args.path_to_image)
          print("Checkpoint :",predict_args.checkpoint_path)
          print("Top n  :",predict_args.top_k)
          print("Category Names :",predict_args.category_names)
          print("GPU/DEVICE :",predict_args.gpu_ena)
          return predict_args
def process_image(image):
#image="flowers/train/1/image_06734.jpg"
          im = Image.open(image)
          resized_image=im.resize((256,256))
          width, height = resized_image.size   # Get dimensions
          new_width,new_height=224,224
          left = (width - new_width)/2
          top = (height - new_height)/2
          right = (width + new_width)/2
          bottom = (height + new_height)/2
          cropped_image=resized_image.crop((left, top, right, bottom))
          np_image=np.asarray(cropped_image)
          np_image=np_image/255
          mean = np.array([0.485, 0.456, 0.406])
          std = np.array([0.229, 0.224, 0.225])
          final_np=(np_image-mean)/std
          c=final_np.transpose(2,0,1)
          return c
def predicting_model():
          args_list=args_mapping()
          checkpoint=torch.load(args_list.checkpoint_path)
          print(checkpoint['architecture'])
          if(checkpoint['architecture'] == 'vgg16'):
                model_to_predict=models.vgg16(pretrained=True)
                classifier= nn.Sequential(OrderedDict([
                                ('first',nn.Linear(checkpoint['input_units'],checkpoint['hidden_units'])),
                                ('first_relu',nn.ReLU()),
                                ('first_drop',nn.Dropout(0.1)),
                                ('final',nn.Linear(checkpoint['hidden_units'],102)),
                                ('final_log',nn.LogSoftmax(dim=1)),
                                ]))
          else:
                model_to_predict=models.densenet121(pretrained=True)
                classifier= nn.Sequential(OrderedDict([
                                ('first',nn.Linear(checkpoint['input_units'],checkpoint['hidden_units'])),
                                ('first_relu',nn.ReLU()),
                                ('first_drop',nn.Dropout(0.1)),
                                ('final',nn.Linear(checkpoint['hidden_units'],102)),
                                ('final_log',nn.LogSoftmax(dim=1)),
                                ]))
          for param in model_to_predict.parameters():
              param.requires_grad=False
          model_to_predict.classifier=classifier
          model_to_predict.load_state_dict(checkpoint['state_dict'])
          
          processed_image=process_image(args_list.path_to_image)
          trans_apply=transforms.Compose([transforms.ToTensor()])
          tens=trans_apply(processed_image)
          tens=tens.permute(1,0,2)
          tens=Variable(tens.unsqueeze(0))
          with torch.no_grad():
               model_to_predict.eval()
               output=model_to_predict.forward(tens.float())
               output_prob=torch.exp(output).data
               a,b=output_prob.topk(args_list.top_k)
               prob=a.numpy() 
               output=b.numpy()
               prob.resize(args_list.top_k)
               output.resize(args_list.top_k)
               data_test_transforms = transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
               if("train" in args_list.path_to_image):
                      dir_consider='flowers/train'
               elif("valid" in args_list.path_to_image):
                      dir_consider='flowers/valid'
               else:
                      dir_consider='flowers/test'
               image_validation_datasets=datasets.ImageFolder(dir_consider,transform=data_test_transforms)
               model_to_predict.class_to_idx=image_validation_datasets.class_to_idx
               outp_classes=[]
               for i in output:
                     for k,v in model_to_predict.class_to_idx.items():
                         if(v == i):
                            outp_classes.append(k)
               with open(args_list.category_names,'r') as f:
                    cat_to_name = json.load(f)
               class_names=[cat_to_name.get(i) for i in outp_classes]               
               for i in range(args_list.top_k):
                    print("{} prediction class:{} , probability:{}".format(i+1,outp_classes[i],prob[i]))
              

               
          
           

predicting_model()
