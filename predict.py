import numpy as np
import pandas as pd
import sys
import math
import os.path as osp
import torch
from torch_geometric.data import Data
from torch_geometric.data import Dataset
from torch_geometric.data import InMemoryDataset
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import DataLoader
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import GCNConv
from torch_geometric.nn import DataParallel
from torch import nn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
from ovito.io import import_file
from ovito.io import export_file
import datetime
starttime = datetime.datetime.now()
#read file
file_path = './predict/z_indent/average_50_*.dump'
node = import_file(file_path)

def voted(data, model1, model2, model3):

	predict1 = model1(data).max(1)[1]
	predict1 = predict1.cpu().numpy()
	predict1 = np.reshape(predict1, (-1,1))
	
	predict2 = model2(data).max(1)[1]
	predict2 = predict2.cpu().numpy()
	predict2 = np.reshape(predict2, (-1,1))
	
	predict3 = model3(data).max(1)[1]
	predict3 = predict3.cpu().numpy()
	predict3 = np.reshape(predict3, (-1,1))

	predict = np.column_stack((predict1, predict2, predict3))
	result = np.zeros((predict.shape[0],1))
	for j in range(predict.shape[0]) :
		result[j] = np.argmax(np.bincount(predict[j]))
	result = np.reshape(result,(1,-1))
	result = pd.DataFrame(result)
	return result

def compute_myproperty(frame, data, result):
	variant = result.to_numpy()
	variant = variant.reshape(-1)
	variant = variant.tolist()
	return variant

class PredictDataset(InMemoryDataset):
	def __init__(self, root, transform=None, pre_transform=None):
		super(PredictDataset, self).__init__(root, transform, pre_transform)
		self.data, self.slices = torch.load(self.processed_paths[0])

	@property
	def raw_file_names(self):
		return []
	
	@property
	def processed_file_names(self):
		return ['/data2/GCN_dataset/dataset_root/predict_set.pt']
	
	def download(self):
		pass

	def process(self):
		data_list = []

		for i in range(len(predict_content)) :
			content = predict_content[i]
			graph = predict_graph[i]
			node_feature = content[:]

			node_feature = torch.FloatTensor(node_feature).squeeze(1)
			source_nodes = graph[:,0]
			target_nodes = graph[:,1]
			edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
			x = node_feature
			num_nodes = x.shape[0]
			data = Data(x=x, edge_index=edge_index, num_nodes = num_nodes)
			data_list.append(data)
		data, slices = self.collate(data_list)
		torch.save((data, slices), self.processed_paths[0])

predict_set = PredictDataset(root='/data2/GCN_dataset/dataset_root/')

class GSGNet(torch.nn.Module):
	def __init__(self):
		super(GSGNet, self).__init__()
		self.conv1 = SAGEConv(predict_set.num_node_features, 40)
		self.conv2 = SAGEConv(40, 24)

	def forward(self, data):
		x, edge_index = data.x, data.edge_index

		x = self.conv1(x, edge_index)
		x = F.relu(x, inplace = True)
		x = F.dropout(x, p = 0.5,training=self.training)
		x = self.conv2(x, edge_index)
		#return x
		return F.log_softmax(x, dim=1)		

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model1 = GSGNet()
model1_checkpoint = torch.load('./Trained_Model/train_model_gsg_1.pth.tar')
model1.load_state_dict(model1_checkpoint['state_dict'])
model1.to(device)
model1.eval()

model2 = GSGNet()
model2_checkpoint = torch.load('./Trained_Model/train_model_gsg_2.pth.tar')
model2.load_state_dict(model2_checkpoint['state_dict'])
model2.to(device)
model2.eval()

model3 = GSGNet()
model3_checkpoint = torch.load('./Trained_Model/train_model_gsg_3.pth.tar')
model3.load_state_dict(model3_checkpoint['state_dict'])
model3.to(device)
model3.eval()

predict_loader = DataLoader(predict_set, batch_size = 1,shuffle=False)
i = 0
for data in predict_loader :
	data = data.to(device)
	pipe = node.compute(i)
	predict = voted(data, model1, model2, model3)
	result = compute_myproperty(i, data, predict)
	pipe.particles_.create_property('Variants', data = result)
	export_file(pipe, './predict/z_indent/predict_%s.dump'%i, 'lammps/dump',columns = ['Particle Identifier','Particle Type','Position.X','Position.Y','Position.Z','Variants'],frame = i)
	i = i + 1
print('Done')
endtime = datetime.datetime.now()
print (endtime - starttime)
