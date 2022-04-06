# coding=utf-8
import torch.nn as nn
from skimage import io
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from data_loader import RescaleT
from data_loader import ToTensorLab
from data_loader import SalObjDataset
from model.DFINet import DFINet
import glob
import timeit
import os
import sys

def normPRED(d):
	ma = torch.max(d)
	mi = torch.min(d)
	dn = (d-mi)/(ma-mi)
	return dn


def save_output(image_name, pred, d_dir):
	predict = pred
	predict = predict.squeeze()
	predict_np = predict.cpu().data.numpy()
	im = Image.fromarray(predict_np*255).convert('RGB')
	image = io.imread(image_name)
	imo = im.resize((image.shape[1], image.shape[0]), resample=Image.BILINEAR)
	img_name = image_name.split("/")[-1]       
	imidx = img_name.split(".")[0]
	imo.save(d_dir+imidx+'.png')


def main():
	# ---------创建模型文件夹
	args ={'root':'./trained_model/','ext':'.pth'}#存放训练模型的地址
	Root=glob.glob(args['root']+'*'+args['ext'])
	
	for p in Root:
		name = p.split("/")[-1]
		name = name.split(".")[0]
		path1='//'#保存路径
		path2=os.path.join(path1,name)
		os.mkdir(path2)#创建文件夹
	# --------- 1. get image path and name ---------
	folder_root='//'   #保存路径
	image_dir = '//'              # path of testing dataset
	img_name_list = glob.glob(image_dir + '*.bmp')
    
	# --------- 2. dataloader ---------
	test_salobj_dataset = SalObjDataset(img_name_list=img_name_list, lbl_name_list=[],
										transform=transforms.Compose([RescaleT(256), ToTensorLab(flag=0)]))
	test_salobj_dataloader = DataLoader(test_salobj_dataset, batch_size=1, shuffle=False, num_workers=5)

	# --------- 3. model define ---------
	print("...load DFINet...")
	
	for pp in Root:
		net =nn.DataParallel(DFINet(in_channels=3))
		net.load_state_dict(torch.load(pp))
		name_folder = pp.split("/")[-1]
		name_folder = name_folder.split(".")[0]
		path_folder = os.path.join(folder_root,name_folder)
		path_folder = os.path.join(path_folder,'')
		print(path_folder)
		net.load_state_dict(torch.load(pp))
		net.cuda()
		net.eval()

		start = timeit.default_timer()
	# --------- 4. inference for each image ---------
		with torch.no_grad():
			for i_test, data_test in enumerate(test_salobj_dataloader):
			    print("inferencing:", img_name_list[i_test].split("/")[-1])
			    inputs_test = data_test['image']
			    inputs_test = inputs_test.type(torch.FloatTensor)
			    inputs_test = inputs_test.cuda()
			#s_out, s0, s1, s2, s3, s4, sb = net(inputs_test)
			    s_out,s0,s1,s2,s3,s4= net(inputs_test)
			# normalization
			    pred = s_out[:, 0, :, :]
			    pred = normPRED(pred)

			# save results to test_results folder
			    save_output(img_name_list[i_test], pred, path_folder)
			#del s_out, s0, s1, s2, s3, s4, sb
			    del s_out,s0,s1,s2,s3,s4
		end = timeit.default_timer()
		print(str(end-start))


if __name__ == "__main__":
	main()

