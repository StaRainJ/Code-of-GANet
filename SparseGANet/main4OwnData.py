import torch
import numpy as np
import os
import cv2
from six.moves import xrange
from loss import MatchLoss
from evaluation import eval_nondecompose, eval_decompose
from utils import tocuda, get_pool_result
import pickle



from config import get_config, print_usage
config, unparsed = get_config()
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'
import torch.utils.data
import sys
from data import collate_fn, CorrespondencesDataset
from oan import OANet as Model
from train import train
from test import test

import os
import scipy.io as sio
import numpy as np
import datetime
import time

print("-------------------------Deep Essential-------------------------")
print("Note: To combine datasets, use .")

def create_log_dir(config):
    if not os.path.isdir(config.log_base):
        os.makedirs(config.log_base)
    if config.log_suffix == "":
        suffix = "-".join(sys.argv)
    result_path = config.log_base+'/'+suffix
    if not os.path.isdir(result_path):
        os.makedirs(result_path)
    if not os.path.isdir(result_path+'/train'):
        os.makedirs(result_path+'/train')
    if not os.path.isdir(result_path+'/valid'):
        os.makedirs(result_path+'/valid')
    if not os.path.isdir(result_path+'/test'):
        os.makedirs(result_path+'/test')
    if os.path.exists(result_path+'/config.th'):
        print('warning: will overwrite config file')
    torch.save(config, result_path+'/config.th')
    # path for saving traning logs
    config.log_path = result_path+'/train'






def test(data_loader, model, config):  # model is class form from oan.py-->  model = Model(config)
    save_file_best = os.path.join(config.model_path, 'model_best.pth')
    if not os.path.exists(save_file_best):
        print("Model File {} does not exist! Quiting".format(save_file_best))
        exit(1)
    # Restore model
    checkpoint = torch.load(save_file_best) # 
    print("******************************************************************* ")    
    state_dict = checkpoint['state_dict']
    accuary = checkpoint['best_acc']
    print("best_acc = " + str(accuary))
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():  ## ???
        name = k[7:]
        new_state_dict[name] = v
    start_epoch = checkpoint['epoch']
    model.load_state_dict(new_state_dict)

    model.cuda()
    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    print("network have {} paramerters in total".format(num_params / 1e6))
    #model = torch.nn.DataParallel(model, device_ids=[0,1])
    start_epoch = checkpoint['epoch']
    print("Restoring from " + str(save_file_best) + ', ' + str(start_epoch) + "epoch...\n" + "best_acc" + str(accuary))
    if config.res_path == '':
        config.res_path = config.model_path[:-5]+'test'
    print('save result to '+config.res_path)
    p,r,indx,time = test_process("test", model, 0, data_loader, config)
    return p,r,indx,time
    #print('test result '+str(va_res))
    
    #ofn = os.path.join(config.res_path, "params.txt")
    #np.savetxt(ofn, np.array(num_params))



def test_process(mode, model, cur_global_step, data_loader, config):
       results, pool_arg = [], []
       yhat_list = []
       ehat_list = []
       eval_step, eval_step_i, num_processor = 100, 0, 8

       test_data = data_loader        
       #print("test_data0 = ",test_data)      
       #print(test_data['xs'].shape)   1x1x2000x4
       test_data = tocuda(test_data)
       #print("test_data= ",test_data)
       
       print("Runing our model...")
       time_start = datetime.datetime.now()
       res_logits, res_e_hat = model(test_data)
       time_end = datetime.datetime.now()
       time_diff = time_end - time_start
       print("Runtime in milliseconds: {}".format(
         float(time_diff.total_seconds() * 1000.0)
    ))
    
       y_hat, e_hat = res_logits[-1], res_e_hat[-1]  #### core print 
       #print("y_hat= ",y_hat)
       #print("e_hat= ",e_hat)
       
       c = torch.tensor(y_hat).squeeze()
       num_save = (c > 0).float().sum().int().item()
       print('num_save',num_save)
       indx, value = torch.topk(c,num_save)
       value,_ = value.sort()
       value += 1
       #print('value',value)
       indx = value.cpu().numpy().tolist()
       GT = data_loader['GT']
       GT = torch.tensor(GT).long().squeeze()
       intersection = [v for v in value if v in GT]
       intersection = torch.tensor(intersection)
       if num_save < 8:
           p = GT/c.size(0)
           r = 1
       else:
           p = intersection.size(0)/num_save
           r = intersection.size(0)/GT.size(0)
        
       mask = (c > 1).type(torch.IntTensor)
       #print("mask= ",mask)
       print("Pre= ",p)
       print("Rec= ",r)
       yhat_list += [mask.detach().cpu().numpy()]
       #print("yhat_list= ",yhat_list)
       print("Ending test!")              
       return p,r,indx,float(time_diff.total_seconds() * 1000.0) 
       
# norm the imput with T 
def norm_input(x):
    x_mean = np.mean(x, axis=0)
    dist = x - x_mean
    meandist = np.sqrt((dist**2).sum(axis=1)).mean()
    scale = np.sqrt(2) / meandist
    T = np.zeros([3,3])
    T[0,0], T[1,1], T[2,2] = scale, scale, 1
    T[0,2], T[1,2] = -scale*x_mean[0], -scale*x_mean[1]
    x = x * np.asarray([T[0,0], T[1,1]]) + np.array([T[0,2], T[1,2]])
    #print('data33norm_input')
    #print('normx =', x)
    return x, T

            
def load_data(var_name,DataDir):
    """Main data loading routine"""
    print("******************************************************************* ")  
    print("Loading test data")

    data = {}

    #load mat data
    #cur_folder=os.getcwd()+DataDir
    cur_folder=DataDir
    # var_name="churchLPM.mat"
    inf = {}
    in_file_name = os.path.join(cur_folder, var_name) 
    with open(in_file_name, "rb") as fr:
        inf=sio.loadmat(fr)
    x=np.hstack((inf['X'],inf['Y']))
    #inf['CorrectIndex'] = np.ones(((np.shape(x))[0],1))
    if not ('CorrectIndex' in locals().keys()): 
        inf['CorrectIndex'] = range(1, (np.shape(x))[0]+1)
    print(inf['CorrectIndex'])
#    if inf['CorrectIndex'] is None:
#        inf['CorrectIndex'] = np.ones(((np.shape(x))[1],1))


    #normalization 
#    length=np.max(inf['X'])
#    weight=np.max(inf['Y'])
#    x=x-np.array([[(length-1)/2+1,(weight-1)/2+1,(length-1)/2+1,(weight-1)/2+1]])
#    x=x/np.array([[(length-1)/2,(weight-1)/2,(length-1)/2,(weight-1)/2]])

    T1, T2 = np.zeros(1), np.zeros(1)
            
    x1, x2 = x[:,:2], x[:,2:4]
    #print('x1',x1,'x2',x2)
    x1, T1 = norm_input(x1)
    x2, T2 = norm_input(x2)
            
    xs = np.concatenate([x1,x2],axis=-1).reshape(1,1,-1,4)
    #preprocessing
    x_list=[]
    x_list=xs.reshape(1,-1,4)
    data["xs"]=torch.from_numpy(xs).float()

    y=np.ones(((np.shape(x_list))[1],1))
    y_list=[]
    y_list=y.reshape(-1,1)
    GT = np.int64(inf['CorrectIndex'])
    data["ys"]=torch.from_numpy(y_list).float()
    data["ts"]=torch.from_numpy(np.ones((1,3))).float()
    data["Rs"]=torch.from_numpy(np.ones((3,3))).float()
    data["GT"]=torch.from_numpy(GT) 
    #print(data["xs"].shape)
    return data
    
def main(config):
    """The main function."""

    # Initialize network
    model = Model(config)  # load model from oan.py--  OANet(nn.Module):
    print('Run here')
    # Run propper mode
    savepath = 'Results'    
    if not os.path.exists(savepath): 
      os.mkdir(savepath)
      
    data = {}
    #print(os.listdir('./jxyData/RegisData'))
    #DataDir='/jxyData/RegisData'
    #DataDir='/data/wangyang/data_dump/importedData/SynData4GANet'
    #DataDir='/data/wangyang/data_dump/importedData/herz'
    DataDir='../Data'     
    result = []
    Indx = []
    #for var_name in list(os.listdir('./jxyData/RegisData')):
    for var_name in list(os.listdir(DataDir)):
        data= load_data(var_name,DataDir)
        aaa=var_name[:-4]
        #print(data)
        p,r,indx,time = test(data, model, config) # run test -- test.test # multiple workors for batch data           
        result.append(str(p)+' '+str(r)+' '+str(time))
        Indx.append(str(indx))
        print(var_name,"    p= {} r= {}    time = {}".format(p,r,time))
        savename = './Results/{}_matches.mat'.format(aaa)
        print(savename)
        sio.savemat(savename,{"name1":'{}'.format(var_name),"index":indx,"time":time})
         
            
if __name__ == "__main__":

    # ----------------------------------------
    # Parse configuration
    config, unparsed = get_config()
    # If we have unparsed arguments, print usage and exit
    if len(unparsed) > 0:
        print_usage()
        exit(1)
    #print(config)
    main(config)

#
# main.py ends here