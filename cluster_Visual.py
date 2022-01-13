
from evaluator1102 import Evaluator
import json
import torch
import torch.utils.data as Data
import modelMe0918
import dataset_load0916
import numpy as np
import os
import logging
import traceback
import loss_func
import random
import time
import argparse
import sys
from tensorboardX import SummaryWriter
from Fed0918 import traditionalFedAvg, FedAvg
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import pandas as pd


# Hyper Parameters
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

hyper_params = {
    # dataset to use (datasets/) [ml-1m, ml-latest, ml-10m] provided
    # 'dataset_path': 'ml-1m',
    # 'dataset_path': 'ml-latest',
    # 'dataset_path': 'mooccube',
    # 'dataset_path': 'mltweeting',
    # 'dataset_path': 'course',
    # 'dataset_path': 'mooc',
    # 'dataset_path': 'ml-10m',
    # 'dataset_path': 'mooccourse',
    'dataset_path': 'peek',
    # 'dataset_path': 'ml-10m-sub',
    # 'result_path': "/home/lab30/lylee/Personlized_FLRec/Personlized_FLRec_GMM2_4/results/mltweeting/",
    # 'model_path': "/home/lab30/lylee/Personlized_FLRec/Personlized_FLRec_GMM2_4/model_wei/mltweeting/",
    # 'result_path': "/home/lab31/FreedomLi/Personlized_FLRec/Personlized_FLRec_GMM2_4/results/mltweeting/",
    # 'model_path': "/home/lab31/FreedomLi/Personlized_FLRec/Personlized_FLRec_GMM2_4/model_wei/mltweeting/",
    # 'result_path': "/home/lab31/FreedomLi/Personlized_FLRec/Personlized_FLRec_GMM2_4/results/ml-1m/",
    # 'model_path': "/home/lab31/FreedomLi/Personlized_FLRec/Personlized_FLRec_GMM2_4/model_wei/ml-1m/",
    # 'result_path': "/home/lab31/FreedomLi/Personlized_FLRec/Personlized_FLRec_GMM2_4/results/ml-latest/",
    # 'model_path': "/home/lab31/FreedomLi/Personlized_FLRec/Personlized_FLRec_GMM2_4/model_wei/ml-latest/",
    'result_path': "/home/lab31/FreedomLi/Personlized_FLRec/Personlized_FLRec_GMM2_4/results/peek/",
    'model_path': "/home/lab31/FreedomLi/Personlized_FLRec/Personlized_FLRec_GMM2_4/model_wei/peek/",
    # alpha
    'kl_weight': 0.05,
    # 'kl_weight': 0.01,
    # beta
    'contrast_weight': 0.1,
    # 'contrast_weight': 0.4,
    # Total epochs
    'epochs': 240,  # m1-1m:100,m1-latest:250,mltweeing:50
    'epochs_1': 17,  # mltweeing:7  cluster=4ʱepoch_1=7;cluster=2ʱepoch_1=2;;cluster=8ʱepoch_1=17;
    'epochs_2': 200,  # mltweeing
    # Set the number of users in evalutaion during training to speed up
    # If set to None, all of the users will be evaluated
    'evaluate_users': None,
    'item_embed_size': 128,
    'rnn_size': 100,
    'hidden_size': 100,
    'latent_size': 64,
    'timesteps': 5,
    'test_prop': 0.2,
    'batch_size': 1,
    # 'batch_size': 64,
    'anneal': False,
    'time_split': True,
    # 'model_func': 'fc_cnn',
    'model_func': 'fc_att',
    'add_eps': True,
    # 'device': 'cuda',
    'device': 'cpu',  # Add
    'check_freq': 1,
    # 'cluster_num': 8,  # GMM cluster
    # 'cluster_num': 4,  # GMM cluster
    'cluster_num': 5,  # GMM cluster
    # 'cluster_num': 2,  # GMM cluster
    # 'cluster_num': 3,  # GMM cluster
    # 'cluster_num': 32,  # GMM cluster
    # 'cluster_num': 16,  # GMM cluster
    # 'cluster_num': 64,  # GMM cluster
    'Ks': [1, 5, 10, 20, 50, 100],
    # 'lr_primal': 1e-3,
    # 'lr_dual': 3e-4,
    # 'lr_prior': 5e-4,
    'lr_primal_1': 0.25e-3,   # mltweeting:lr(0.25,0.7,1.25) is better than lr(0.5,1.5,2.5)
    'lr_dual_1': 0.75e-4,
    'lr_prior_1': 1.25e-4,
    # 'lr_primal': 0.01e-3,   # mltweeting:lr(0.25,0.7,1.25) is better than lr(0.5,1.5,2.5)
    # 'lr_dual': 0.03e-4,
    # 'lr_prior': 0.05e-4,
    'lr_primal': 1e-7,   # mltweeting:lr(0.25,0.7,1.25) is better than lr(0.5,1.5,2.5)
    'lr_dual': 3e-8,
    'lr_prior': 5e-8,
    # 'lr_primal': 0.1e-3,   # mltweeting:lr(0.25,0.7,1.25) is better than lr(0.5,1.5,2.5)
    # 'lr_dual': 0.3e-4,
    # 'lr_prior': 0.5e-4,
    # 'lr_primal': 0.5e-3,   # ml-1m, k = 8, lr(0.5,1.5,2.5) is better than lr(1,3,5)
    # 'lr_dual': 1.5e-4,
    # 'lr_prior': 2.5e-4,
    'l2_regular': 1e-2,
    'l2_adver': 1e-1,
    'total_step': 2000000,
    # 'total_users': 6034  #6034 ml-1m, #604 ml-latest, # 15062 mooccube
    # 'total_users': 15062  #mooccube
    'total_users': 3758  #mltweeting
    # 'total_users': 4996  #peek
    # 'total_users': 5000  #ml-10m-sub
}


def train():    
    # --------------------------Cluster the clients to groups---------------------------    
    # latent variable view
    latent_x_infer = np.load(hyper_params['model_path'] + hyper_params['dataset_path'] + '_' + 'stage1' + '_'  + "latent_x_inferred.npy",allow_pickle=True) 
    latent_x_inferred = latent_x_infer.item()
    
    latent_z_t = sorted(latent_x_inferred.items(), key=lambda item:item[0])
    latent_zz = {}
    for i in range(len(latent_z_t)):
        latent_zz[latent_z_t[i][0]] = latent_z_t[i][1]
    tmpp = []  
    for i in latent_zz.keys():
        tmpp.append(latent_zz[i].detach().numpy().tolist())
    latent_zz = torch.tensor(tmpp)   # [6034, 1, 200, 64] 
    latent_zz = torch.mean(torch.squeeze(latent_zz), dim=1)  # [6034, 64]
    
    # real_x view
    real_x = np.load(hyper_params['model_path'] + hyper_params['dataset_path'] + '_' + 'stage1' + '_'  + "real_x.npy",allow_pickle=True) 
    real_x = real_x.item()
    real_xx = sorted(real_x.items(), key=lambda item:item[0])
    r_x = {}
    for i in range(len(real_xx)):
        r_x[real_xx[i][0]] = real_xx[i][1]
    tmp = []  
    for i in r_x.keys():
        tmp.append(r_x[i].detach().numpy().tolist())
    r_x = torch.tensor(tmp)   # [6034, 1, 50, 128] 
    # print(r_x.shape)
    r_x = torch.mean(torch.squeeze(r_x), dim=1)  # [6034, 128]
    # print(r_x.shape)
    
    # user_ferture:cat(view1, view2) 
    feature_user = torch.cat((latent_zz, r_x), 1)  # # [6034, 192]
    # print(feature_user.shape)
    
    #--------------Hierarchical clustering-----------------  
    if hyper_params['cluster_num']==2:
        gmm = GaussianMixture(n_components=hyper_params['cluster_num'], covariance_type='spherical', tol=0.001, init_params='random', random_state=3)
        # gmm.fit(latent_zz)
        # cluster = gmm.predict(latent_zz)
        gmm.fit(feature_user)
        cluster = gmm.predict(feature_user)
        # print(cluster)
        probs = gmm.predict_proba(feature_user)
        # print(probs)   
        z_cluster_t = []
        for i in range(len(cluster)):
            tmp = {}
            tmp[cluster[i]] = i   # {cluster K: client ID}
            z_cluster_t.append(tmp)   
        z_each_cluster_t = {}
        for _ in z_cluster_t:
            for k, v in _.items():
                z_each_cluster_t.setdefault(k, []).append(v)  # {cluster K: [client ID1, client ID2,...]}
        z_each_cluster_tmp = sorted(z_each_cluster_t.items(), key=lambda item:item[0])
        z_cluster = {}
        for i in range(len(z_each_cluster_tmp)):
            z_cluster[z_each_cluster_tmp[i][0]] = z_each_cluster_tmp[i][1]    
            
        client2cluster = {}
        for k, v in z_cluster.items():
            for i in range(len(v)):
                client2cluster[v[i]] = k
        client2cluster_rank = sorted(client2cluster.items(), key=lambda item:item[0])
        label = []
        for i in range(len(client2cluster_rank)):
            label.append(client2cluster_rank[i][1])
        labels = np.array(label) 
        labels_series =  pd.Series(labels)
        
        feature_user_tmp = feature_user.numpy()
        feature_user_tmp_series =  pd.DataFrame(feature_user_tmp, index=None, columns=None) 
        results = pd.concat([feature_user_tmp_series, labels_series], axis=1)
        
        # visualization
        tsne=TSNE()
        tsne.fit_transform(feature_user)  
        # tsne=pd.DataFrame(tsne.embedding_,index=feature_user_tmp.index) 
        tsne=pd.DataFrame(tsne.embedding_) 
        d = tsne[results.iloc[:,-1]==0]
        plt.plot(d[0], d[1], 'r.') 
        d = tsne[results.iloc[:,-1]==1]
        plt.plot(d[0], d[1], 'b*') 
        plt.show()
        plt.savefig(hyper_params['result_path']+hyper_params['dataset_path']+str(hyper_params['cluster_num'])+str('Hiera')+'cluster_visual.pdf')
        plt.close()
    else:  # split by tree
        gmm = GaussianMixture(2, covariance_type='spherical', tol=0.001, init_params='random', random_state=3)
        # gmm.fit(latent_zz)
        # cluster = gmm.predict(latent_zz)
        gmm.fit(feature_user)
        cluster = gmm.predict(feature_user)
        # print(cluster)
        probs = gmm.predict_proba(feature_user)
        # print(probs)
        z_cluster_t = []
        for i in range(len(cluster)):
            tmp = {}
            tmp[cluster[i]] = i   # {cluster K: client ID}
            z_cluster_t.append(tmp)   
        z_each_cluster_t = {}
        for _ in z_cluster_t:
            for k, v in _.items():
                z_each_cluster_t.setdefault(k, []).append(v)  # {cluster K: [client ID1, client ID2,...]}
        z_each_cluster_tmp = sorted(z_each_cluster_t.items(), key=lambda item:item[0])
        # print(z_each_cluster_tmp)
        z_cluster_tmp = {}
        for i in range(len(z_each_cluster_tmp)):
            z_cluster_tmp[z_each_cluster_tmp[i][0]] = z_each_cluster_tmp[i][1]
        # print(len(z_cluster_tmp[1]))  # 0:3758-1070=2688,1:1070
        z_cluster = {}
        for i in range(2):
            clusterID = z_cluster_tmp[i]
            feature_user_clusterID = feature_user[clusterID]
            # print(feature_user_clusterID.shape)   # [2688, 192]
            gmm = GaussianMixture(int(hyper_params['cluster_num']/2), covariance_type='spherical', tol=0.001, init_params='random', random_state=3)
            gmm.fit(feature_user_clusterID)
            subcluster = gmm.predict(feature_user_clusterID)
            sub_cluster_t = []
            for j in range(len(subcluster)):
                tmp = {}
                tmp[subcluster[j]] = j   # {cluster K: client ID}
                sub_cluster_t.append(tmp) 
            sub_each_cluster_t = {}
            for _ in sub_cluster_t:
                for k, v in _.items():
                    sub_each_cluster_t.setdefault(k, []).append(v)  # {cluster K: [client ID1, client ID2,...]}
            for k in range(len(sub_each_cluster_t)):
                z_cluster[str(i)+str(k)] =  sub_each_cluster_t[k]
        # print(z_cluster)
        # print(len(z_cluster['00']))  # mltweeting:1416
        # print(len(z_cluster['01']))  # mltweeting:1272
        # print(len(z_cluster['10']))  # mltweeting:376
        # print(len(z_cluster['11']))  # mltweeting:694
                
        client2cluster = {}
        for k, v in z_cluster.items():
            for i in range(len(v)):
                client2cluster[v[i]] = k
        client2cluster_rank = sorted(client2cluster.items(), key=lambda item:item[0])
        label = []
        for i in range(len(client2cluster_rank)):
            label.append(client2cluster_rank[i][1])
        labels = np.array(label) 
        labels_series =  pd.Series(labels)
        
        feature_user_tmp = feature_user.numpy()
        feature_user_tmp_series =  pd.DataFrame(feature_user_tmp, index=None, columns=None) 
        results = pd.concat([feature_user_tmp_series, labels_series], axis=1)
        
        # visualization
        tsne=TSNE()
        tsne.fit_transform(feature_user)  
        # tsne=pd.DataFrame(tsne.embedding_,index=feature_user_tmp.index) 
        tsne=pd.DataFrame(tsne.embedding_) 
        d = tsne[results.iloc[:,-1]=='00']
        plt.plot(d[0], d[1], 'r.') 
        d = tsne[results.iloc[:,-1]=='01']
        plt.plot(d[0], d[1], 'go') 
        d = tsne[results.iloc[:,-1]=='10']
        plt.plot(d[0], d[1], 'b*') 
        d = tsne[results.iloc[:,-1]=='11']
        plt.plot(d[0], d[1], 'y.') 
        plt.show()
        plt.savefig(hyper_params['result_path']+hyper_params['dataset_path']+str(hyper_params['cluster_num'])+str('Hiera')+'cluster_visual.pdf')
        plt.close()

# train()

def cluster():       
    #--------------clustering directly-----------------    
    # latent variable view
    latent_x_infer = np.load(hyper_params['model_path'] + hyper_params['dataset_path'] + '_' + 'stage1' + '_'  + "latent_x_inferred.npy",allow_pickle=True) 
    latent_x_inferred = latent_x_infer.item()    
    latent_z_t = sorted(latent_x_inferred.items(), key=lambda item:item[0])
    latent_zz = {}
    for i in range(len(latent_z_t)):
        latent_zz[latent_z_t[i][0]] = latent_z_t[i][1]
    tmpp = []  
    for i in latent_zz.keys():
        tmpp.append(latent_zz[i].detach().numpy().tolist())
    latent_zz = torch.tensor(tmpp)   # [6034, 1, 200, 64] 
    latent_zz = torch.mean(torch.squeeze(latent_zz), dim=1)  # [6034, 64]
    
    # real_x view
    real_x = np.load(hyper_params['model_path'] + hyper_params['dataset_path'] + '_' + 'stage1' + '_'  + "real_x.npy",allow_pickle=True) 
    real_x = real_x.item()
    real_xx = sorted(real_x.items(), key=lambda item:item[0])
    r_x = {}
    for i in range(len(real_xx)):
        r_x[real_xx[i][0]] = real_xx[i][1]
    tmp = []  
    for i in r_x.keys():
        tmp.append(r_x[i].detach().numpy().tolist())
    r_x = torch.tensor(tmp)   # [6034, 1, 50, 128] 
    # print(r_x.shape)
    r_x = torch.mean(torch.squeeze(r_x), dim=1)  # [6034, 128]
    # print(r_x.shape)
    
    # cat(view1, view2) 
    feature_user = torch.cat((latent_zz, r_x), 1)  # # [6034, 192]
    # print(feature_user.shape)
    
    gmm = GaussianMixture(n_components=hyper_params['cluster_num'], covariance_type='spherical', tol=0.001, init_params='random', random_state=3)
    # gmm.fit(latent_zz)
    # cluster = gmm.predict(latent_zz)
    gmm.fit(feature_user)
    cluster = gmm.predict(feature_user)
    # print(cluster)
    probs = gmm.predict_proba(feature_user)
    # print(probs)
    
    z_cluster_t = []
    for i in range(len(cluster)):
        tmp = {}
        tmp[cluster[i]] = i   # {cluster K: client ID}
        z_cluster_t.append(tmp)   
    z_each_cluster_t = {}
    for _ in z_cluster_t:
        for k, v in _.items():
            z_each_cluster_t.setdefault(k, []).append(v)  # {cluster K: [client ID1, client ID2,...]}
    z_each_cluster_tmp = sorted(z_each_cluster_t.items(), key=lambda item:item[0])
    z_cluster = {}
    for i in range(len(z_each_cluster_tmp)):
        z_cluster[z_each_cluster_tmp[i][0]] = z_each_cluster_tmp[i][1]
    
    client2cluster = {}
    for k, v in z_cluster.items():
        for i in range(len(v)):
            client2cluster[v[i]] = k
    client2cluster_rank = sorted(client2cluster.items(), key=lambda item:item[0])
    label = []
    for i in range(len(client2cluster_rank)):
        label.append(client2cluster_rank[i][1])
    labels = np.array(label)
    labels_series =  pd.Series(labels)
    feature_user_tmp = feature_user.numpy()
    feature_user_tmp_series =  pd.DataFrame(feature_user_tmp, index=None, columns=None) 
    results = pd.concat([feature_user_tmp_series, labels_series], axis=1)
    
    # visualization
    tsne=TSNE()
    tsne.fit_transform(feature_user)
    tsne=pd.DataFrame(tsne.embedding_) 
    d = tsne[results.iloc[:,-1]==0]
    plt.plot(d[0], d[1], 'r.') 
    d = tsne[results.iloc[:,-1]==1]
    plt.plot(d[0], d[1], 'gs') 
    d = tsne[results.iloc[:,-1]==2]
    plt.plot(d[0], d[1], 'b*') 
    d = tsne[results.iloc[:,-1]==3]
    plt.plot(d[0], d[1], 'yx') 
    d = tsne[results.iloc[:,-1]==4]
    plt.plot(d[0], d[1], 'md') 
    plt.show()
    plt.savefig(hyper_params['result_path']+hyper_params['dataset_path']+str(hyper_params['cluster_num'])+str('Direct')+'cluster_visual.pdf')
    plt.close()
cluster()