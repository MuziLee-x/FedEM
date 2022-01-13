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


# Hyper Parameters
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

hyper_params = {
    # dataset to use (datasets/) [ml-1m, ml-latest, ml-10m] provided
    # 'dataset_path': 'ml-1m',
    # 'dataset_path': 'ml-latest',
    # 'dataset_path': 'mooccube',
    'dataset_path': 'mltweeting',
    # 'dataset_path': 'course',
    # 'dataset_path': 'mooc',
    # 'dataset_path': 'ml-10m',
    # 'dataset_path': 'mooccourse',
    # 'dataset_path': 'peek',
    # 'dataset_path': 'ml-10m-sub',
    # 'result_path': "/home/lab30/lylee/Personlized_FLRec/Personlized_FLRec_GMM2_4/results/mltweeting/",
    # 'model_path': "/home/lab30/lylee/Personlized_FLRec/Personlized_FLRec_GMM2_4/model_wei/mltweeting/",
    'result_path': "/home/lab31/FreedomLi/Personlized_FLRec/Personlized_FLRec_GMM2_4/results/mltweeting/",
    'model_path': "/home/lab31/FreedomLi/Personlized_FLRec/Personlized_FLRec_GMM2_4/model_wei/mltweeting/",
    # alpha
    'kl_weight': 0.05,
    # 'kl_weight': 0.01,
    # beta
    'contrast_weight': 0.1,
    # 'contrast_weight': 0.4,
    # Total epochs
    'epochs': 240,  # m1-1m:100,m1-latest:250,mltweeing:50
    'epochs_1': 17,  # mltweeing:7  cluster=4时epoch_1=7;cluster=2时epoch_1=2;;cluster=8时epoch_1=17;
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
    'cluster_num': 4,  # GMM cluster
    # 'cluster_num': 2,  # GMM cluster
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
dataset_info = json.load(open('./dataset_info.json', 'rt'))[hyper_params['dataset_path']]
hyper_params['total_users'] = dataset_info[0]
hyper_params['total_items'] = dataset_info[1]
hyper_params['seq_len'] = dataset_info[2]

info_str = 'dataset:' + hyper_params['dataset_path'] + ' lr1:' + str(hyper_params["lr_primal"]) + ' lr2:' + str(
    hyper_params['lr_dual']) + ' kl:' + str(hyper_params['kl_weight']) + ' contrast:' + str(hyper_params['contrast_weight']) + ' batch:' + str(hyper_params['batch_size']) + ' model_func:' + hyper_params['model_func']
path_str = f'{hyper_params["model_func"]}_{hyper_params["dataset_path"]}_kl_{hyper_params["kl_weight"]}_contrast_{hyper_params["contrast_weight"]}_dropout_0.5_addeps_{hyper_params["add_eps"]}'


# Parser
parser = argparse.ArgumentParser(prog='train')
parser.add_argument("-m", "--msg", default="no description")
args = parser.parse_args()
train_msg = args.msg


# Setup Seed.
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(1111)


# Config logging module.
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
local_time_str = time.strftime("%Y%m%d_%H%M%S", time.localtime())
handler = logging.FileHandler(
    "model_log/log_" + local_time_str + '_' + train_msg.replace(' ', '_') + ".txt")

handler.setLevel(logging.INFO)
formatter = logging.Formatter(
    '%(asctime)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

logger.info(train_msg)
logger.info(info_str)
logger.info('Using CUDA:' + os.environ['CUDA_VISIBLE_DEVICES'])

# Accelerate with CuDNN.
# torch.backends.cudnn.benchmark = True

# Load data at first.
dataset_load0916.load_data(hyper_params)
user_dataset, train_dict = dataset_load0916.generate_train_data(hyper_params)
user_dataloader = Data.DataLoader(
    user_dataset, batch_size=hyper_params['batch_size'], shuffle=True)

test_dataset, _ = dataset_load0916.generate_test_data(hyper_params)
test_dataloader = Data.DataLoader(
    test_dataset, batch_size=hyper_params['batch_size'], shuffle=True)

# Generate validate dataset.
val_dataset, val_dict = dataset_load0916.generate_validate_data(hyper_params)
# val_dataloader = Data.DataLoader(val_dataset, batch_size=hyper_params['batch_size'], shuffle=True)
def generate_cluster_val_dataset(val_dict, index: list, hyper_params):
    val_cluster_data_x = torch.zeros([len(index), hyper_params['seq_len']], dtype=torch.long)
    val_cluster_data_y = torch.zeros([len(index), hyper_params['seq_len']], dtype=torch.long)
    val_cluster_padding = torch.zeros([len(index), hyper_params['seq_len']], dtype=torch.bool)
    val_cluster_user_id = torch.zeros([len(index)], dtype=torch.long)
    val_cluster_cur_cnt = torch.zeros([len(index)], dtype=torch.long)
    for i, user_id in enumerate(index):
        val_cluster_data_x[i] = val_dict[user_id][0]
        val_cluster_data_y[i] = val_dict[user_id][1]
        val_cluster_padding[i] = val_dict[user_id][2]
        val_cluster_user_id[i] = val_dict[user_id][3]
        val_cluster_cur_cnt[i] = val_dict[user_id][4]
    return torch.utils.data.TensorDataset(val_cluster_data_x, val_cluster_data_y, val_cluster_padding, val_cluster_user_id, val_cluster_cur_cnt)

# Build the model.
print('Building net...')
logger.info('Building net...')

net = modelMe0918.Model(hyper_params).to(hyper_params['device'])
adversary = modelMe0918.Adversary(hyper_params).to(hyper_params['device'])
contrast_adversary = modelMe0918.GRUAdversary(hyper_params).to(hyper_params['device'])


print(net)
print('Net build finished.')
logger.info('Net build finished.')

# ---------------------------------------Optimizer-----------------------------------------
optimizer_primal = torch.optim.AdamW([{
    'params': net.parameters(),
    'lr': hyper_params['lr_primal'],
    'weight_decay': hyper_params['l2_regular']}
])
optimizer_dual = torch.optim.SGD([{
    'params': net.encoder.parameters(),
    'lr': hyper_params['lr_dual'],
    'weight_decay': hyper_params['l2_adver']
}, {
    'params': contrast_adversary.parameters(),
    'lr': hyper_params['lr_dual'],
    'weight_decay': hyper_params['l2_adver']
}
])
optimizer_prior = torch.optim.SGD([{
    'params': adversary.parameters(),
    'lr': hyper_params['lr_prior'],
    'weight_decay': hyper_params['l2_adver']}
])


print('User datasets loaded and saved.')
logger.info('User datasets loaded and saved.')


# Evaluator
evaluator = Evaluator(hyper_params=hyper_params, logger=logger)


def train():
    writer = SummaryWriter(f'./runs/{path_str}')
    print('Start training...')
    logger.info('Start training...')
    global_step = 0
    mebank = loss_func.MetricShower()    
    
    # --------------------------The first stage---------------------------
    latent_z_inferred = {}  # key:0~6033
       
    global net
    global adversary
    global contrast_adversary
    
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
        
    
    # --------------------------The second stage---------------------------   
    # net.load_state_dict(torch.load(hyper_params['model_path'] + hyper_params['dataset_path'] + '_' + str(hyper_params['cluster_num']) + '_' + 'net.pth'))  
    # adversary.load_state_dict(torch.load(hyper_params['model_path'] + hyper_params['dataset_path'] + '_' + str(hyper_params['cluster_num']) + '_' +'adversary.pth'))
    # contrast_adversary.load_state_dict(torch.load(hyper_params['model_path'] + hyper_params['dataset_path'] + '_' + str(hyper_params['cluster_num']) + '_' +'contrast_adversary.pth'))
    
    net.load_state_dict(torch.load(hyper_params['model_path'] + hyper_params['dataset_path'] + '_' + 'stage1' + '_' + 'net.pth'))  
    adversary.load_state_dict(torch.load(hyper_params['model_path'] + hyper_params['dataset_path'] + '_' + 'stage1' + '_' +'adversary.pth'))
    contrast_adversary.load_state_dict(torch.load(hyper_params['model_path'] + hyper_params['dataset_path'] + '_' + 'stage1' + '_' +'contrast_adversary.pth'))
    
    for k in z_cluster.keys():  
        clients_id = z_cluster[k] 
        
        if k.startswith('0'):
            if k.endswith('0'):
                sample_Aaugmentation = random.sample(z_cluster['01'], int(len(z_cluster['01'])*0.2))
                clients_train_Aaugmentation = list(set(clients_id).union(set(sample_Aaugmentation)))
            else:
                sample_Aaugmentation = random.sample(z_cluster['00'], int(len(z_cluster['00'])*0.2))
                clients_train_Aaugmentation = list(set(clients_id).union(set(sample_Aaugmentation)))
        else:
            if k.endswith('1'):
                sample_Aaugmentation = random.sample(z_cluster['10'], int(len(z_cluster['10'])*0.2))
                clients_train_Aaugmentation = list(set(clients_id).union(set(sample_Aaugmentation)))    
            else: 
                sample_Aaugmentation = random.sample(z_cluster['11'], int(len(z_cluster['11'])*0.2))
                clients_train_Aaugmentation = list(set(clients_id).union(set(sample_Aaugmentation)))     
        
        # The test set of the cluster_k
        val_cluster_dataset = generate_cluster_val_dataset(val_dict, clients_id, hyper_params)
        val_cluster_dataloader = Data.DataLoader(val_cluster_dataset, batch_size=hyper_params['batch_size'], shuffle=True)  
        
        net.load_state_dict(torch.load(hyper_params['model_path'] + hyper_params['dataset_path'] + '_' + 'stage1' + '_' + 'net.pth'))  
        adversary.load_state_dict(torch.load(hyper_params['model_path'] + hyper_params['dataset_path'] + '_' + 'stage1' + '_' +'adversary.pth'))
        contrast_adversary.load_state_dict(torch.load(hyper_params['model_path'] + hyper_params['dataset_path'] + '_' + 'stage1' + '_' +'contrast_adversary.pth'))         
        #The performance of the pre-trained global model on the cluster
        hr_, _, _ = evaluator.evaluate_stage2(k, hyper_params, net, contrast_adversary, adversary, dataloader=val_cluster_dataloader, validate=True, evaluate_users=hyper_params['evaluate_users'])
        
        for name, param in net.named_parameters():
            # print(name, param)
            # if name == 'encoder.linear_o.weight' or name == 'encoder.linear_o.bias':
            if name == 'encoder.key.weight':
            # if name == 'encoder.query.weight' or name == 'encoder.key.weight' or name == 'encoder.value.weight':
                param.requires_grad = True
            else:
                param.requires_grad = False
                
        # # for name, param in adversary.named_parameters():
            # # param.requires_grad = False
        # # for name, param in contrast_adversary.named_parameters():
            # # param.requires_grad = False
                    
        optimizer_primal = torch.optim.AdamW([{'params': filter(lambda p: p.requires_grad, net.parameters()),'lr': hyper_params['lr_primal'],'weight_decay': hyper_params['l2_regular']}])
        # for name, param in net.named_parameters():
            # if param.requires_grad:
                # print(name)
        
        for epoch in range(hyper_params['epochs_2']):
            net.train()
                       
            vae_clusters = []
            kl_clusters = []
            adver_clusters = []
            prior_clusters = []       
            
            net_clusters = []
            adversary_clusters = []
            contrast_adversary_clusters = []
                        
            sample_client = random.sample(clients_train_Aaugmentation, 32)                        
            for cid in sample_client:    # clients in cluster k                
                batchx, batchy, padding, user_id, cur_cnt = train_dict[cid]
                batchx = batchx.unsqueeze(0)
                batchy = batchy.unsqueeze(0)
                padding = padding.unsqueeze(0)
                padding_z = (1.0 - padding.float()).unsqueeze(2)
                user_id = torch.from_numpy(np.array(user_id)).unsqueeze(0)
                cur_cnt = torch.from_numpy(np.array(cur_cnt)).unsqueeze(0)
                batchx = batchx.to(hyper_params['device'])   # print(batchx.shape)  [64,200]
                batchy = batchy.to(hyper_params['device'])
                padding = padding.to(hyper_params['device'])
                user_id = user_id.to(hyper_params['device'])
                cur_cnt = cur_cnt.to(hyper_params['device'])
                
                # Forward.
                optimizer_primal.zero_grad()
                optimizer_dual.zero_grad()
                pred, x_real, z_inferred, out_embed = net(batchx)
                # --------------------------VAE---------------------------
                multi_loss = loss_func.vae_loss(pred, batchy, cur_cnt, padding, hyper_params)
                if hyper_params['anneal']:
                    anneal = global_step / \
                            hyper_params['total_step'] * hyper_params['kl_weight']
                else:
                    anneal = hyper_params['kl_weight']
                kl_loss = loss_func.kl_loss(adversary, x_real, z_inferred, padding, KL_WEIGHT=anneal)
                adver_loss = loss_func.adversary_loss(
                    contrast_adversary, x_real, z_inferred, padding, CONTRAST_WEIGHT=hyper_params['contrast_weight'])
                loss = multi_loss + adver_loss + kl_loss
                loss.backward()
                optimizer_primal.step()
                optimizer_dual.step()            
                # --------------------------ADVER------------------------------
                optimizer_prior.zero_grad()
                adver_kl_loss = loss_func.adversary_kl_loss(adversary, x_real.detach(), z_inferred.detach(), padding)
                adver_kl_loss.backward()
                optimizer_prior.step()
                
                net_local_stage2 = net.state_dict()
                contrast_adversary_stage2 = contrast_adversary.state_dict()
                adversary_stage2 = adversary.state_dict()
                
                net_clusters.append(net_local_stage2)
                adversary_clusters.append(adversary_stage2)
                contrast_adversary_clusters.append(contrast_adversary_stage2)
                
                vae_clusters.append(multi_loss.item())
                kl_clusters.append(kl_loss.item())
                adver_clusters.append(adver_loss.item())
                prior_clusters.append(adver_kl_loss.item())
            
            # cluster's model
            net_cluster_tmp = traditionalFedAvg(net_clusters)
            contrast_adversary_cluster_tmp = traditionalFedAvg(contrast_adversary_clusters)
            adversary_cluster_tmp = traditionalFedAvg(adversary_clusters)
            
            net.load_state_dict(net_cluster_tmp)
            contrast_adversary.load_state_dict(contrast_adversary_cluster_tmp)
            adversary.load_state_dict(adversary_cluster_tmp)
            
            vae = np.array(vae_clusters).mean()
            kl = np.array(kl_clusters).mean()
            adver = np.array(adver_clusters).mean()
            prior = np.array(prior_clusters).mean()
            mebank.store({'vae': vae, 'kl': kl, 'adver': adver, 'prior': prior})  

            # Show Loss
            print(
                f'EPOCH:({epoch}/{hyper_params["epochs_2"]}),STEP:{global_step}/{hyper_params["total_step"]},vae:{mebank.get("vae")},kl:{mebank.get("kl")},contrast:{mebank.get("adver")},prior:{mebank.get("prior")}')
            logger.info(
                f'EPOCH:({epoch}/{hyper_params["epochs_2"]}),STEP:{global_step}/{hyper_params["total_step"]},vae:{mebank.get("vae")},kl:{mebank.get("kl")},contrast:{mebank.get("adver")},prior:{mebank.get("prior")}')

            writer.add_scalar('loss', mebank.get("vae"), global_step=epoch)
            writer.flush()
            mebank.clear()

            # test the cluster's model
            hr, _, _ = evaluator.evaluate_stage2(k, hyper_params, net, contrast_adversary, adversary, dataloader=val_cluster_dataloader, validate=True, evaluate_users=hyper_params['evaluate_users'])
            writer.add_scalar('hr10', hr[2], global_step=epoch)
            writer.flush()          
            net.train()
            adversary.train()
        print('***************************************************************************************************')
        
    #evaluator.evaluate(hyper_params, net, contrast_adversary, adversary, dataloader=test_dataloader,validate=False)
    writer.close()


# Main
if __name__ == '__main__':
    # Train the model.
    try:
        train()
        logger.info('Finished.')
    except Exception as err:
        err_info = traceback.format_exc()
        print(err_info)
        logger.info(err_info)
        logger.info('Error.')
