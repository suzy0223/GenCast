from __future__ import division

import torch
import numpy as np
from torch import nn
from utils.funcs import *
from utils.gen_fake_val import *
from utils.eval import *
from utils.fastDTW_adj_gen import *
from models.gencast import *
import torch.nn.functional as F
import random
import argparse
import sys
import os
import time
import warnings
from os import path
from sklearn.metrics.pairwise import cosine_similarity
import logging
import copy

warnings.filterwarnings("ignore")



def entropy(predictions, dim=2, reduction='none'):
    epsilon = 1e-5
    H = -predictions * torch.log(predictions + epsilon)
    H = H.sum(dim=dim)
    if reduction == 'mean':
        return H.mean()
    else:
        return H


def compute_huber_delta(errors, quantile=0.9, use_abs=True):
    if isinstance(errors, torch.Tensor):
        errors_np = errors.detach().cpu().numpy().flatten()
    else:
        errors_np = np.array(errors).flatten()
    if use_abs:
        errors_np = np.abs(errors_np)
    delta = np.quantile(errors_np, quantile)
    delta = float(round(delta))

    print(f"Selected delta at Q{int(quantile*100)}: {delta:.4f}")
    return delta


def analyse_physical_loss_distribution(losses):
    mean = losses.mean().item()
    median = losses.median().item()
    q25 = torch.quantile(losses, 0.25).item()
    q75 = torch.quantile(losses, 0.75).item()
    min_val = losses.min().item()
    max_val = losses.max().item()
    std = losses.std().item()
    var = losses.var().item()
    iqr = q75 - q25
    range_ = max_val - min_val
    abs_q90 = torch.quantile(torch.abs(losses), 0.90).item()
    abs_q95 = torch.quantile(torch.abs(losses), 0.95).item()
    abs_85 = torch.quantile(torch.abs(losses), 0.85).item()
    abs_80 = torch.quantile(torch.abs(losses), 0.80).item()
    abs_75 = torch.quantile(torch.abs(losses), 0.75).item()
    
    print(f"  Mean     = {mean:.4f}, Median = {median:.4f}, Std = {std:.4f}, Var = {var:.4f}")
    print(f"  Min-Max  = {min_val:.4f} ~ {max_val:.4f} (range={range_:.4f})")
    print(f"  Q25-Q75  = {q25:.4f} ~ {q75:.4f} (IQR={iqr:.4f})")
    print(f"  Abs Q95  = {abs_q95:.4f}, abs Q90 = {abs_q90:.4f}, abs Q85 = {abs_85:.4f}, abs Q80 = {abs_80:.4f}, abs Q75 = {abs_75:.4f}")


def compute_derivative_loss(output, p, t, fspd, compute_delta=False, delta=5):
    dv_p = torch.autograd.grad(output, p, grad_outputs=torch.ones_like(output), create_graph=True)[0]
    dv_t = torch.autograd.grad(output, t, grad_outputs=torch.ones_like(output), create_graph=True)[0]
    dv_p = dv_p.mean(dim=-1, keepdim=True)
    dv_t = dv_t.mean(dim=-1, keepdim=True)

    loss_raw = dv_t + (2 * output - fspd) * dv_p
    if compute_delta:
        loss = torch.mean(loss_raw ** 2)
        loss_flat = loss_raw.detach().clone().view(-1)
        return loss, loss_flat
    
    if delta is not None and delta > 0:
        loss = torch.mean(F.huber_loss(loss_raw, torch.zeros_like(loss_raw), delta=delta, reduction='none'))
    else:
        loss = torch.mean(loss_raw ** 2)
    
    loss_flat = loss_raw.detach().clone().view(-1)

    return loss, loss_flat

def train_predict(args, loader, model, optimizer, criterion, device, A_s, A_t, know_sensor_list, masked_sensor_ids,
                  his_length, tempe, weight, spa_emb, fspd_o, compute_delta=False):
    batch_loss = 0
    batch_predict_loss = 0
    batch_ada_loss = 0
    batch_ent = 0
    batch_phys = 0
    epoch_phy_losses = []

    # training embedding first
    for idx, (inputs_m, inputs_s, targets, time_mat, weather) in enumerate(loader):
        model.train()
        optimizer.zero_grad()

        # (B,T,N,C)
        inputs_m = inputs_m.to(device)
        inputs_s = inputs_s.to(device)
        targets = targets.to(device)
        time_mat = time_mat.to(device)
        weather = weather.to(device)
        time_mat.requires_grad_(True)
        spa_emb.requires_grad_(True)

        # (B,T,N,C)
        outputs_m, rep_m, attn_list, zc_list, SE = model(inputs_m, time_mat, spa_emb, weather, A_s, A_t,True)
        _, rep_s, _, _, _ = model(inputs_s, time_mat, spa_emb, weather, A_s, A_t,True)

        norm1 = rep_m.norm(dim=1)
        norm2 = rep_s.norm(dim=1)
        sim_matrix = torch.mm(rep_m, torch.transpose(rep_s, 0, 1)) / torch.mm(norm1.view(-1, 1), norm2.view(1, -1))
        sim_matrix = torch.exp(sim_matrix / tempe)

        diag = inputs_m.shape[0]
        pos_sim = sim_matrix[range(diag), range(diag)]
            
        u_loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        u_loss = torch.mean(-torch.log(u_loss))

        # know node: predict loss
        predict_targets = targets[:, his_length:, know_sensor_list + masked_sensor_ids, :]
        predict_outputs = outputs_m[:, :, know_sensor_list + masked_sensor_ids, :]
        loss_predict = criterion(predict_outputs, predict_targets).to(device)
        
        # entropy minimization
        ent = 0
        for attn in attn_list:
            ent += entropy(attn, dim=2).mean()
        ent /= len(attn_list)
        ent = args.lamda*ent
        
        phys_loss, phy_raw = compute_derivative_loss(outputs_m, SE, time_mat, fspd_o, compute_delta, args.delta)
        epoch_phy_losses.append(phy_raw)

        phys_loss = args.theta * phys_loss
        u_loss = weight * u_loss

        loss = loss_predict + u_loss + ent + phys_loss
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)
        optimizer.step()
        batch_loss += loss.detach().cpu().item()
        batch_predict_loss += loss_predict.detach().cpu().item()
        batch_ada_loss += u_loss.detach().cpu().item()
        batch_ent += ent.detach().cpu().item()
        batch_phys += phys_loss.detach().cpu().item()
    epoch_phy_losses_tensor = torch.cat(epoch_phy_losses, dim=0)
    if compute_delta:
        args.delta = compute_huber_delta(epoch_phy_losses_tensor, quantile=args.quantile, use_abs=True)
        print(f"Computed delta for Huber loss: {args.delta}")
        print("Beginning training with computed delta...")
    else:
        analyse_physical_loss_distribution(epoch_phy_losses_tensor)

    return batch_loss / (idx + 1), batch_predict_loss / (idx + 1), batch_ada_loss / (idx + 1), batch_ent / (idx+1), batch_phys / (idx+1)


@torch.no_grad()
def test_predict(loader, model, device, A_s, A_t, know_list, unknow_list, his_length, spa_emb):
    test_pred = []
    test_gt = []
    # m_feat as training, truth feat as label
    for idx, (inputs_m, inputs_s, targets, time_mat, weather) in enumerate(loader):
        model.eval()
        
        # (B,T,N,C)
        inputs_m = inputs_m.to(device)
        inputs_s = inputs_s.to(device)
        targets = targets.to(device)
        time_mat = time_mat.to(device)
        weather = weather.to(device)

        outputs, _, _, _, _ = model(inputs_m, time_mat, spa_emb, weather, A_s, A_t,False)

        targets = targets.detach().cpu().numpy()
        outputs = outputs.detach().cpu().numpy()

        predict_targets = targets[:, his_length:, unknow_list, :]
        predict_outputs = outputs[:, :, unknow_list, :]
        predict_targets = np.reshape(predict_targets, (-1, len(unknow_list))).transpose()
        predict_outputs = np.reshape(predict_outputs, (-1, len(unknow_list))).transpose()
        test_gt.append(predict_targets)
        test_pred.append(predict_outputs)

        if idx == 0:
            predict_targets.shape

    test_pred = np.concatenate(test_pred, axis=1)
    test_gt = np.concatenate(test_gt, axis=1)
    logging.debug("Ground truth shape = {}".format(test_gt.shape))
    RMSE, MAE, MAPE, SMAPE, R2 = metric(test_gt, test_pred)
    return RMSE, MAE, MAPE, SMAPE, R2




if __name__ == "__main__":
    """
    Model training
    """

    # parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default="pems08", type=str)
    parser.add_argument('--split_type', default=1, type=int)
    parser.add_argument('--aug_ratio', default=0.5, type=float)
    parser.add_argument('--a_sg_nk', default=0.5, type=float)
    parser.add_argument('--tempe', default=0.5, type=float)
    parser.add_argument('--lamda', default=1.0, type=float)
    parser.add_argument('--theta', default=1.0, type=float)
    parser.add_argument('--delta', default=0, type=float)
    parser.add_argument('--look_back', default=12, type=int)
    parser.add_argument('--lweight', default=0.5, type=float)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--lr', default=1e-2, type=float)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--patience', default=50, type=int)
    parser.add_argument('--debug', default=1, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--spatial_group', default=5, type=int)
    parser.add_argument('--channel_group', default=2, type=int)
    parser.add_argument('--hash_precision', default=8, type=int)
    parser.add_argument('--unknown_ratio', default=0.5, type=float)
    parser.add_argument('--log_dir', default='log_train', type=str)
    parser.add_argument('--model_dir', default='model_saving', type=str)
    parser.add_argument('--quantile', default=0.9, type=float)
    args = parser.parse_args()

    logging_fn = '{}/{}/Cross_WeatherTime_geo_hash_phy_sg_{}_cg_{}_lamda_{}_ar{}_t{}_{}_s{}_stype{}_lr{}_lw{}_unknown_ratio{}_precision{}_theta{}_lb{}_quantile{}'.format(args.log_dir,args.dataset,str(args.spatial_group),str(args.channel_group),str(args.lamda),str(args.aug_ratio),str(args.tempe),str(args.a_sg_nk),str(args.seed),str(args.split_type),str(args.lr),str(args.lweight),str(args.unknown_ratio),str(args.hash_precision),str(args.theta),str(args.look_back),str(args.quantile))
    saving_dir = '{}/{}'.format(args.model_dir,args.dataset)
    model_saving_fn = '{}/{}/Cross_WeatherTime_geo_hash_phy_sg_{}_cg_{}_lamda_{}_ar{}_t{}_{}_s{}_stype{}_lr{}_lw{}_unknown_ratio{}_precision{}_theta{}_lb{}_quantile{}.pkl'.format(args.model_dir,args.dataset,str(args.spatial_group),str(args.channel_group),str(args.lamda),str(args.aug_ratio),str(args.tempe),str(args.a_sg_nk),str(args.seed),str(args.split_type),str(args.lr),str(args.lweight),str(args.unknown_ratio),str(args.hash_precision),str(args.theta),str(args.look_back),str(args.quantile))

    log_dir = '{}/{}'.format(args.log_dir,args.dataset)
    
    if path.isdir(saving_dir):
        pass
    else:
        os.makedirs(saving_dir)

    if path.isdir(log_dir):
        pass
    else:
        os.makedirs(log_dir)

    if path.isdir(args.model_dir):
        pass
    else:
        os.makedirs(args.model_dir)

    log = open(logging_fn, 'w')
    logging.basicConfig(level=logging.DEBUG if args.debug == 1 else logging.INFO,
                        format="[%(filename)s:%(lineno)s%(funcName)s()] -> %(message)s",
                        handlers=[logging.FileHandler(logging_fn, mode='w'),
                                  logging.StreamHandler()]
                        )

    logging.info('python ' + ''.join(sys.argv))
    logging.info('==============================')
    

    seed = args.seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    device = torch.device('cuda:0')
    logging.info('Model save at = {}, Running on {}, seed={}'.format(model_saving_fn,device,args.seed))
    # criterion = nn.HuberLoss('mean', 1)
    criterion = nn.MSELoss()
    
    if args.dataset == "metr_la":
        Td = 288
        his_len = 24
        pred_len = 24
        node_num = 207
        HT = Td/24 
        dat_dir = "../Dataset/metr/"
        f_dir = dat_dir
        feat_fn = dat_dir + "metr-data-imputed.npy"
        sensor_ids_fn = dat_dir + "graph_sensor_locations.csv"
        weather_fn = dat_dir + "hourly_weather.npy"
        
        unknow_size = int(args.unknown_ratio*node_num)
        valid_size = int(0.1*node_num)
        unknown_set_fn = dat_dir + "unknow_continous_nodes_" + str(unknow_size) + "_"+str(args.split_type)+".npy"
        valid_set_fn = dat_dir + "valid_continous_nodes_" + str(unknow_size) +"_"+ str(valid_size) +"_"+str(args.split_type)+".npy"

        feat, sensor_ids, sensor_dist, adj_s,geo_idx = load_metr(feat_fn, sensor_ids_fn, normalized_k=0.05)
        _, _, _, adj_s_sg,_ = load_metr(feat_fn, sensor_ids_fn, args.a_sg_nk)
        spa_emb = np.load(dat_dir + "geo_hashes_embeddings_precision" + str(args.hash_precision) +".npy")
        print("spa_emb shape = {}".format(spa_emb.shape))

    # weather (N, T, D)
    weather =np.load(weather_fn)
    look_back=args.look_back
    # remove frist time_len
    weather = weather[25-look_back:,:] 
    # remove last 24 hours
    weather = weather[:-24,:]
    # generate time window, 12 hours a window
    weather = seq2instance(weather, look_back)
    # repeat to shape (num_sample, 12, T , N , D), that means every hour repeat 12 times
    weather = np.repeat(np.expand_dims(weather, axis=1), HT, axis=1)
    # reshape to (num_samples,T, N, D)
    weather = weather.reshape(-1, look_back, node_num, weather.shape[-1])
    # saving record
    # np.save(dir + '12hour_weather.npy', record)
    logging.debug("Weather shape = {}".format(weather.shape))
    
    
    logging.debug("spatial adj for NN:")
    logging.debug(adj_s)
    logging.debug("spatial adj of subgraph:")
    logging.debug(adj_s_sg)


    if path.isdir(f_dir):
        pass
    else:
        os.makedirs(f_dir)

    # construct unobserved, observed and valid set
    unknow_set = np.sort(np.load(unknown_set_fn))
    unknow_list = list(unknow_set)
    unknow_list.sort()
    unknow_set = set(unknow_set)

    valid_set = np.sort(np.load(valid_set_fn))
    valid_list = list(valid_set)
    valid_list.sort()
    valid_set = set(valid_set)

    full_set = set(range(0, node_num))
    know_set = full_set - unknow_set - valid_set
    know_list = list(know_set)
    know_list.sort()

    know_valid = know_list + valid_list
    know_valid.sort()
    valid_idx_pos = []
    known_idx_pos = []
    for i in range(len(know_valid)):
        if know_valid[i] in valid_list:
            valid_idx_pos.append(i)
        else:
            known_idx_pos.append(i)

    # TE only has time of the day; has no day of the week
    total_day = int(feat.shape[1] / Td)
    TE = gen_TE(Td, total_day)
    TE = encode_TE(TE, Td)
    feat = feat[:, :total_day * Td]
    weather = weather[:total_day * Td]
    geo_idx = torch.tensor(geo_idx).to(device)

    # weather = weather[:, :total_day * Td]
    # (35136, 12, 400, 4)
    feat_valid = feat[know_valid, :]
    weather_valid = weather[:, :, know_valid, :]
    geo_idx_valid = geo_idx[know_valid, :]
    spa_emb = torch.from_numpy(spa_emb).to(device)
    # spa_emb = torch.mean(spa_emb, dim=1)
    spa_emb_valid = spa_emb[know_valid, :]
    # weather_valid = weather[know_valid, :]
    

    logging.info("dataset = {}, node num = {}, knowset.size = {}, split type = {}".format(args.dataset,node_num,len(know_list),args.split_type))

    sensor_dist_valid = sensor_dist[know_valid, :][:, know_valid]
    e_feat_t = gen_fake_val_weighed_mel(feat, sensor_dist, unknow_set)
    e_valid_feat_t = gen_fake_val_weighed_mel(feat_valid, sensor_dist_valid, set(valid_idx_pos))

    # time intervals per day
    train_ratio = 0.7
    total_day = int(feat.shape[1] / Td)
    train_day = int(total_day * train_ratio)
    train_length = train_day * Td
    sample_length = 7 * Td
    K_dtw_k = 1
    K_dtw_u = 1
    observed_ratio = 0.4

    # generate the free way speed for all sensors from e_feat_t
    fspd = free_spd(e_feat_t, train_day, Td)
    print("fspd shape = {}".format(fspd.shape))
    fspd = torch.from_numpy(fspd).to(device)

    # used to test
    logging.info("Feat shape = {} TE shape = {}".format(feat.shape, TE.shape))
    _, test_dataloader_t = generate_dataset_tatt_no_slide_cl(e_feat_t, feat, TE, weather, train_length, his_len, pred_len,
                                                                 args.batch_size)
        # used to valid
    _, valid_dataloader_t = generate_dataset_tatt_no_slide_cl(e_valid_feat_t, feat_valid, TE, weather_valid, train_length, his_len,
                                                                  pred_len, args.batch_size)
    for i in know_list:
        for j in unknow_list:
            adj_s[i, j] = 0

    if path.isfile(
            f_dir + "Dtw_similarity_continous_nodes_" + str(len(know_valid)) + "_kk_" + str(args.split_type) + ".npy"):
        A_dtw_know = np.load(
            f_dir + "Dtw_similarity_continous_nodes_" + str(len(know_valid)) + "_kk_" + str(args.split_type) + ".npy")
    else:
        A_dtw_know = gen_dtw_adj(feat[:, :sample_length], Td, train_day, unknow_list, know_valid, "know")
        np.save(
            f_dir + "Dtw_similarity_continous_nodes_" + str(len(know_valid)) + "_kk_" + str(args.split_type) + ".npy",
            A_dtw_know)
    if path.isfile(f_dir + "Adj_dtw_continous_nodes_" + str(len(know_valid)) + "_kk_" + str(args.split_type) + ".npy"):
        W_dtw_know = np.load(
            f_dir + "Adj_dtw_continous_nodes_" + str(len(know_valid)) + "_kk_" + str(args.split_type) + ".npy")
    else:
        W_dtw_know = gen_temporal_adj(A_dtw_know, K_dtw_k, unknow_set, "know")
        np.save(f_dir + "Adj_dtw_continous_nodes_" + str(len(know_valid)) + "_kk_" + str(args.split_type) + ".npy",
                W_dtw_know)

    # the temporal rela between know and unknow
    if path.isfile(
            f_dir + "Dtw_similarity_continous_nodes_" + str(len(know_valid)) + "_ku_" + str(args.split_type) + ".npy"):
        A_dtw_unknow = np.load(
            f_dir + "Dtw_similarity_continous_nodes_" + str(len(know_valid)) + "_ku_" + str(args.split_type) + ".npy")
    else:
        A_dtw_unknow = gen_dtw_adj(e_feat_t[:, :sample_length], Td, train_day, unknow_list, know_valid, "unknow")
        np.save(
            f_dir + "Dtw_similarity_continous_nodes_" + str(len(know_valid)) + "_ku_" + str(args.split_type) + ".npy",
            A_dtw_unknow)
    if path.isfile(f_dir + "Adj_dtw_continous_nodes_" + str(len(know_valid)) + "_ku_" + str(args.split_type) + ".npy"):
        W_dtw_unknow = np.load(
            f_dir + "Adj_dtw_continous_nodes_" + str(len(know_valid)) + "_ku_" + str(args.split_type) + ".npy")
    else:
        W_dtw_unknow = gen_temporal_adj(A_dtw_unknow, K_dtw_u, unknow_set, "unknow")
        np.save(f_dir + "Adj_dtw_continous_nodes_" + str(len(know_valid)) + "_ku_" + str(args.split_type) + ".npy",
                W_dtw_unknow)

    # ignore the unkown to know
    W_dtw_unknow[know_list, :] = 0
    W_dtw = W_dtw_unknow + W_dtw_know
    for i in range(W_dtw.shape[0]):
        W_dtw[i, i] = 1

    filters = [[64], [64], [64]]

    logging.info("GNN Filters")
    logging.info(filters)
    A_s = get_normalized_weighted_adj(adj_s)
    A_s = A_s.to(device)
    
    A_t = get_normalized_connective_adj(W_dtw)
    A_t = A_t.to(device)

    # generate the W_dtw for valiadation
    W_dtw_know_valid = np.copy(W_dtw_know[know_valid, :][:, know_valid])
    W_dtw_know_valid[:, valid_idx_pos] = 0
    W_dtw_know_valid[valid_idx_pos, :] = 0
    # After masking locations, ought to update the adj_temporal for them
    A_dtw_valid = gen_dtw_adj(e_valid_feat_t[:, :sample_length], Td, train_day, valid_idx_pos, known_idx_pos, "unknow")
    W_dtw_valid = gen_temporal_adj(A_dtw_valid, K_dtw_u, set(valid_idx_pos), "unknow")
    W_dtw_valid[known_idx_pos, :] = 0
    W_dtw_know_valid = W_dtw_valid + W_dtw_know_valid
    for i in range(W_dtw_know_valid.shape[0]):
        W_dtw_know_valid[i, i] = 1
    A_t_know_valid = get_normalized_connective_adj(W_dtw_know_valid)
    A_t_know_valid = A_t_know_valid.to(device)

    adj_s_valid = adj_s[know_valid, :][:, know_valid]
    for i in known_idx_pos:
        for j in valid_idx_pos:
            adj_s_valid[i, j] = 0
    A_s_valid = get_normalized_weighted_adj(adj_s_valid)
    A_s_valid = A_s_valid.to(device)

    # generate sub-graphs for each location
    adj_s_sg_o = adj_s_sg[know_list, :][:, know_list]
    khop = 1
    neighbourhood_size = []
    neighbourhood = []
    for i in range(len(know_list)):
        khop_neighbours = select_khop_neighbour(adj_s_sg_o, [i], khop, know_list)
        neighbourhood.append(khop_neighbours)
        neighbourhood_size.append(len(khop_neighbours))
    neighbourhood_size_avg = sum(neighbourhood_size) / len(neighbourhood_size)
    logging.debug("Neighbourhood Size = {}".format(neighbourhood_size))
    logging.info("Average Neighbourhood Size = {}".format(neighbourhood_size_avg))


    feat_o = feat[know_list, :]
    weather_o = weather[:,:,know_list, :]
    geo_idx_o = geo_idx[know_list, :]
    adj_s_o = adj_s[know_list, :][:, know_list]
    W_dtw_know_o = W_dtw_know[know_list, :][:, know_list]
    A_s_o = A_s[know_list, :][:, know_list]
    observed_list = [i for i in range(len(know_list))]
    A_dtw_unknow_o = A_dtw_unknow[know_list, :][:, know_list]
    sensor_dist_o = sensor_dist[know_list, :][:, know_list]
    fspd_o = fspd[know_list, :]
    # transfer to torch
    spa_emb_o = spa_emb[know_list, :]

    model_predict = USPGCN_MultiD_Inductive_Tattr(his_len, 64, filters, device, pred_len, args.spatial_group, args.channel_group)
    model_predict = model_predict.to(device)
    best_model = model_predict
    best_rmse = 10000
    rand = np.random.RandomState(args.seed)  # Fixed random output
    lweight = torch.tensor(args.lweight)
    params = list(model_predict.parameters())
    optimizer_predict = torch.optim.Adam(params, lr=args.lr)

    # ---- saving inital state ----
    init_model_state = copy.deepcopy(model_predict.state_dict())
    init_optim_state = copy.deepcopy(optimizer_predict.state_dict())
    
    logging.info("Initial weight = {} tempeture={}".format(lweight, args.tempe))

    start = time.time()
    
    for epoch in range(args.epochs):
        logging.info("=====Epoch {}=====".format(epoch))
        logging.debug('Preprocessing...')
        # remove masked locations relation from know locations' matrix
        W_dtw_know_cur = np.copy(W_dtw_know_o)

        cur_masked_ids = []
        while len(cur_masked_ids) < int(len(know_list) * (args.aug_ratio)):
            masked_num = int((len(know_list)* (args.aug_ratio))/neighbourhood_size_avg)
            if masked_num<1:
                masked_num = 1
            cur_sg_ids =list(rand.choice(list(range(0,len(know_list))),masked_num,replace=False))

            for sg_id in cur_sg_ids:
                cur_masked_ids += neighbourhood[sg_id]
                cur_masked_ids = list(set(cur_masked_ids))
                
                if len(cur_masked_ids) >= int(len(know_list) * (args.aug_ratio)):
                    break

        if len(cur_masked_ids) == len(know_list):
            del cur_masked_ids[int(len(know_list) * args.aug_ratio):len(cur_masked_ids)]
        logging.info('masked node num = {}'.format(len(cur_masked_ids)))

        W_dtw_know_cur[:, cur_masked_ids] = 0
        W_dtw_know_cur[cur_masked_ids, :] = 0

        observed_list_cur = list(set(observed_list) - set(cur_masked_ids))

        # the features input into model
        logging.debug('Generating pseudo observations ....')
        m_feat = gen_fake_val_weighed_mel(feat_o, sensor_dist_o, set(cur_masked_ids))
        
        # After masking locations, ought to update the adj_temporal for them
        A_dtw_mask = gen_dtw_adj(m_feat[:, :sample_length], Td, train_day, cur_masked_ids, observed_list_cur, "unknow")
        W_dtw_mask = gen_temporal_adj(A_dtw_mask, K_dtw_u, set(cur_masked_ids), "unknow")
        W_dtw_mask[observed_list_cur, :] = 0
        W_dtw_cur = W_dtw_mask + W_dtw_know_cur
        for i in range(W_dtw_cur.shape[0]):
            W_dtw_cur[i, i] = 1
        A_t_cur = get_normalized_connective_adj(W_dtw_cur)
        A_t_cur = A_t_cur.to(device)

        adj_s_cur = np.copy(adj_s_o)
        for i in observed_list_cur:
            for j in cur_masked_ids:
                adj_s_cur[i, j] = 0
        adj_s_cur = get_normalized_weighted_adj(adj_s_cur)
        adj_s_cur = adj_s_cur.to(device)

        m_train_dataloader_m, m_test_dataloader_m = generate_dataset_tatt_no_slide_cl(m_feat, feat_o, TE, weather_o,
                                                                                          train_length, his_len,
                                                                                          pred_len, args.batch_size)
        
        logging.debug('Training...')
        # compute the delta of huber loss in frist epoch
        if epoch==0:
            loss, ploss, dloss, ent, phys = train_predict(args, m_train_dataloader_m, model_predict, optimizer_predict, criterion, device,
                                           adj_s_cur, A_t_cur, observed_list_cur, cur_masked_ids, his_len, args.tempe,
                                           lweight, spa_emb_o, fspd_o, compute_delta=True)
            model_predict.load_state_dict(init_model_state)
            optimizer_predict.load_state_dict(init_optim_state)
            
        loss, ploss, dloss, ent, phys = train_predict(args, m_train_dataloader_m, model_predict, optimizer_predict, criterion, device,
                                           adj_s_cur, A_t_cur, observed_list_cur, cur_masked_ids, his_len, args.tempe,
                                           lweight, spa_emb_o, fspd_o)
        logging.debug('Evaluating...')

        valid_rmse, valid_mae, valid_mape, valid_smape, valid_r2 = test_predict(valid_dataloader_t, model_predict, device,
                                                                      A_s_valid, A_t_know_valid, known_idx_pos,
                                                                      valid_idx_pos, his_len,spa_emb_valid)
        logging.info(f'##Training## loss: {loss}, prediction loss: {ploss}, CL loss {dloss}, ent loss {ent}, phy loss {phys}\n' +
              f'##Validation## rmse loss: {valid_rmse}, mae loss: {valid_mae}, mape loss: {valid_mape}, smape loss: {valid_smape} ,r2 loss: {valid_r2}')

        if valid_rmse < best_rmse:
            best_rmse = valid_rmse
            logging.info("best rmse = {:.4f}\n".format(best_rmse))
            best_list = [loss, valid_rmse, valid_mae, valid_mape]
            best_epoch = epoch
            torch.save(model_predict.state_dict(),model_saving_fn)
        if epoch - best_epoch > args.patience:
            logging.info("early stop at epoch {}".format(epoch))
            break
    
    end = time.time()
    logging.info("Finish training! Final weight = {}, tempeture = {}".format(lweight, args.tempe))
    model_predict.load_state_dict(
        torch.load(model_saving_fn))
    valid_rmse, valid_mae, valid_mape, valid_smape, valid_r2 = test_predict(valid_dataloader_t, model_predict, device, A_s_valid,
                                                                  A_t_know_valid, known_idx_pos, valid_idx_pos, his_len,spa_emb_valid)
    start_test = time.time()
    test_rmse, test_mae, test_mape, test_smape, test_r2 = test_predict(test_dataloader_t, model_predict, device, A_s, A_t,
                                                              know_valid, unknow_list, his_len,spa_emb)
    end_test = time.time()
    
    logging.info('\n##Best Epoch## {}'.format(best_epoch))
    logging.info('##on train data## loss: {:.4f}'.format(best_list[0]))
    logging.info('##on valid data predict## rmse loss: {:.4f}, mae loss: {:.4f}, mape loss: {:.4f}, smape: {:.4f}, r2 loss: {:.4f}\n'.format(valid_rmse,valid_mae,valid_mape,valid_smape, valid_r2))
    logging.info('##on test data predict## rmse loss: {:.4f}, mae loss: {:.4f}, mape loss: {:.4f}, smape: {:.4f}, r2 loss: {:.4f}\n'.format(test_rmse,test_mae,test_mape,test_smape,test_r2))
    logging.info('training time: {:.1f}h'.format(((end - start))/60/60))
    logging.info('testing time: {:.1f}s'.format(((end_test - start_test))))