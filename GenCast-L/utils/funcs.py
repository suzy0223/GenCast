from copyreg import pickle
import numpy as np
import pandas as pd
import pickle
import torch
from torch.utils.data import DataLoader, Dataset
import math
from geographiclib.geodesic import Geodesic
import random
from sklearn.metrics.pairwise import cosine_similarity
        
class MyDatasetTatt_Noslide_CL(Dataset):
    def __init__(self, data, label, TE, weather, split_start, split_end, his_length, pred_length):
        split_start = int(split_start)
        split_end = int(split_end)
        self.data = data[:,split_start: split_end]
        self.label = label[:,split_start: split_end]
        # weather
        self.weather = weather[split_start: split_end]
        # print("self.weather", self.weather.shape)
        self.mean = self.data.mean()
        self.std = self.data.std()
        # (35136, 12, 400, 4)
        # mean and std for each dimension, axis=-1
        self.weather_mean = self.weather.mean(axis=(0,1,2))
        self.weather_std = self.weather.std(axis=(0,1,2))

        self.his_length = his_length
        self.pred_length = pred_length
        self.TE = TE
    
    def __getitem__(self, index):
        x = self.data[:, index*self.his_length: (index+1)* self.his_length]
        # x = self.data[:, index: index + self.his_length]
        x = (x - self.mean) / self.std
        # X (T, N, C)
        x = x.transpose()
        x = np.expand_dims(x,axis=2)

        x1 = self.label[:, index*self.his_length: (index+1)* self.his_length]
        # x = self.data[:, index: index + self.his_length]
        x1 = (x1 - self.mean) / self.std
        # X (T, N, C)
        x1 = x1.transpose()
        x1 = np.expand_dims(x1,axis=2)

        y = self.label[:, index*self.his_length: (index+1)* self.his_length + self.pred_length]
        y = y.transpose()
        y = np.expand_dims(y,axis=2)
        # TE (2, T)
        te = self.TE[:,index*self.his_length: (index+1)* self.his_length]
        # TE (2, 1, T)
        te = np.expand_dims(te, axis=1)
        # TE (2, N, T)
        te = np.tile(te, (1, x.shape[1], 1))
        # print("te", te.shape)

        # TE  (2, N, T) to (T,N,2)
        te = te.transpose(2,1,0)
        # te = np.expand_dims(te,axis=2)

        # weather (time_window, look_back, N, D) - (look_back, N, D); select first steps. look_back_window
        weather = self.weather[(index+1)* self.his_length-1: (index+1)* self.his_length]
        # normilise weather by each dimension
        weather = (weather - self.weather_mean) / self.weather_std
        # squeeze
        weather = np.squeeze(weather)

        return torch.Tensor(x), torch.Tensor(x1), torch.Tensor(y), torch.Tensor(te), torch.Tensor(weather)
    def __len__(self):
        return self.data.shape[1] // self.pred_length - 1
        # return int((self.data.shape[1]-self.pred_length) / self.his_length)

class MyDatasetTatt_CL(Dataset):
    def __init__(self, data, label, TE, weather, split_start, split_end, his_length, pred_length):
        split_start = int(split_start)
        split_end = int(split_end)
        self.data = data[:,split_start: split_end]
        self.label = label[:,split_start: split_end]
        self.weather = weather[split_start: split_end]
        self.mean = self.data.mean()
        self.std = self.data.std()
        # print("self.mean", self.mean)
        # print("self.std", self.std)

        # mean and std for each dimension, axis=-1
        self.weather_mean = self.weather.mean(axis=(0,1,2))
        self.weather_std = self.weather.std(axis=(0,1,2))
        # print("self.weather_mean", self.weather_mean.shape)
        # print("self.weather_std", self.weather_std.shape)
        self.std = self.data.std()
        self.his_length = his_length
        self.pred_length = pred_length
        self.TE = TE
    
    def __getitem__(self, index):
        # data: (N, T)
        x = self.data[:, index: index + self.his_length]
        x = (x - self.mean) / self.std
        # X (T, N, C)
        x = x.transpose()
        x = np.expand_dims(x,axis=2)

        x1 = self.label[:, index: index + self.his_length]
        x1 = (x1 - self.mean) / self.std
        # X (T, N, C)
        x1 = x1.transpose()
        x1 = np.expand_dims(x1,axis=2)

        y = self.label[:, index: index + self.his_length + self.pred_length]
        y = y.transpose()
        y = np.expand_dims(y,axis=2)
        # TE (T, 2)
        te = self.TE[:,index: index + self.his_length]
        te = np.expand_dims(te, axis=1)
        # TE (2, N, T)
        te = np.tile(te, (1, x.shape[1], 1))
        # print("te", te.shape)

        # TE  (2, N, T) to (T,N,2)
        te = te.transpose(2,1,0)


        # weather (time_wondow, look_back, N, D) - (look_back, N, D); select first steps. look_back_window
        weather = self.weather[index + self.his_length-1: index + self.his_length]
        # normilise weather by each dimension
        weather = (weather - self.weather_mean) / self.weather_std
        # squeeze
        weather = np.squeeze(weather)


        return torch.Tensor(x), torch.Tensor(x1), torch.Tensor(y), torch.Tensor(te), torch.Tensor(weather)

    def __len__(self):
        return self.data.shape[1] - self.his_length - self.pred_length + 1



def generate_dataset_tatt_cl(data,label,TE, weather,train_length,his_length=24,pred_length=24,batch_size=32):
    """
    Args:
        data: input dataset, shape like T * N
        batch_size: int 
        train_ratio: float, the ratio of the dataset for training
        his_length: the input length of time series for prediction
        pred_length: the target length of time series of prediction

    Returns:
        train_dataloader: torch tensor, shape like batch * N * his_length * features
        test_dataloader: torch tensor, shape like batch * N * pred_length * features
    """
    train_dataset = MyDatasetTatt_CL(data, label, TE, weather, 0, train_length, his_length, pred_length)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    # print(train_dataset.__len__())

    test_dataset = MyDatasetTatt_Noslide_CL(data, label, TE, weather, train_length, data.shape[1], his_length, pred_length)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    return train_dataloader, test_dataloader


def generate_dataset_tatt_no_slide_cl(data,label,TE, weather, train_length,his_length=24,pred_length=24,batch_size=32):
    """
    Args:
        data: input dataset, shape like T * N
        batch_size: int 
        train_ratio: float, the ratio of the dataset for training
        his_length: the input length of time series for prediction
        pred_length: the target length of time series of prediction

    Returns:
        train_dataloader: torch tensor, shape like batch * N * his_length * features
        test_dataloader: torch tensor, shape like batch * N * pred_length * features
    """ 
    start_idx = np.random.choice(range(his_length))
    train_dataset = MyDatasetTatt_Noslide_CL(data, label, TE, weather, start_idx, train_length, his_length, pred_length)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = MyDatasetTatt_Noslide_CL(data, label, TE, weather, train_length, data.shape[1], his_length, pred_length)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, test_dataloader


def load_mel_data(feat_fn, sensor_fn, adj_type="connectivity", normalized_k=0.05):
    #  speed = 0; volume = 1
    feat = np.load(feat_fn)[:,:]
    sensors = np.load(sensor_fn)
    lon = sensors[:,2]
    lat = sensors[:,3]
    latlon = encode_geolocation(lat, lon)
    dist_mx = latlon2dist(lat, lon)

    distances = dist_mx[~np.isinf(dist_mx)].flatten()
    std = distances.std()
    adj_mx = np.exp(-np.square(dist_mx/std))
    adj_mx[adj_mx < normalized_k] = 0
        
    adj_mx[np.isinf(adj_mx)] = 0
    
    return feat, sensors,dist_mx,adj_mx,latlon

def load_pems_data(feat_fn,sensor_fn, normalized_k = 0.05):
    # feat(N, T); sensors(id,lat,lon) N = 325
    df = pd.read_hdf(feat_fn)
    # transfer_set = df.as_matrix()
    sensors = pd.read_csv(sensor_fn,header=None)
    sensor_ids = sensors[0].tolist()
    sensors = np.array(sensors)

    lat = sensors[:,1]
    lon = sensors[:,2]
    latlon = encode_geolocation(lat, lon)
    dist_mx = latlon2dist(lat, lon)

    distances = dist_mx[~np.isinf(dist_mx)].flatten()
    std = distances.std()
    adj_mx = np.exp(-np.square(dist_mx/std))
    adj_mx[adj_mx < normalized_k] = 0
        
    adj_mx[np.isinf(adj_mx)] = 0

    feat = np.array(df).transpose()
    
    return feat, sensors,dist_mx,adj_mx,latlon


def normalize_latlon(latlon):
    # min-max normalization
    latlon[:,0] = (latlon[:,0] - latlon[:,0].min()) / (latlon[:,0].max() - latlon[:,0].min())
    latlon[:,1] = (latlon[:,1] - latlon[:,1].min()) / (latlon[:,1].max() - latlon[:,1].min())

    return latlon


def load_pems_data_np(feat_fn,sensor_fn, normalized_k = 0.05):
    # feat(N, T, C[speed,flow]); sensors(id,lat,lon)
    df =np.load(feat_fn)[:,:,0]
    # transfer_set = df.as_matrix()
    sensors = pd.read_csv(sensor_fn)
    lat = np.array(sensors['lat'])
    lon = np.array(sensors['lon'])

    latlon = encode_geolocation(lat, lon)
    # print("latlon.shape", latlon.shape)

    # cat lat lon
    # latlon = normalize_latlon(np.concatenate((np.array(sensors['lat']).reshape(-1,1),np.array(sensors['lon']).reshape(-1,1)),axis=1))
    dist_mx = latlon2dist(lat, lon)
    sensors = np.array(sensors)

    distances = dist_mx[~np.isinf(dist_mx)].flatten()
    std = distances.std()
    adj_mx = np.exp(-np.square(dist_mx/std))
    adj_mx[adj_mx < normalized_k] = 0
        
    adj_mx[np.isinf(adj_mx)] = 0
    
    return df, sensors,dist_mx,adj_mx, latlon



def load_air(feat_fn,sensor_fn, normalized_k = 0.05):
    # feat(N, T, C[speed,flow]); sensors(id,lat,lon)
    df =np.load(feat_fn)
    # transfer_set = df.as_matrix()
    sensors = pd.read_csv(sensor_fn)
    lat = np.array(sensors['lat'])
    lon = np.array(sensors['lon'])
    dist_mx = latlon2dist(lat, lon)
    sensors = np.array(sensors)

    distances = dist_mx[~np.isinf(dist_mx)].flatten()
    std = distances.std()
    adj_mx = np.exp(-np.square(dist_mx/std))
    adj_mx[adj_mx < normalized_k] = 0
        
    adj_mx[np.isinf(adj_mx)] = 0
    
    return df, sensors,dist_mx,adj_mx

def load_metr(feat_fn,sensor_fn, normalized_k = 0.05):
    # feat(N, T); sensors(id,lat,lon)
    df =np.load(feat_fn)
    # transfer_set = df.as_matrix()
    sensors = pd.read_csv(sensor_fn)
    lat = np.array(sensors['latitude'])
    lon = np.array(sensors['longitude'])

    latlon = encode_geolocation(lat, lon)

    # cat lat lon
    # latlon = normalize_latlon(np.concatenate((np.array(sensors['lat']).reshape(-1,1),np.array(sensors['lon']).reshape(-1,1)),axis=1))
    dist_mx = latlon2dist(lat, lon)
    sensors = np.array(sensors)

    distances = dist_mx[~np.isinf(dist_mx)].flatten()
    std = distances.std()
    adj_mx = np.exp(-np.square(dist_mx/std))
    adj_mx[adj_mx < normalized_k] = 0
        
    adj_mx[np.isinf(adj_mx)] = 0
    
    return df, sensors,dist_mx,adj_mx,latlon



def select_khop_neighbour(A,sensor_ids,khop,know_list):
    for k in range(khop):
        sids_list = []
        for sid in sensor_ids:
            relation = A[sid,:]
            ids = np.where(relation!=0)[0]
            for i in ids:
                sids_list.append(i)
        sensor_ids = list(set(sids_list))
    return sensor_ids

def latlon2dist(lat, lon):
    loc_num = lat.shape[0]
    dist_mat = np.zeros([loc_num,loc_num])
    for i in range(loc_num):
        for j in range(i+1, loc_num):
            s = Geodesic.WGS84.Inverse(lat[i],lon[i],lat[j],lon[j])
            dist_mat[i,j] = s['s12']/1000
    return dist_mat+dist_mat.T

def get_normalized_connective_adj(A,is_self=False):
    """
    Returns a normallized tensor.
    """
    D = np.array(np.sum(A, axis=1)).reshape((-1,))
    D[D <= 10e-5] = 10e-5    # Prevent infs
    diag = np.reciprocal(np.sqrt(D))
    A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), A),
                         diag.reshape((1, -1)))
    if is_self:
        A_wave = A_wave + np.eye(A_wave.shape[0])

    return torch.from_numpy(A_wave.astype(np.float32))

def get_normalized_weighted_adj(adj):
    """
    Returns a normallized tensor.
    """
    A = np.copy(adj)
    node_num = A.shape[0]
    for i in range(node_num):
        adj_idx = np.where(A[i]!=0)[0]
        adj_i = A[i,adj_idx]
        adj_i = np.exp(adj_i)
        adj_sum = np.sum(adj_i)
        adj_i = adj_i/adj_sum
        A[i,adj_idx] = np.copy(adj_i)
    for i in range(node_num):
        A[i][i] = 1

    return torch.from_numpy(A.astype(np.float32))

def get_normalized_mat(A):
    """
    Returns a normallized tensor.
    """
    D = torch.sum(A, dim=1).reshape((-1,))
    D[D <= 10e-5] = 10e-5    # Prevent infs
    diag = torch.reciprocal(torch.sqrt(D))
    # print(diag.reshape((1, -1)).shape)
    A_wave = torch.multiply(torch.multiply(diag.reshape((-1, 1)),A),
                         diag.reshape((1, -1)))

    return A_wave

def gen_TE(T,total_day):
    num_interval = total_day*T
    TE = list(set(range(0, num_interval)))
    TE = np.array(TE)
    TE = TE % T
    TE = TE.astype(np.int32)
    TE = TE[np.newaxis]

    return TE

def encode_TE(TE, T):
    sin_t = np.sin(2*np.pi*TE/T)
    cos_t = np.cos(2*np.pi*TE/T)
    TE = np.concatenate((sin_t, cos_t), axis=0)

    return TE


def encode_geolocation(lat, lon):
    if len(lat.shape) == 1:
        lat = lat.reshape(-1,1)
        lon = lon.reshape(-1,1)
    lat = lat/90
    lon = lon/180

    lat_enc = np.concatenate((np.sin(np.pi*lat), np.cos(np.pi*lat)), axis=-1)
    lon_enc = np.concatenate((np.sin(np.pi*lon), np.cos(np.pi*lon)), axis=-1)

    return np.concatenate((lat_enc, lon_enc), axis=-1)
    


def seq2instance(data, look_back):

    num_step, N, dims = data.shape
    num_sample = num_step - look_back + 1
    x = np.zeros((num_sample, look_back, N, dims))

    for i in range(num_sample):
        x[i] = data[i: i + look_back]
    return x


def free_spd(spd, days, T):
    non_peak_spd_list = []
    slot_per_h = T//24
    print("slot_per_h",slot_per_h)
    for i in range(days):
        non_peak_spd_list.append(spd[:,(i*T):(i*T+6*slot_per_h)])
        non_peak_spd_list.append(spd[:,(i*T+9*slot_per_h):(i*T+16*slot_per_h)])
        non_peak_spd_list.append(spd[:,(i*T+22*slot_per_h):(i*T+24*slot_per_h)])
    non_peak_spds = np.concatenate(non_peak_spd_list,axis=-1)
    free_spd = np.percentile(non_peak_spds,[85],axis=-1).transpose()

    return free_spd

def calculate_random_walk_matrix(adj_mx):
    """
    Returns the random walk adjacency matrix. This is for D_GCN
    """
    adj_mx = sp.coo_matrix(adj_mx)
    d = np.array(adj_mx.sum(1))
    d_inv = np.power(d, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    random_walk_mx = d_mat_inv.dot(adj_mx).tocoo()
    return random_walk_mx.toarray()

def calculate_random_walk_matrix_torch(adj_mx):
    """
    Returns the random walk adjacency matrix using PyTorch tensors.
    """
    adj_mx = torch.tensor(adj_mx, dtype=torch.float32)
    d = adj_mx.sum(dim=1)  # Row sum (degree matrix)
    d_inv = torch.where(d > 0, 1.0 / d, torch.zeros_like(d))  # Avoid division by zero
    d_mat_inv = torch.diag(d_inv)
    
    random_walk_mx = torch.matmul(d_mat_inv, adj_mx)  # Compute D^(-1) * A
    return random_walk_mx