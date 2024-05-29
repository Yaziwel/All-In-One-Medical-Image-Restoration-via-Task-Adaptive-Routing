import numpy as np 
from matplotlib import pyplot as plt  

file = "all_grads_cc12m.npz"
with open(file,'rb') as f:
    all_g = dict(np.load(f, allow_pickle=True))
    in_g = all_g['imagenet'].item()
    bw_g = all_g['bookswiki_pretrain'].item()
    cc_g = all_g['cc12m_caption'].item() 
    
keys = ['encoder.layers.3.ffn.dense2.weight', 'encoder.layers.7.ffn.dense2.weight', 'encoder.layers.11.ffn.dense2.weight']

index = 2
feat_shape = in_g[keys[index]][0].shape
in_g1 = np.array(in_g[keys[index]]).astype(np.float32).reshape(100, -1)
bw_g1 = np.array(bw_g[keys[index]]).astype(np.float32).reshape(100, -1)
cc_g1 = np.array(cc_g[keys[index]]).astype(np.float32).reshape(100, -1) 

# in_g1.shape 

threshold = 0.0
in_g1_t = in_g1 * (np.abs(in_g1) > threshold * in_g1.std(-1, keepdims=True))
bw_g1_t = bw_g1 * (np.abs(bw_g1) > threshold * bw_g1.std(-1, keepdims=True))
cc_g1_t = cc_g1 * (np.abs(cc_g1) > threshold * cc_g1.std(-1, keepdims=True)) 

# cc_g1_t.shape 

in_g1_t_norm = np.sqrt(np.sum(in_g1_t**2, -1, keepdims=True))
bw_g1_t_norm = np.sqrt(np.sum(bw_g1_t**2, -1, keepdims=True))
cc_g1_t_norm = np.sqrt(np.sum(cc_g1_t**2, -1, keepdims=True)) 

deltaii = in_g1_t @ in_g1_t.T / in_g1_t_norm
deltabi = bw_g1_t @ in_g1_t.T / bw_g1_t_norm
deltaci = cc_g1_t @ in_g1_t.T / cc_g1_t_norm
deltaib = in_g1_t @ bw_g1_t.T / in_g1_t_norm
deltabb = bw_g1_t @ bw_g1_t.T / bw_g1_t_norm
deltacb = cc_g1_t @ bw_g1_t.T / cc_g1_t_norm
deltaic = in_g1_t @ cc_g1_t.T / in_g1_t_norm
deltabc = bw_g1_t @ cc_g1_t.T / bw_g1_t_norm
deltacc = cc_g1_t @ cc_g1_t.T / cc_g1_t_norm 

# j != i
ind = 0 # sum along the j-axis
deltaii_sum = (deltaii.sum(ind)-np.diag(deltaii))/(deltaii.shape[1]-1)
deltabi_sum = (deltabi.sum(ind)-np.diag(deltabi))/(deltabi.shape[1]-1)
deltaci_sum = (deltaci.sum(ind)-np.diag(deltaci))/(deltaci.shape[1]-1)
deltaib_sum = (deltaib.sum(ind)-np.diag(deltaib))/(deltaib.shape[1]-1)
deltabb_sum = (deltabb.sum(ind)-np.diag(deltabb))/(deltabb.shape[1]-1)
deltacb_sum = (deltacb.sum(ind)-np.diag(deltacb))/(deltacb.shape[1]-1)
deltaic_sum = (deltaic.sum(ind)-np.diag(deltaic))/(deltaic.shape[1]-1)
deltabc_sum = (deltabc.sum(ind)-np.diag(deltabc))/(deltabc.shape[1]-1)
deltacc_sum = (deltacc.sum(ind)-np.diag(deltacc))/(deltacc.shape[1]-1) 

interference = np.array([[np.mean(deltaii_sum / deltaii_sum), np.mean(deltabi_sum / deltaii_sum), np.mean(deltaci_sum / deltaii_sum)],
                         [np.mean(deltaib_sum / deltabb_sum), np.mean(deltabb_sum / deltabb_sum), np.mean(deltacb_sum / deltabb_sum)],
                         [np.mean(deltaic_sum / deltacc_sum), np.mean(deltabc_sum / deltacc_sum), np.mean(deltacc_sum / deltacc_sum)]])