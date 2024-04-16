# In[16]:
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
import torch
print(torch.__version__)
import torch.nn as nn
import torch.nn.functional as F
from tcn import TemporalConvNet as TCN
import rff
from torch import linalg as LA
import json
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from scipy.interpolate import griddata


# In[15]:
Length = 1 
E        = 70e3
nu       = 0.3
lmbda_np = E*nu/((1+nu)*(1-2*nu))
mu_np    = E/2/(1+nu)
rho_np   = 2700.
alpha    = 2.31e-5  # thermal expansion coefficient
kappa_np = alpha*(2*mu_np + 3*lmbda_np)
cV_np    = 910e-6 * rho_np
k_np     = 237e-6
T0_np    = 293.
DTaT_np  = 50
DTaB_np  = 10    
Nincr     = 50
t         = np.logspace(1, 3, Nincr+1)
dt_array  = np.diff(t)
rel_tol_network = 1e-10

device = torch.device('cpu')
if torch.cuda.is_available():
    print("CUDA is available, running on GPU")
    device = torch.device('cuda')
    device_string = 'cuda'
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    device_string = 'cpu'
    print("CUDA not available, running on CPU")
    
# In[16]:

def ConvergenceCheck( arry , rel_tol ):
    num_check = 20

    # Run minimum of 2*num_check iterations
    if len( arry ) < 2 * num_check :
        return False

    mean1 = np.mean( arry[ -2*num_check : -num_check ] )
    mean2 = np.mean( arry[ -num_check : ] )

    if np.abs( mean2 ) < 1e-6:
        print('Loss value converged to abs tol of 1e-6' )
        return True     

    if ( np.abs( mean1 - mean2 ) / np.abs( mean2 ) ) < rel_tol:
        print('Loss value converged to rel tol of ' + str(rel_tol) )
        return True
    else:
        return False
    
# In[18]:

Scale_Edot    = 1
D_in          = 4   # time, x_coord, y_coord, Edot_tr
D_out         = 1   # temperature


torch.manual_seed(2020)
torch.set_printoptions(precision=5)

T_n       = np.load('T_n.npy')
flux_n    = np.load('flux_n.npy')
inputs_n  = np.load('inputs_n.npy')
T_g       = np.load('T_g.npy')
flux_g    = np.load('flux_g.npy')
inputs_g  = np.load('inputs_g.npy')
inputs_o  = np.load('inputs_o.npy')
coord_g_T = np.load('coord_g_T.npy')
coord_n   = np.load('coord_n.npy')

T_n      = torch.tensor(T_n).double()
flux_n   = torch.tensor(flux_n).double()
inputs_n = torch.tensor(inputs_n).double()
inputs_n = inputs_n.to(device)
inputs_n.requires_grad_(True);   inputs_n.retain_grad()

T_g      = torch.tensor(T_g).double()
flux_g   = torch.tensor(flux_g).double()
inputs_g = torch.tensor(inputs_g).double()
inputs_g = inputs_g.to(device)
inputs_g.requires_grad_(True);   inputs_g.retain_grad()

inputs_o = torch.tensor(inputs_o).double()
inputs_o = inputs_o.to(device)
inputs_o.requires_grad_(True);   inputs_o.retain_grad() 


# Load dictionary
with open('T_n_GT.json', 'r') as handle:
    T_n_GT = json.load(handle)
# Convert lists back to numpy arrays
for key, value in T_n_GT.items():
    T_n_GT[key] = np.reshape(value["data"], value["shape"])

# Load dictionary
with open('ux_n_GT.json', 'r') as handle:
    ux_n_GT = json.load(handle)
# Convert lists back to numpy arrays
for key, value in ux_n_GT.items():
    ux_n_GT[key] = np.reshape(value["data"], value["shape"])

# Load dictionary
with open('uy_n_GT.json', 'r') as handle:
    uy_n_GT = json.load(handle)
# Convert lists back to numpy arrays
for key, value in uy_n_GT.items():
    uy_n_GT[key] = np.reshape(value["data"], value["shape"])
    
# In[24]:
class Seq2Seq(nn.Module):
    def __init__(self, input_size, output_size):
        super(Seq2Seq, self).__init__()
        num_channels1  = [16] * 4
        num_channels2  = [16] * 4
        # num_channels3  = [128] * 5
        enc_out_size   = 8
        act_func       = 'tanh' # tanh relu silu

        self.tcn1    = TCN(input_size,        num_channels1, act_func, kernel_size=11, dropout=0.00).double() 
        self.tcn2    = TCN(num_channels1[-1], num_channels2, act_func, kernel_size=11, dropout=0.00).double()
        # self.tcn3    = TCN(num_channels2[-1], num_channels3, act_func, kernel_size=8, dropout=0.00).double()
        self.encd    = rff.layers.GaussianEncoding(sigma=0.07, input_size=num_channels2[-1], encoded_size=enc_out_size).double()
        self.linear1 = nn.Linear(2*enc_out_size, output_size).double()
        self.init_weights()

    def init_weights(self):
        self.linear1.weight.data.normal_(0, 0.1)

    def forward(self, x):
        y1  = self.tcn1(x.transpose(1,2))
        y2  = self.tcn2(y1)
        # y3  = self.tcn3(y2)
        y4  = self.encd(y2.transpose(1,2))
        y5  = self.linear1(y4)
        return y5

# In[25]:
model = Seq2Seq(input_size=D_in, output_size=D_out)
model = model.to(device)

load_flag = True
opt_flag  = 'LBFGS'

if load_flag == True:
    PATH  = './tcn_rect_plate_model.ckpt'
    model = torch.load(PATH)
    model.eval()
    print('Model is loaded')

# In[27]:
def get_T(inputs):
    T_net = model(inputs)
    coeff =  torch.einsum('ij,ij->ij', inputs[:,:,2], (inputs[:,:,2] - Length)).unsqueeze(2) / (Length**2)
    T     = (inputs[:,:,2]/Length * DTaT_np).unsqueeze(2) + torch.einsum('ijk,ijk->ijk', T_net, coeff) + ((Length - inputs[:,:,2])/Length * DTaB_np).unsqueeze(2)
    return T

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
print(count_parameters(model))

# In[37]:
T_ntwrk_n1  = get_T(inputs_n[:coord_n.shape[0]//4, :, :]).cpu().detach().numpy()
T_ntwrk_n2  = get_T(inputs_n[coord_n.shape[0]//4:2*coord_n.shape[0]//4, :, :]).cpu().detach().numpy()
T_ntwrk_n3  = get_T(inputs_n[2*coord_n.shape[0]//4:3*coord_n.shape[0]//4, :, :]).cpu().detach().numpy()
T_ntwrk_n4  = get_T(inputs_n[3*coord_n.shape[0]//4:, :, :]).cpu().detach().numpy()
T_ntwrk_n   = np.concatenate((T_ntwrk_n1, T_ntwrk_n2, T_ntwrk_n3, T_ntwrk_n4), axis = 0)

# Error evaluation
T_n_ifenn  = {}
abs_error_T  = {}
rel_error_T  = {}
for key in T_n_GT.keys():
    key2               = int(key)
    T_n_ifenn[key2]    = torch.tensor(T_ntwrk_n[:,key2,:].squeeze()).double() 
    abs_error_T[key2]  = torch.absolute(T_n_ifenn[key2]  - torch.tensor(T_n_GT[key]))
    rel_error_T[key2]  = 100*torch.div(abs_error_T[key2], torch.tensor(T_n_GT[key]))   

# Separate the columns
inc_plot = 49
x  = coord_n[:, 0]
y  = coord_n[:, 1]
T  = T_ntwrk_n[:,inc_plot,:].squeeze()
Ea = abs_error_T[inc_plot].cpu().numpy()
Er = rel_error_T[inc_plot].cpu().numpy()

x = coord_n[:, 0]
y = coord_n[:, 1]
grid_x, grid_y = np.mgrid[x.min():x.max():75j, y.min():y.max():75j]
grid_zT  = griddata((x, y), T, (grid_x, grid_y), method='linear')
grid_zEa = griddata((x, y), Ea, (grid_x, grid_y), method='linear')
grid_zEr = griddata((x, y), Er, (grid_x, grid_y), method='linear')

plt.figure()
plt.imshow(grid_zT.T, extent=(x.min(), x.max(), y.min(), y.max()), origin='lower', cmap='viridis', aspect='auto')
plt.colorbar()
plt.title('Temperature variation in C')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

plt.figure()
plt.imshow(grid_zEa.T, extent=(x.min(), x.max(), y.min(), y.max()), origin='lower', cmap='viridis', aspect='auto')
plt.colorbar()
plt.title('Abs Error in C')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

plt.figure()
plt.imshow(grid_zEr.T, extent=(x.min(), x.max(), y.min(), y.max()), origin='lower', cmap='viridis', aspect='auto')
plt.colorbar()
plt.title('Relative Error (%)')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
