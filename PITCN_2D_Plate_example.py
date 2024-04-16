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

load_flag = False
opt_flag  = 'LBFGS'

if load_flag == True:
    PATH  = './tcn_rect_plate_model.ckpt'
    model = torch.load(PATH)
    model.eval()
    print('Model is loaded')

epochs_adam    = 1
lr_rate_adam   = 0.001
optimizer_adam = torch.optim.Adam(model.parameters(), lr=lr_rate_adam)

epochs_lbfgs    = 10000
LBFGS_max_iter  = 1
lr_rate_lbfgs   = 1.6
history_size    = 100
optimizer_lbgfs = torch.optim.LBFGS(model.parameters(), lr=lr_rate_lbfgs, history_size = history_size, max_iter=LBFGS_max_iter, line_search_fn='strong_wolfe', tolerance_change=1e-9, tolerance_grad=1e-9)

# In[26]:

def temperature_change_rate(T):
  # Calculate the temperature change rate using the finite difference method
    T0    = torch.zeros((T.shape[0], 1, 1)).double()
    T     = torch.cat((T0, T), axis = 1)
    dT_dt = torch.div((T[:, 1:, 0] - T[:, :-1, 0]), torch.tensor(dt_array).double()).unsqueeze(2)
    return dT_dt

# In[27]:

def get_T(inputs):
    T_net = model(inputs)
    coeff =  torch.einsum('ij,ij->ij', inputs[:,:,2], (inputs[:,:,2] - Length)).unsqueeze(2) / (Length**2)
    T     = (inputs[:,:,2]/Length * DTaT_np).unsqueeze(2) + torch.einsum('ijk,ijk->ijk', T_net, coeff) + ((Length - inputs[:,:,2])/Length * DTaB_np).unsqueeze(2)
    return T


# In[29]:
def temperature_gradient(T, inputs):
    # input features: time, x_coord, y_coord, Edot_tr
    gradT = torch.empty((len(inputs), Nincr, 2))
    dTdxy = torch.autograd.grad(T, inputs, torch.ones((inputs.size()[0],inputs.size()[1], 1), device=device),create_graph=True, retain_graph=True)[0]
    dTdx = dTdxy[:, :, 1] # w.r.t. x_coord
    dTdy = dTdxy[:, :, 2] # w.r.t. y_coord
    gradT[:,:,0] = dTdx
    gradT[:,:,1] = dTdy
    return gradT


# In[30]:


def heat_flux(T, inputs):
    gradT = temperature_gradient(T, inputs)
    flux  = - k_np * gradT
    return flux


# In[31]:
def flux_divergence(q, inputs):
    # input features: time, x_coord, y_coord, Edot_tr
    dq1dxy = torch.autograd.grad(q[:,:,0].unsqueeze(2), inputs, torch.ones((inputs.size()[0],inputs.size()[1], 1), device=device), create_graph=True, retain_graph=True)[0]
    dq2dxy = torch.autograd.grad(q[:,:,1].unsqueeze(2), inputs, torch.ones((inputs.size()[0],inputs.size()[1], 1), device=device), create_graph=True, retain_graph=True)[0]
    dq1dx  = dq1dxy[:, :, 1].unsqueeze(2) # w.r.t. x_coord
    dq1dy  = dq1dxy[:, :, 2].unsqueeze(2) # w.r.t. y_coord
    dq2dx  = dq2dxy[:, :, 1].unsqueeze(2) # w.r.t. x_coord
    dq2dy  = dq2dxy[:, :, 2].unsqueeze(2) # w.r.t. y_coord
    div_q  = dq1dx + dq2dy
    return div_q


# In[32]:


mse_metric = torch.nn.MSELoss()
# loss_term = torch.nn.HuberLoss(reduction='mean', delta=1.0)
loss_term = torch.nn.MSELoss()

Scale_PDE    = 1e0
l_reg_lambda = 1e-8

# In[32]:

def loss_function(epoch, inputs_g, T_g, flux_g, inputs_o):
    
    Edot_tr_g     = inputs_g[:,:,3].unsqueeze(2)/Scale_Edot
    T_ntwrk_g     = get_T(inputs_g)   
    q_g           = heat_flux(T_ntwrk_g, inputs_g)
    div_q         = flux_divergence(q_g, inputs_g)
    Tdot          = temperature_change_rate(T_ntwrk_g)
    R             = cV_np * Tdot + kappa_np * T0_np * Edot_tr_g + div_q
    # L1            = loss_term(R, torch.zeros_like(R))
    L1            = LA.norm(R)
    
    # NBC residual
    T_ntwrk_o     = get_T(inputs_o).double()
    q_o           = heat_flux(T_ntwrk_o, inputs_o)
    # L2            = loss_term(q_o[:,0], torch.zeros_like(q_o[:,0]))
    L2            = LA.norm(q_o[:,0])

    # Temperature data at Gauss points
    R_Tdata_g     = T_ntwrk_g - T_g
    # L3            = loss_term(R_Tdata_g, torch.zeros_like(R_Tdata_g))
    L3            = LA.norm(R_Tdata_g)

    # Flux data at Gauss points
    R_qdata_g     = q_g - flux_g
    # L4            = loss_term(R_qdata_g, torch.zeros_like(R_qdata_g))
    L4            = LA.norm(R_qdata_g)

    # Total loss
    loss          = L1 + 200*L2 + L3 + L4 
    print(' Epoch=', epoch, ' LOSS=', loss.item(), ' PDE=', L1.item(), ' fluxBC=', L2.item(), ' Tdata=', L3.item(), ' qdata=', L4.item())
    return loss


# In[34]:
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
print(count_parameters(model))

PATH = './tcn_rect_plate_model.ckpt'

loss_history = {}
Temp_history = np.zeros((Nincr, len(coord_n), 1))
tempL        = []
for epoch in range(epochs_adam):
    def closure():
        loss    = loss_function(epoch, inputs_g, T_g, flux_g, inputs_o)
        optimizer_adam.zero_grad()
        loss.backward(retain_graph=True)
        tempL.append(loss.item())
        return loss
    optimizer_adam.step(closure)
    if ConvergenceCheck(tempL , rel_tol_network):
        break

for epoch in range(epochs_lbfgs):
    def closure():
        loss    = loss_function(epoch, inputs_g, T_g, flux_g, inputs_o)
        optimizer_lbgfs.zero_grad()
        loss.backward(retain_graph=True)
        tempL.append(loss.item())
        return loss
    optimizer_lbgfs.step(closure)
    if epoch % 100 == 0:
        torch.save(model, PATH)
        print('Model is saved')
    if ConvergenceCheck(tempL , rel_tol_network):
        break

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
    rel_error_T[key2]  = torch.div(abs_error_T[key2], torch.tensor(T_n_GT[key]))   


# Separate the columns
inc_plot = 49
x  = coord_n[:, 0]
y  = coord_n[:, 1]
T  = T_ntwrk_n[:,inc_plot,:].squeeze()
Ea = abs_error_T[inc_plot].cpu().numpy()
Er = rel_error_T[inc_plot].cpu().numpy()

# Create a regular grid to interpolate onto
grid_x, grid_y = np.mgrid[min(x):max(x):100j, min(y):max(y):100j]

# Interpolate the data
grid_T  = griddata((x, y), T, (grid_x, grid_y), method='cubic')
grid_Ea = griddata((x, y), Ea, (grid_x, grid_y), method='cubic')
grid_Er = griddata((x, y), Er, (grid_x, grid_y), method='cubic')

# Create the contour plot
plt.figure()
plt.contourf(grid_x, grid_y, grid_T)
plt.xlabel('X')
plt.ylabel('Y')
plt.colorbar(label='Temperature')
plt.title('Temperature Contours')
plt.show() 

# Create the contour plot
plt.figure()
plt.contourf(grid_x, grid_y, grid_Ea)
plt.xlabel('X')
plt.ylabel('Y')
plt.colorbar()
plt.title('Absolute Error')
plt.show() 

# Create the contour plot
plt.figure()
plt.contourf(grid_x, grid_y, grid_Er)
plt.xlabel('X')
plt.ylabel('Y')
plt.colorbar()
plt.title('Relative Error')
plt.show() 
