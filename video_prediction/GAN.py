

import torch.nn as nn
from torch.autograd import Variable
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from dataloader import NATOPSData


class Encoder(nn.Module):

    def __init__(self, kernel_size=3):
        super(Encoder, self).__init__()
        '''
        Encoder network for generator 
        input size (N,3,64,64)
        output size(N,256,8,8)
        '''
        
        self.conv1 = nn.Conv2d(3,32,kernel_size,padding=1)
        self.conv2 = nn.Conv2d(32,64,kernel_size,padding=1)
        self.bn1 = nn.BatchNorm2d(64,affine=False)
        
        self.pool = nn.AvgPool2d(2,stride=2)
        self.conv3 = nn.Conv2d(64,128,kernel_size,padding=1)
        self.bn2 = nn.BatchNorm2d(128,affine=False)
        self.conv4 = nn.Conv2d(128,256,kernel_size,padding=1)
        self.bn3 = nn.BatchNorm2d(256,affine=False)
        self.conv5 = nn.Conv2d(256,256,kernel_size,padding=1)
        self.bn4 = nn.BatchNorm2d(256,affine=False)
        
        #output (256*8*8)
    def forward(self,input_tensor):
        h1 = self.conv1(input_tensor)
        h1 = F.leaky_relu(h1)
        #print(h1.size())
        h2 = self.conv2(h1)
        h2 = self.bn1(h2)
        h2 = F.leaky_relu(h2)
        #print(h2.size())
        h3 = self.pool(h2)
        h3 = self.conv3(h3)
        h3 = self.bn2(h3)
        h3 = F.leaky_relu(h3)
        #print(h3.size())
        h4 = self.pool(h3)
        h4 = self.conv4(h4)
        h4 = self.bn3(h4)
        h4 = F.leaky_relu(h4)
        #print(h4.size())
        h5 = self.pool(h4)
        h5 = self.conv5(h5)
        h5 = self.bn4(h5)
        out = F.leaky_relu(h5)
        cache = (h2,h3,h4)

        return out,cache
        

class ConvLSTMCell(nn.Module):

    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.
        
        Parameters
        ----------
        input_size: (int, int)
            Height and width of input tensor as (height, width).
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.height, self.width = input_size
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding     = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias        = bias
        
        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        
        h_cur, c_cur = cur_state
        
        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis
        
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1) 
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next

    def init_hidden(self, batch_size):
        #return (Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width)).cuda(),
        #        Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width)).cuda())
        return (Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width)),
                Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width)))

class ConvLSTM(nn.Module):

    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=True, bias=True, return_all_layers=True):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim  = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.height, self.width = input_size

        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i-1]

            cell_list.append(ConvLSTMCell(input_size=(self.height, self.width),
                                          input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """
        
        Parameters
        ----------
        input_tensor: todo 
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful
            
        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        # Implement stateful ConvLSTM
        hidden_state = self._init_hidden(input_tensor.size(0),hidden_state)

        layer_output_list = []
        last_state_list   = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):

                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)
                
            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append(h)

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list   = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size,hidden_state):
        init_states = []
        if type(hidden_state) == None:
            for i in range(self.num_layers):
                init_states.append(self.cell_list[i].init_hidden(batch_size))
        #implement initialization with hidden state
        else:
            if not len(hidden_state)==self.num_layers:
                raise ValueError('Inconsistent list length.')
            for i in range(self.num_layers):
                init_states.append(hidden_state[i])
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                    (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


class Decoder(nn.Module):

    def __init__(self,p,kernel_size=3,upsample_size=2):
        super(Decoder, self).__init__()
        #convLSTM out: (N,256,8,8)
        #encoder out: (N,256,8,8)
        #noise z : (N,p,8,8)
        #input size:(N,512+p,8,8)
        
        self.s1 = Gate(2,3,256)
        self.s2 = Gate(4,3,128)
        self.s3 = Gate(8,3,64)
        
        self.conv6 = nn.Conv2d(512+p,256,kernel_size,padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        #LeakyReLU
        self.upsample = nn.Upsample(scale_factor=upsample_size, mode='bilinear')
        #Gating  is it gonna backprop?
        
        self.conv7 = nn.Conv2d(256,128,kernel_size,padding=1)
        self.bn7 = nn.BatchNorm2d(128)
        #LeakyReLU
        #upsample
        #Gating
        
        self.conv8 = nn.Conv2d(128,64,kernel_size,padding=1)
        self.bn8 = nn.BatchNorm2d(64)
        #LeakyReLU
        #upsample
        #Gating
        
        self.conv9 = nn.Conv2d(64,64,kernel_size,padding=1)
        self.bn9 = nn.BatchNorm2d(64)
        #LeakyReLU
        self.conv10 = nn.Conv2d(64,3,kernel_size,padding=1)
        #Tanh
        
    def forward(self,lstm_out,z,encoder_out,encoder_cache):
        #z - tiling noise size(p*8*8)
        #lstm_out:(N,512+P,8,8)
        
        e3,e2,e1 = encoder_cache
        decoder_input = torch.cat((lstm_out,encoder_out,z),1)
        u1 = self.conv6(decoder_input)
        u1 = self.bn6(u1)
        u1 = F.leaky_relu(u1)
        u1 = self.upsample(u1)
        
        s1 = self.s1(lstm_out)
        
        u2 = s1*u1 + (1-s1)*e1
        u2 = self.conv7(u2)
        u2 = self.bn7(u2)
        u2 = F.leaky_relu(u2)
        u2 = self.upsample(u2)
        
        s2 = self.s2(lstm_out)
        
        u3 = s2*u2 + (1-s2)*e2
        u3 = self.conv8(u3)
        u3 = self.bn8(u3)
        u3 = F.leaky_relu(u3)
        u3 = self.upsample(u3)
        
        s3 = self.s3(lstm_out)
        u4 = s3*u3 + (1-s3)*e3
        u4 = self.conv9(u4)
        u4 = self.bn9(u4)
        u4 = F.leaky_relu(u4)
        u4 = self.conv10(u4)
        out = F.tanh(u4)
        return out
        

class Gate(nn.Module):

    def __init__(self, upsample_size,kernel_size,num_filters):
        super(Gate, self).__init__()
        self.upsample = nn.Upsample(scale_factor=upsample_size, mode='bilinear')
        self.conv = nn.Conv2d(256,num_filters,kernel_size,padding=1)
    def forward(self,lstm_out):
        h = self.upsample(lstm_out)
        
        h = self.conv(h)
        h = F.leaky_relu(h)
        out = F.sigmoid(h)
        return out
        


# In[7]:


class Generator(nn.Module):

    def __init__(self,batch_size,q,p,kernel_size=3):
        super(Generator, self).__init__()
        self.num_keypoints = q
        self.p = p
        self.encode1 = nn.Sequential(
            nn.Conv2d(3,32,kernel_size,padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(32,64,kernel_size,padding=1),
            nn.BatchNorm2d(64,affine=False),
            nn.LeakyReLU())
        
        self.encode2 = nn.Sequential(
            nn.AvgPool2d(2,stride=2),
            nn.Conv2d(64,128,kernel_size,padding=1),
            nn.BatchNorm2d(128,affine=False),
            nn.LeakyReLU())
        
        self.encode3 = nn.Sequential(
            nn.AvgPool2d(2,stride=2),
            nn.Conv2d(128,256,kernel_size,padding=1),
            nn.BatchNorm2d(256,affine=False),
            nn.LeakyReLU()
        )
        self.encode4 = nn.Sequential(
            nn.AvgPool2d(2,stride=2),
            nn.Conv2d(256,256,kernel_size,padding=1),
            nn.BatchNorm2d(256,affine=False),
            nn.LeakyReLU()
        )
        
        self.fc_q = nn.Linear(q,1)
        
        #ConvLSTM(input_size,input_dim,hidden_dim,kernel_size,num_layer)
        self.lstm = ConvLSTM((8,8), q, 256, (3,3), 1)
        
        self.decode = Decoder(p,kernel_size,upsample_size=2)
        
        
    def forward(self,y_a,y_m,z):
        N,T,_ = y_m['velocity'].size()
        y_l = y_m['label'].view(N,1,-1).repeat(1,T,1)
        y_v = torch.cat((y_m['velocity'],y_l),2)
        #y_v (N,T,q)
        
        #y_v (N,T,q) (currently just input N,T,1)
        #y_a (N,3,64,64)
        
        #tile y_v
        q = self.fc_q(y_v)
        t = F.sigmoid(q)
        y_v = t * q + (1-t)*q
        
        #TODO: check implementation of tiling
        y_v = y_v.view(N,-1,1,1,1).repeat(1,1,self.num_keypoints,8,8)
        
        #encoder
        e1 = self.encode1(y_a)
        e2 = self.encode2(e1)
        e3 = self.encode3(e2)
        e_out = self.encode4(e3)
        initial_states = []
        #TODO: check if initial cell state is 0?
        #print(e_out.size())
        initial_cell = Variable(torch.zeros_like(e_out))
        initial_states.append((e_out,initial_cell))
        #print(y_v.size())
        
        lstm_out,_ = self.lstm(y_v, initial_states)
        #lstm_out (N,T,256,8,8)
        #print('lstm_out size', len(lstm_out),lstm_out[0].size())
        lstm_out = lstm_out[0]
        
        #create random noise z
        #z = torch.rand((N,p))
        z = z.view(N,self.p,1,1).repeat(1,1,8,8)
        
        
        decode_out = torch.zeros(N,T,3,64,64)
        #decode each time step output
        for t in range(lstm_out.size()[1]):
            decode_out[:,t,:,:,:] = self.decode(lstm_out[:,t,:,:,:],z,e_out,(e1,e2,e3))
        
        return decode_out
        
        
        


class Appearance_D(nn.Module):

    def __init__(self,kernel_size=4):
        super(Appearance_D, self).__init__()
        #network 1
        self.network1 = nn.Sequential(
            nn.Conv2d(3,64,kernel_size,stride=2,padding=1),
            nn.BatchNorm2d(64,affine=False),
            nn.LeakyReLU(),
            nn.Conv2d(64,128,kernel_size,stride=2,padding=1),
            nn.BatchNorm2d(128,affine=False),
            nn.LeakyReLU(),
            nn.Conv2d(128,256,kernel_size,stride=2,padding=1),
            nn.BatchNorm2d(256,affine=False),
            nn.LeakyReLU())
        
        #network 2
        #input to 2: (256*4,8,8)
        
        self.network2_l1 = nn.Sequential(
            nn.ConvTranspose2d(256*4,256,3,padding=1),
            nn.BatchNorm2d(256,affine=False),
            nn.LeakyReLU())
        self.network2_l2 = nn.Sequential(
            nn.Conv2d(256,512,kernel_size,stride=2,padding=1),
            nn.BatchNorm2d(512,affine=False),
            nn.LeakyReLU())
        self.network2_l3 = nn.Sequential(
            nn.Conv2d(512,1024,kernel_size,stride=4),
            nn.BatchNorm2d(1024,affine=False),
            nn.LeakyReLU())
        self.fc = nn.Sequential(
            nn.Linear(1024,64),
            nn.LeakyReLU(),
            nn.Linear(64,1),
            nn.Sigmoid())
      
    def forward(self,x,y_a):
        '''
        -input x(N,T,3,64,64)
               y(1,3,64,64)
        -output predictions torch tensor of (N)
        '''
        N,T,_,_,_ = x.size()
        y_out = self.network1(y_a) #(1,256,8,8)
        predictions = torch.zeros(N,T-3)
        layer1_out = torch.zeros(N,T-3,256,8,8)
        layer2_out = torch.zeros(N,T-3,512,4,4)
        for t in range(T-3):
            #x_temp (N,3,64,64)
            x_temp_1 = x[:,t,:,:,:]
            x_temp_2 = x[:,t+1,:,:,:]
            x_temp_3 = x[:,t+2,:,:,:]
            x_out_1 = self.network1(x_temp_1) #(N,256,8,8)
            
            x_out_2 = self.network1(x_temp_2) #(N,256,8,8)
            
            x_out_3 = self.network1(x_temp_3) #(N,256,8,8)
            x_1 = torch.cat((x_out_1,x_out_2,x_out_3,y_out),1) #(N,256*4,8,8)
            x_2 = self.network2_l1(x_1)
            layer1_out[:,t,:,:,:] = x_2
            
            x_2 = self.network2_l2(x_2)
            layer2_out[:,t,:,:,:] = x_2
            
            x_2 = self.network2_l3(x_2)
            
            x_2 = torch.squeeze(x_2) #(N,1024)
            out = self.fc(x_2) #(N,1)
            predictions[:,t] = torch.squeeze(out)
            
        predictions = torch.mean(predictions,1)
        #prediction is the average across all time steps
            
        return predictions,layer1_out,layer2_out
        
        



class Motion_D(nn.Module):

    def __init__(self,q,num_classes,kernel_size=4):
        super(Motion_D, self).__init__()
        #input y_a (3*64*64) 
        #y_l (c*4*4) ??
        #x (3*64*64)
        self.q = q
        self.encoder = nn.Sequential(
            nn.Conv2d(3,64,kernel_size,stride=2,padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64,128,kernel_size,stride=2,padding=1),
            nn.BatchNorm2d(128,affine=False),
            nn.LeakyReLU(),
            nn.Conv2d(128,256,kernel_size,stride=2,padding=1),
            nn.BatchNorm2d(256,affine=False),
            nn.LeakyReLU())
        
        #ConvLSTM(input_size,input_dim,hidden_dim,kernel_size,num_layer)
        self.lstm = ConvLSTM((8,8), 256, 256, (3,3), 1)
        
        #last hidden state h_out (N*256*8*8)
        self.conv4 = nn.Conv2d(256,64,kernel_size,stride=2,padding=1)
        self.bn4 = nn.BatchNorm2d(64,affine=False)
        #LeakyRelu -> flatten
        self.fc_h1 = nn.Linear(1024,64)
        self.bn_fc1 = nn.BatchNorm1d(64,affine=False)
        #LeakyRelu
        self.fc_h2 = nn.Linear(64,num_classes)
        self.bn_fc2 = nn.BatchNorm1d(num_classes,affine=False)
        #LeakyRelu -> softmax
        
        #output: ot (N,256,8,8)
        self.conv6 = nn.Conv2d(256,64,kernel_size,stride=2,padding=1)
        self.bn6 = nn.BatchNorm2d(64,affine=False)
        #leakyRelu -> flatten
        self.fc_o = nn.Linear(1024,q)
        
        #concatenate y_l (c*4*4) + (64*4*4) = (c+64,4,4)
        self.conv5 = nn.Conv2d(num_classes+64,64,kernel_size,stride=2)
        self.bn5 = nn.BatchNorm1d(64,affine=False)
        #leakyrelu
        self.fc_y = nn.Linear(64,1)
        #sigmoid
        
        #TODO forward
    def forward(self,x,y_a,y_m):
        #print('x shape',x.size())
        y_l = y_m['label']
        N = x.size()[0]
        T = x.size()[1]
        # x (N,T,3,64,64)
        # y_a (N,3,64,64)
        #y_l = (N,C)
        x = x.view(-1,3,64,64)
        
        x_encoded = self.encoder(x)
        y_encoded = self.encoder(y_a)
        
        x_encoded = x_encoded.view(N,T,256,8,8)
        
        initial_states = []
        #TODO: check if initial cell state is 0?
        initial_cell = Variable(torch.zeros_like(y_encoded))
        initial_states.append((y_encoded,initial_cell))
        
        
        o,h_last = self.lstm(x_encoded, initial_states)
        o = o[0]
        h_last = h_last[0]
        
        h_out = self.conv4(h_last)
        h_out = self.bn4(h_out)
        h_out = F.leaky_relu(h_out)
        
        #print('h_out shape',h_out.size())
        l_out = h_out.view(N,-1)
        l_out = self.fc_h1(l_out)
        l_out = self.bn_fc1(l_out)
        l_out = F.leaky_relu(l_out)
        l_out = self.fc_h2(l_out)
        l_out = self.bn_fc2(l_out)
        l_out = F.softmax(F.leaky_relu(l_out))
        #y_m_l output (label predicted)
        
        #concatenate y_l and h_out
        #print('yl',y_l.shape)
        y_l = y_l.view(N,-1,1,1).repeat(1,1,4,4)
        #print('y_l shape ', y_l.size())
        predict = torch.cat((y_l,h_out),1)
        #print('predict shape', predict.size())
        predict = self.conv5(predict)
        predict = torch.squeeze(predict)
        #print('predict shape after squeeze',predict.size())
        predict = self.bn5(predict)
        predict = F.leaky_relu(predict)
        predict = self.fc_y(predict)
        predict_out = F.sigmoid(predict)
        
        #predict keypoints 
        keypoints = torch.zeros(N,T,self.q)
        #decode each time step output
        for t in range(T):
            input_o = o[:,t,:,:,:]
            k_out = self.conv6(input_o)
            k_out = self.bn6(k_out)
            #print('k_out shape',k_out.size())
            k_out = F.leaky_relu(k_out)
            k_out = k_out.view(N,-1)
            k_out = self.fc_o(k_out)
            k_out = F.tanh(k_out)
            #print('k_out each time step',k_out.size())
            keypoints[:,t,:] = k_out
        #print(keypoints.size())
        return predict_out, keypoints,l_out
        
        
        
def ranking_loss(discriminator_type,d_m_1,d_m_2,d_a_1,d_a_2):
    """
    Inputs: discriminator_type 'a' for appearance 'm' for motion
    d_m_1 triplets of Dm layer1 output of ((x|y,y),(x_hat|y,y),(x_hat|y',y'))
    Pytorch variable of shape (N,T,C,H,W)
    
    Outputs: the amount of violation for d(x_y, x_hat_y) < d(x_y, x_hat_y_prime)
    
    """
    
    G1,G1_pos_hat, G1_neg_hat = None,None,None
    G2,G2_pos_hat, G2_neg_hat = None,None,None
    if discriminator_type == 'a':
        #Using Da
        G1 = gram_matrix(d_a_1[0])
        G1_pos_hat = gram_matrix(d_a_1[1])
        G1_neg_hat = gram_matrix(d_a_1[2])
        
        G2 = gram_matrix(d_a_2[0])
        G2_pos_hat = gram_matrix(d_a_2[1])
        G2_neg_hat = gram_matrix(d_a_2[2])
        
        
    else:
        #Using Dm
        
        G1 = gram_matrix(d_m_1[0])
        G1_pos_hat = gram_matrix(d_m_1[1])
        G1_neg_hat = gram_matrix(d_m_1[2])
        
        G2 = gram_matrix(d_m_2[0])
        G2_pos_hat = gram_matrix(d_m_2[1])
        G2_neg_hat = gram_matrix(d_m_2[2])
        
    N,_,_ = G1.size()
    
    
    d1_pos = torch.norm(G1 - G1_pos_hat)/N
    d1_neg = torch.norm(G1 - G1_neg_hat)/N
    d2_pos = torch.norm(G2 - G2_pos_hat)/N
    d2_neg = torch.norm(G2 - G2_neg_hat)/N
    
    
    return d1_pos, d1_neg,d2_pos,d2_neg

    
    


def gram_matrix(features, normalize=True):
    """
    Compute the Gram matrix from features.
    
    Inputs:
    - features: PyTorch Variable of shape (N, T, C, H, W) giving features for
      a batch of N videos.
    - normalize: optional, whether to normalize the Gram matrix
        If True, divide the Gram matrix by the number of neurons (H * W * C)
    
    Returns:
    - gram: PyTorch Variable of shape (N, C, C) giving the
      (optionally normalized) Gram matrices for the N input images.
    """
    N,T,C,H, W = features.shape
    #G = features[..., None] * features[..., None].permute(0, 5, 3, 4, 1,2)
    w = features.view(N,T*C,H*W)
    G = torch.matmul(w.permute(0,2,1),w)
    if normalize:
        return G / (H * W * C*T)
    else:
        return G


def bce_loss(input, target):
    """
    Numerically stable version of the binary cross-entropy loss function.

    As per https://github.com/pytorch/pytorch/issues/751
    See the TensorFlow docs for a derivation of this formula:
    https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits

    Inputs:
    - input: PyTorch Variable of shape (N, ) giving scores.
    - target: PyTorch Variable of shape (N,) containing 0 and 1 giving targets.

    Returns:
    - A PyTorch Variable containing the mean BCE loss over the minibatch of input data.
    """
    ###########################
    ######### TO DO ###########
    ###########################
    loss = None
    neg_abs = - input.abs()
    loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
    loss = loss.mean()
    return loss


def discriminator_loss(logits_r,logits_m,logits_f,logits_f_prime):
    """
    Computes the discriminator loss described in the homework pdf
    using the bce_loss function.
    
    Inputs:
    - logits_r: PyTorch Variable of shape (N,) giving scores for the real data.
    - logits_f: PyTorch Variable of shape (N,) giving scores for the fake data.
    
    Returns:
    - loss: PyTorch Variable containing (scalar) the loss for the discriminator.
    """
    ###########################
    ######### TO DO ###########
    ###########################
    loss = None
    loss = torch.log(logits_r) + 0.5*(torch.log(1-logits_m)+0.5*(torch.log(1-logits_f)+torch.log(1-logits_f_prime)))
    loss = loss.mean()
    return loss


def discriminator_loss_aux(y_m,y_m_prime,v_r,v_f,v_f_prime,l_r,l_f,l_f_prime):
    '''
    y_m, y_m_prime: dictionary 
    y_m['velocity'] = (N,T,q)
    v_r, v_f,v_f_prime = (N,T,q)
    l_r,l_f,l_f_prime = (N,C)
    
    '''
    y_v = y_m['velocity']
    y_v_prime = y_m_prime['velocity']
    y_l = y_m['label']
    #one_hot encode to scalar
    y_l = (y_l == 1).nonzero()[:,1]
    
    LMSE = F.mse_loss(y_v,v_r) + F.mse_loss(y_v,v_f)+ F.mse_loss(y_v_prime,v_f_prime)
    
    LCE = F.cross_entropy(l_r, y_l) + F.cross_entropy(l_f, y_l) + F.cross_entropy(l_f_prime, y_l)
    
    return LMSE + LCE
    



def train_step(X,Y,D_m,D_a, G, D_m_solver,D_a_solver, G_solver,batch_size=2,num_epochs=10,q=42,p=128):
    #X,Y = loader_train
    '''
    -D_m,D_a,G: models for discriminator and generator
    -D_m_solver,D_a_solver, G_solver: torch.optim Optimizers to use for training the
      discriminator and generator.
      
    X: x|y of size N,T,3,64,64
    Y:triplet of (ya,ym), (ya',ym), (ya,ym')  
    ya=(N,3,64,64) ym:pair of label,velocity ym['label'] = (N,C) ym['velocity'] = (N,T,q)
    
    
    '''
    x_y = X
    (y_a,y_m),(y_a_prime,_),(_,y_m_prime)=Y
    N = batch_size
    z = torch.rand((N,p))  #3
    
    x_hat_y = G(y_a,y_m,z).detach()
    x_hat_y_a = G(y_a_prime,y_m,z).detach()
    x_hat_y_m = G(y_a,y_m_prime,z).detach()
    #4
    
    
    
    #train discriminator a
    D_a_solver.zero_grad()
    s_r_a,_,_ = D_a(x_y,y_a)
    s_f_a,_,_ = D_a(x_hat_y,y_a)
    s_m_a,_,_ = D_a(x_y,y_a_prime)
    s_f_prime_a,_,_ = D_a(x_hat_y_a,y_a_prime)
    
    d_a_loss = discriminator_loss(s_r_a,s_m_a,s_f_a,s_f_prime_a)
    
    d_a_loss.backward()
    D_a_solver.step()
    #TODO:verify Daux is not needed for D_a
    
    #train discriminator m
    D_m_solver.zero_grad()
    s_r_m,v_r,l_r = D_m(x_y,y_a,y_m)
    s_f_m,v_f,l_f = D_m(x_hat_y,y_a,y_m)
    s_m_m,v_m,l_m = D_m(x_y,y_a,y_m_prime)
    s_f_prime_m,v_f_prime,l_f_prime = D_m(x_hat_y_m,y_a,y_m_prime)
    
    d_m_loss = discriminator_loss(s_r_m,s_m_m,s_f_m,s_f_prime_m)
    
    d_m_loss_aux = discriminator_loss_aux(y_m,y_m_prime,v_r,v_f,v_f_prime,l_r,l_f,l_f_prime)
    
    #TODO:verify '-' in paper
    d_m_loss -= d_m_loss_aux
    
    d_m_loss.backward()
    D_m_solver.step()
    
    
    #generator loss 
    G_solver.zero_grad()
    #x_y
    x_hat_y = G(y_a,y_m,z)
    x_hat_y_a = G(y_a_prime,y_m,z)
    x_hat_y_m = G(y_a,y_m_prime,z)
    
    #using appearance discriminator
    discriminator_type = 'a'
    _,r_l1_a,r_l2_a = D_a(x_y,y_a)
    s_f_a,f_l1_a,f_l2_a = D_a(x_hat_y_a,y_a)
    s_f_prime_a, f_prime_l1_a,f_prime_l2_a = D_a(x_hat_y_a,y_a_prime)
    
    #(x|y,y),(x_hat|y,y),(x_hat|y',y')
    d_a_1 = (r_l1_a,f_l1_a,f_prime_l1_a)
    d_a_2 = (r_l2_a,f_l2_a,f_prime_l2_a)
    d1_pos_a, d1_neg_a,d2_pos_a,d2_neg_a = ranking_loss(discriminator_type,None,None,d_a_1,d_a_2)
    
    G_loss_a = s_f_a.log() + s_f_prime_a.log()
    G_loss_a = G_loss_a.mean()
    G_loss_aux = F.l1_loss(x_hat_y[1:],x_y[1:]) + d1_pos_a+d2_pos_a
    
    G_rank_loss_a = torch.clamp(0.01-d1_neg_a+d1_pos_a,max=0) + torch.clamp(0.01-d2_neg_a+d2_pos_a,max=0)
    
    
    #using motion discriminator
    discriminator_type = 'm'
    #_,v_r,l_r = D_m(x_y,y_a,y_m).detach()
    s_f_m,v_f,l_f = D_m(x_hat_y,y_a,y_m)
    s_f_prime_m,v_f_prime,l_f_prime = D_m(x_hat_y_m,y_a,y_m_prime)
    
    G_loss_m = s_f_m.log() + s_f_prime_m.log()
    G_loss_m = G_loss_m.mean()
    #TODO: add ranking loss for motion
    
    G_loss = G_loss_a - G_loss_aux - G_rank_loss_a + G_loss_m
    
    
    #torch.clamp(0.0001-d1_neg+d1_pos,max=0) + torch.clamp(0.0001-d2_neg+d2_pos,max=0)
        
    G_loss.backward()
    G_solver.step()
    
    return G_loss,d_m_loss,d_a_loss,d_m_loss_aux,G_loss_a,G_loss_aux,G_rank_loss_a,G_loss_m
    
    


def train(train_loader,batch_size,T,q,p,c,num_epochs,device): 
    G = Generator(batch_size,q,p).to(device)
    D_m = Motion_D(18,c).to(device)
    D_a = Appearance_D().to(device)
    G_solver = optim.Adam(G.parameters(), lr=1e-3,betas=(0.5, 0.999))
    D_m_solver = optim.Adam(D_m.parameters(), lr=1e-3,betas=(0.5, 0.999))
    D_a_solver = optim.Adam(D_a.parameters(), lr=1e-3,betas=(0.5, 0.999))
    
    for epoch in range(num_epochs):  # TODO decide epochs
        print('-----------------Epoch = %d-----------------' % (epoch + 1))
        for step,(X,Y) in enumerate(train_loader):
            X = X.to(device)
            X = X.permute(0,1,4,2,3)
            X = X.type(torch.FloatTensor)
            #Y = Y.to(device)
            y_a,y_m = Y
            y_a = y_a.permute(0,3,1,2)
            y_a = y_a.type(torch.FloatTensor)
            y_m_l = y_m['label']
            y_m_l = y_m_l.type(torch.FloatTensor)
            y_m_v = y_m['velocity']
            y_m_v = y_m_v.type(torch.FloatTensor)
             #shuffle y_a to get y_a',y_m
            r=torch.randperm(batch_size)
            y_a_prime = y_a[r]

            r=torch.randperm(batch_size)
            l_prime = y_m_l[r]
            k_prime = y_m_v[r]
            l_prime = l_prime.to(device)
            k_prime = k_prime.to(device)
            y_m_l = y_m_l.to(device)
            y_m_v = y_m_v.to(device)
            y_m_prime = {'label':l_prime,'velocity':k_prime}
            y_m = {'label':y_m_l,'velocity':y_m_v}
            y_a = y_a.to(device)
            #y_m = y_m.to(device)
            y_a_prime = y_a_prime.to(device)
            #y_m_prime = y_m_prime.to(device)
            Y = ((y_a,y_m),(y_a_prime,y_m),(y_a,y_m_prime))
            
            G_loss,d_m_loss,d_a_loss,d_m_loss_aux,G_loss_a,G_loss_aux,G_rank_loss_a,G_loss_m = train_step(X,Y,D_m,D_a, G, D_m_solver,
                                                  D_a_solver, G_solver, batch_size,num_epochs,q,p)
            if(step%100==0):
                print("step%d G_loss%.3f d_m_loss%.3f d_a_loss%.3f d_m_loss_aux%.3f G_loss_a%.3f G_loss_aux%.3f G_rank_loss_a%.3f G_loss_m%.3f" %(step,G_loss.item(),d_m_loss.item(),
                                                                     d_a_loss.item(),d_m_loss_aux.item(),G_loss_a.item(), G_loss_aux.item(),G_rank_loss_a.item(),G_loss_m.item()))
            

def main():
    dset_train = NATOPSData("videos/reshaped.hdf5","natops/data/segmentation.txt","keypoints.h5")
    train_loader = DataLoader(dset_train, batch_size=10, shuffle=True, num_workers=1)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    train(train_loader,batch_size=10,T=30,q=42,p=128,c=24,num_epochs=10,device="cpu")
if __name__ == '__main__':
    main()

