import torch
import torch.nn as nn
import torch.nn.functional as F

# class LocalShift(nn.Module):
#     def __init__(self, dim):
#         super(LocalShift, self).__init__()
#         # Define the layers for training
#         self.conv1x1 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, groups=dim, bias=False)
#         self.conv3x3 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=1, groups=dim, bias=False)
#         self.conv5x5 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=5, padding=2, groups=dim, bias=False) 
#         self.alpha = nn.Parameter(torch.randn(4), requires_grad=True) 
        

#         # Define the layers for testing
#         self.conv5x5_reparam = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=5, padding=2, groups=dim, bias = False) 
#         self.repram_flag = True

#     def forward_train(self, x):
#         out1x1 = self.conv1x1(x)
#         out3x3 = self.conv3x3(x)
#         out5x5 = self.conv5x5(x) 
#         # import pdb 
#         # pdb.set_trace() 
        
        
#         out = self.alpha[0]*x + self.alpha[1]*out1x1 + self.alpha[2]*out3x3 + self.alpha[3]*out5x5
#         return out

#     def reparam_5x5(self):
#         # Combine the parameters of conv1x1, conv3x3, and conv5x5 to form a single 5x5 depth-wise convolution 
        
#         padded_weight_1x1 = F.pad(self.conv1x1.weight, (2, 2, 2, 2)) 
#         padded_weight_3x3 = F.pad(self.conv3x3.weight, (1, 1, 1, 1)) 
        
#         identity_weight = F.pad(torch.ones_like(self.conv1x1.weight), (2, 2, 2, 2)) 
        
#         combined_weight = self.alpha[0]*identity_weight + self.alpha[1]*padded_weight_1x1 + self.alpha[2]*padded_weight_3x3 + self.alpha[3]*self.conv5x5.weight 
        
#         device = self.conv5x5_reparam.weight.device 

#         combined_weight = combined_weight.to(device)

#         self.conv5x5_reparam.weight = nn.Parameter(combined_weight)


#     def forward(self, x): 
        
#         if self.training: 
#             self.repram_flag = True
#             out = self.forward_train(x) 
#         elif self.training == False and self.repram_flag == True:
#             self.reparam_5x5() 
#             self.repram_flag = False 
#             out = self.conv5x5_reparam(x)
#         elif self.training == False and self.repram_flag == False:
#             out = self.conv5x5_reparam(x)
        
#         return out 



class LocalShift(nn.Module):
    def __init__(self, dim):
        super(LocalShift, self).__init__()
        # Define the layers for training
        self.conv1x1 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, groups=dim, bias=False)
        self.conv3x3 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=1, groups=dim, bias=False)
        self.conv5x5 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=5, padding=2, groups=dim, bias=False) 
        self.conv7x7 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=7, padding=3, groups=dim, bias=False)
        self.beta = nn.Parameter(torch.randn(5), requires_grad=True) 
        self.alpha = nn.Parameter(torch.randn(4), requires_grad=True) 
        

        # Define the layers for testing
        self.conv7x7_reparam = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=7, padding=3, groups=dim, bias = False) 
        self.conv5x5_reparam = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=5, padding=2, groups=dim, bias = False) 
        self.repram_flag = True

    def forward_train(self, x):
        out1x1 = self.conv1x1(x)
        out3x3 = self.conv3x3(x)
        out5x5 = self.conv5x5(x) 
        out7x7 = self.conv7x7(x) 
        # import pdb 
        # pdb.set_trace() 
        
        
        out = self.beta[0]*x + self.beta[1]*out1x1 + self.beta[2]*out3x3 + self.beta[3]*out5x5 + self.beta[4]*out7x7
        return out

    def reparam_7x7(self):
        # Combine the parameters of conv1x1, conv3x3, and conv5x5 to form a single 5x5 depth-wise convolution 
        
        padded_weight_1x1 = F.pad(self.conv1x1.weight, (3, 3, 3, 3)) 
        padded_weight_3x3 = F.pad(self.conv3x3.weight, (2, 2, 2, 2)) 
        padded_weight_5x5 = F.pad(self.conv5x5.weight, (1, 1, 1, 1)) 
        
        identity_weight = F.pad(torch.ones_like(self.conv1x1.weight), (3, 3, 3, 3)) 
        
        combined_weight = self.beta[0]*identity_weight + self.beta[1]*padded_weight_1x1 + self.beta[2]*padded_weight_3x3 + self.beta[3]*padded_weight_5x5 + self.beta[4]*self.conv7x7.weight 
        
        device = self.conv7x7_reparam.weight.device 

        combined_weight = combined_weight.to(device)

        self.conv7x7_reparam.weight = nn.Parameter(combined_weight)


    def forward(self, x): 
        
        if self.training: 
            self.repram_flag = True
            out = self.forward_train(x) 
        elif self.training == False and self.repram_flag == True:
            self.reparam_7x7() 
            self.repram_flag = False 
            out = self.conv7x7_reparam(x)
        elif self.training == False and self.repram_flag == False:
            out = self.conv7x7_reparam(x)
        
        return out 

dim = 12

model = LocalShift(dim=dim) 
x = torch.randn((1,dim,128,128)) 
y = model(x) 
model.eval()
z = model(x)

result = torch.mean(torch.abs(y-z)) 
print(result)