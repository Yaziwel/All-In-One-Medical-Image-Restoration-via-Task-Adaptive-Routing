import torch
from einops import rearrange
import pdb

# def raster2snake_or_snake2raster(x, resolution, transpose=False): 

#     h, w = resolution 
#     b, _, c = x.shape
#     x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)  

#     if transpose:
#         x = torch.transpose(x, 2, 3) 
#         h, w = x.shape[2], x.shape[3]

#     odd_rows = x[:, :, ::2, :]
#     even_rows = x[:, :, 1::2, :] 
#     even_rows_flip = torch.flip(even_rows, dims=[3]) 
    
#     # pdb.set_trace()

#     y = torch.stack([odd_rows, even_rows_flip], dim=2) 
#     y = y.transpose(2, 3).reshape(b, c, -1, w)
#     y = rearrange(y, 'b c h w -> b (h w) c') 
    
#     return y 


def image2snake(x, transpose=False): 


    b, c, h, w = x.shape
    if transpose:
        x = torch.transpose(x, 2, 3) 
        h, w = x.shape[2], x.shape[3]

    odd_rows = x[:, :, ::2, :]
    even_rows = x[:, :, 1::2, :] 
    even_rows_flip = torch.flip(even_rows, dims=[3]) 
    
    # pdb.set_trace()

    y = torch.stack([odd_rows, even_rows_flip], dim=2) 
    y = y.transpose(2, 3).reshape(b, c, -1, w)
    y = rearrange(y, 'b c h w -> b (h w) c') 
    
    return y


def snake2image(x, resolution, transpose=False): 
    
    h, w = resolution 
    b, _, c = x.shape 
    
    x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w) 
    


    odd_rows = x[:, :, ::2, :]
    even_rows = x[:, :, 1::2, :] 
    even_rows_flip = torch.flip(even_rows, dims=[3]) 
    y = torch.stack([odd_rows, even_rows_flip], dim=2) 
    y = y.transpose(2, 3).reshape(b, c, h, w)

    if transpose:
        y = torch.transpose(y, 2, 3) 

    return y


# 示例用法
x = torch.randn(1, 1, 2, 6)  # 示例输入张量的形状为[2, 3, 4, 5] 

b, c, h, w = x.shape 


# y = image2snake(x, transpose=False) # image2snake
# z = snake2image(y, resolution=(h, w), transpose=False) # snake2raster 

transpose=True
n = image2snake(x, transpose=transpose) # raster2snake 
m = snake2image(n, resolution=(w, h) if transpose else (h, w), transpose=transpose) # snake2raster  do not need transpose
# out = rearrange(m, 'b (h w) c -> b c h w', h=w, w=h)  
# out = torch.transpose(out, -1, -2)
