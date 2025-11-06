import torch
import torchvision
import timm
import math

import torch
from torch import nn
import torch.distributions as dist

# from resnet import resnet18
from torchvision.models import resnet18
from torchvision import transforms
from torchvision.ops import roi_align
import math

import timm


class PerturbedTopK(nn.Module):

    def __init__(self, k: int, num_samples: int = 500, sigma: float = 0.05):
        super(PerturbedTopK, self).__init__()
    
        self.num_samples = num_samples
        self.sigma = sigma
        self.k = k
    
    def __call__(self, x):

        return PerturbedTopKFunction.apply(x, self.k, self.num_samples, self.sigma)
    



class PerturbedTopKFunction(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, x, k: int, num_samples: int = 500, sigma: float = 0.05):
        b, d = x.shape
        
        noise = torch.normal(mean=0.0, std=1.0, size=(b, num_samples, d)).to(x.device)
        perturbed_x = x[:, None, :] + noise * sigma
        
        topk_results = torch.topk(perturbed_x, k=k, dim=-1, sorted=False)
        indices = topk_results.indices
        indices = torch.sort(indices, dim=-1).values

        perturbed_output = torch.nn.functional.one_hot(indices, num_classes=d).float()
        indicators = perturbed_output.mean(dim=1)

        ctx.k = k
        ctx.num_samples = num_samples
        ctx.sigma = sigma
        ctx.perturbed_output = perturbed_output
        ctx.noise = noise

        return indicators


    @staticmethod
    def backward(ctx, grad_output):
        if grad_output is None:
            return tuple([None] * 5)

        noise_gradient = ctx.noise
        expected_gradient = (
            torch.einsum("bnkd,bne->bkde", ctx.perturbed_output, noise_gradient)
            / ctx.num_samples
            / ctx.sigma
        ) * float(ctx.k)
        
        grad_input = torch.einsum("bkd,bkde->be", grad_output, expected_gradient)
        
        return (grad_input,) + tuple([None] * 5)


class Scorer(nn.Module):

    def __init__(self, n_channel):
        super().__init__()    
        
        resnet = resnet18(weights="IMAGENET1K_V1")
        
        # Define the scorer as a sequence of layers from the ResNet model,
        # followed by a Conv2d layer with 1 output channel and a kernel size of 3,
        # and a MaxPool2d layer with a kernel size of (2,2) and a stride of (2,2)
        self.scorer = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            nn.Conv2d(128, 1, kernel_size=3, padding=0),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        )
        del resnet
    
    def forward(self, x):
        x = self.scorer(x)
        return x.squeeze(1)

# class Scorer(nn.Module): ##ViT
#     ''' Scorer network '''

#     def __init__(self, n_channel=3):
#         super().__init__()
        
#         # Create a ViT model with pretrained weights
#         vit = timm.create_model('vit_base_patch16_224', pretrained=True)
        
#         # Modify the ViT model to accept 3-channel input and output a single channel
#         self.scorer = nn.Sequential(
#             vit.patch_embed,
#             vit.pos_drop,
#             vit.blocks,
#             vit.norm,
#             nn.Conv2d(768, 1, kernel_size=1),  # Adjust the number of input channels if necessary
#             nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
#         )
#         # Delete the ViT model as it is not needed anymore
#         del vit
    
#     def forward(self, x):
#         # Pass the input through the scorer layers
#         x = self.scorer(x)
#         return x.squeeze(1)

def convert_to_2d_idx(idx, h, w):
    return idx // w, idx % w

class DPS(nn.Module):

    def __init__(self, n_channel, high_size, low_size, score_size, k, num_samples, sigma, patch_size, model_name, device, mc_samples=0):
        super().__init__()
        
        self.patch_size = patch_size
        self.k = k
        self.device = device

        self.scorer = Scorer(n_channel).to(device)

        self.TOPK = PerturbedTopK(k, num_samples, sigma)
        self.mc_samples = mc_samples

        h, w = high_size
        self.h_score, self.w_score = score_size

        self.scale_h = h // self.h_score
        self.scale_w = w // self.w_score
        padded_h = self.scale_h * self.h_score + patch_size - 1
        padded_w = self.scale_w * self.w_score + patch_size - 1
        top_pad = (patch_size - self.scale_h) // 2
        left_pad = (patch_size - self.scale_w) // 2
        bottom_pad = padded_h - top_pad - h
        right_pad = padded_w - left_pad - w
    
        self.padding = (left_pad, right_pad, top_pad, bottom_pad)
        self.downscale_transform = transforms.Resize(low_size)
    
    def forward(self, x_high):
        
        b, c = x_high.shape[:2]
        patch_size = self.patch_size
        device = self.device
        x_low = self.downscale_transform(x_high).to(device)

        ### Score patches to get indicators
        if torch.isnan(x_low).any() or torch.isinf(x_low).any():
            print("NaN/Inf in x_low before scorer")
        scores_2d = self.scorer(x_low)
        if torch.isnan(scores_2d).any() or torch.isinf(scores_2d).any():
            print("NaN/Inf in scorer output")
        scores_1d = scores_2d.view(b, -1)

        scores_min = scores_1d.min(axis=-1, keepdims=True)[0]
        scores_max = scores_1d.max(axis=-1, keepdims=True)[0]
        scores_1d =  (scores_1d - scores_min) / (scores_max - scores_min + 1e-5)

        indicators = self.TOPK(scores_1d).view(b, self.k, self.h_score, self.w_score)
        if torch.isnan(indicators).any() or torch.isinf(indicators).any():
            print("NaN/Inf in indicators after TOPK")
        
        self.indicators = indicators
    
        x_high_pad = torch.nn.functional.pad(x_high, self.padding, "constant", 0)
        patches = torch.zeros((b, self.k, c, patch_size, patch_size))

        patches = patches.to(device)
        x_high_pad = x_high_pad.to(device)
        for i in range(self.h_score):
            for j in range(self.w_score):
                start_h = i*self.scale_h
                start_w = j*self.scale_w

                current_patches = x_high_pad[:, :, start_h : start_h + patch_size , start_w : start_w + patch_size]
                weight = indicators[:, :, i, j]
                patches += torch.einsum('bchw,bk->bkchw', current_patches, weight)

        return patches.view(-1, c, self.patch_size, self.patch_size)

class Position_proposal(torch.nn.Module):
    def __init__(self, device, num_proposals=5, low_size=(660, 910), patch_size=(448, 448), num_channels=3):
        super(Position_proposal, self).__init__()
        self.device = device
        self.num_proposals = num_proposals
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.proposal_net = torch.nn.Linear()
        self.downscale_transform = transforms.Resize(low_size)

    def make_boxes(self, row_array, col_array, h_high, w_high):
        batch_size = h_high.shape[0]
        row_array_pixel = ((h_high - self.patch_size[0]).repeat((1, self.num_proposals)) * row_array).reshape(batch_size * self.num_proposals, 1)
        col_array_pixel = ((w_high - self.patch_size[1]).repeat((1, self.num_proposals)) * col_array).reshape(batch_size * self.num_proposals, 1)
        boxes = torch.tensor((batch_size * self.num_proposals, 5))

        boxes[:, 1] = row_array_pixel
        boxes[:, 2] = col_array_pixel
        boxes[:, 3] = row_array_pixel + self.patch_size[0]
        boxes[:, 4] = col_array_pixel + self.patch_size[1]
        
        for i in range(batch_size):
            boxes[i] = i
        return boxes

    def make_batch(self, x_high, boxes):
        batch_size = x_high.shape[0]
        out = torch.tensor((batch_size * self.num_proposals, x_high.shape[1], self.patch_size[0] + 2, self.patch_size[1] + 2))
        boxes_rounded = boxes
        boxes_rounded[:, 1:3] = torch.floor(boxes_rounded[:, 1:3])
        boxes_rounded[:, 3:5] = torch.ceil(boxes_rounded[:, 3:5])

        for patch_id in range(batch_size * self.num_proposals):
            out[patch_id, :, :, :] = x_high[boxes_rounded[patch_id, 0], :, boxes_rounded[patch_id, 1]: boxes_rounded[patch_id, 3], boxes_rounded[patch_id, 2]: boxes_rounded[patch_id, 4]]
        
        boxes[:, 1] -= boxes_rounded[1]
        boxes[:, 2] -= boxes_rounded[3]
        boxes[:, 3] -= boxes_rounded[1]
        boxes[:, 4] -= boxes_rounded[3]
        return out, boxes

    def extract_patch(self, x_high, row_array, col_array):
        batch_size, c_high, h_high, w_high = x_high.shape
        h_high = torch.tensor([h_high for i in range(batch_size)]).reshape((batch_size, 1))
        w_high = torch.tensor([w_high for i in range(batch_size)]).reshape((batch_size, 1))

        x_patches = torch.tensor((batch_size * self.num_proposals, self.num_channels, self.patch_size[0], self.patch_size[1]), device=self.device)
        boxes = self.make_boxes(row_array, col_array, h_high, w_high)
        img_parts, boxes = self.make_batch(x_high, boxes)
        img_parts, boxes = img_parts.to(self.device), boxes.to(self.device)
        return roi_align(input=img_parts, boxes=boxes, output_size=self.patch_size)
    
    def forward(self, x_high):
        x_low = self.downscale_transform(x_high)
        return 1

def adjust_sigma(n_epoch_warmup, n_epoch, max_sigma, DPS, loader, step):
    max_steps = int(n_epoch * len(loader))
    warmup_steps = int(n_epoch_warmup * len(loader))
    
    if step < warmup_steps:
        sigma = max_sigma * step / warmup_steps
    else:
        step -= warmup_steps
        max_steps -= warmup_steps
        
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        end_sigma = 1e-5
        sigma = max_sigma * q + end_sigma * (1 - q)
    
    DPS.TOPK.sigma = sigma


class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        attn = self.dropout(torch.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)
        return output

class MultiHeadCrossAttention(nn.Module):
    def __init__(self, n_token, n_head, d_model, d_k, d_v, attn_dropout=0.1, dropout=0.1):
        super().__init__()
        self.n_token = n_token
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.q = nn.Parameter(torch.empty((1, n_token, d_model)))
        q_init_val = math.sqrt(1 / d_k)
        nn.init.uniform_(self.q, a=-q_init_val, b=q_init_val)
        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)
        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5, attn_dropout=attn_dropout)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, x):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        n_token = self.n_token
        b, len_seq = x.shape[:2]
        q = self.w_qs(self.q).view(1, n_token, n_head, d_k)
        k = self.w_ks(x).view(b, len_seq, n_head, d_k)
        v = self.w_vs(x).view(b, len_seq, n_head, d_v)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        x = self.attention(q, k, v)
        x = x.transpose(1, 2).contiguous().view(b, n_token, -1)
        x = self.dropout(self.fc(x))
        x += self.q
        x = self.layer_norm(x)
        return x

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.w_2(torch.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual
        x = self.layer_norm(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, n_token, d_model, d_inner, n_head, d_k, d_v, attn_dropout=0.1, dropout=0.1):
        super().__init__()
        self.n_token = n_token
        self.crs_attn = MultiHeadCrossAttention(n_token, n_head, d_model, d_k, d_v, attn_dropout=attn_dropout, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, x):
        x = self.crs_attn(x)
        x = self.pos_ffn(x)
        return x

class Transformer(nn.Module):
    def __init__(self, n_layer, n_token, n_head, d_k, d_v, d_model, d_inner, attn_dropout=0.1, dropout=0.1):
        super().__init__()
        self.layer_stack = nn.ModuleList([
            EncoderLayer(n_token[i], d_model, d_inner, n_head, d_k, d_v, attn_dropout=attn_dropout, dropout=dropout)
            for i in range(n_layer)])

    def forward(self, x):
        for enc_layer in self.layer_stack:
            x = enc_layer(x)
        return x

class TransformerClassifier(nn.Module): ## Implementation inspired by original Transformer paper: Vaswani, A., et al. "Attention Is All You Need." NeurIPS 2017. arXiv:1706.03762
    def __init__(self, input_dim, num_classes, n_layer=2, n_token=(1,1), n_head=16, d_k=64, d_v=64, d_model=1024, d_inner=2048, attn_dropout=0.0, dropout=0.1):
        super(TransformerClassifier, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, n_token[0], d_model))
        self.transformer = Transformer(n_layer, n_token, n_head, d_k, d_v, d_model, d_inner, attn_dropout, dropout)
        self.fc = nn.Linear(d_model, num_classes)
        self.input_dim = input_dim
        self.n_samples = n_token[0]

    def forward(self, x):
        x = x.view(-1, self.n_samples, self.input_dim)
        x = self.embedding(x)
        x = x + self.pos_encoder
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.fc(x)
        return x
    
class FlexibleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(FlexibleMLP, self).__init__()
        layers = []
        current_dim = input_dim
        self.input_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            current_dim = hidden_dim

        layers.append(nn.Linear(current_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.reshape(-1, self.input_dim)
        return self.model(x)