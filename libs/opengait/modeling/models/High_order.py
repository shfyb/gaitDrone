import torch
import torch.nn as nn
from ..base_model import BaseModel
from ..modules import SetBlockWrapper, HorizontalPoolingPyramid, PackSequenceWrapper, SeparateFCs, SeparateBNNecks, \
    BasicConv2d, BasicConv3d
import torch.nn.functional as F
import pdb

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable

import torch.fft as afft


def trans(x):
    n, c, s, h, w = x.size()
    x = x.transpose(1, 2).reshape(-1, c, h, w)
    return x


def trans_out(x, n, s):
    output_size = x.size()
    return x.reshape(n, s, *output_size[1:]).transpose(1, 2).contiguous()


class larange(nn.Module):
    def __init__(self, inchannels, top_k):
        super(larange, self).__init__()
        self.inchannels = inchannels
        self.channel_combine1 = nn.Conv2d(self.inchannels, 1, 1)
        self.channel_combine2 = nn.Conv2d(self.inchannels, 1, 1)
        self.topk = top_k

    def forward(self, x):
        n, _, s, _, _ = x.size()
        x_0 = x[:, :, 0:s - 1, :, :]
        x_1 = x[:, :, 1:s, :, :]
        x_0 = trans(x_0)
        x_1 = trans(x_1)
        X_0 = self.channel_combine1(x_0).squeeze(1)
        X_1 = self.channel_combine2(x_1).squeeze(1)
        ns, h, w = X_0.size()
        X_0 = X_0.view(ns, h * w, 1)
        X_1 = X_1.view(ns, 1, h * w)
        attention = F.softmax(X_0 * X_1, dim=2)
        top_k_values = attention.topk(self.topk, dim=-1)[0]

        top_k_values = top_k_values.view(ns, self.topk, h, w)
        top_k_values = trans_out(top_k_values, n, s - 1)
        return top_k_values


class CompactBilinearPooling(nn.Module):
    """
    Compute compact bilinear pooling over two bottom inputs.

    Args:

        output_dim: output dimension for compact bilinear pooling.

        sum_pool: (Optional) If True, sum the output along height and width
                  dimensions and return output shape [batch_size, output_dim].
                  Otherwise return [batch_size, height, width, output_dim].
                  Default: True.

        rand_h_1: (Optional) an 1D numpy array containing indices in interval
                  `[0, output_dim)`. Automatically generated from `seed_h_1`
                  if is None.

        rand_s_1: (Optional) an 1D numpy array of 1 and -1, having the same shape
                  as `rand_h_1`. Automatically generated from `seed_s_1` if is
                  None.

        rand_h_2: (Optional) an 1D numpy array containing indices in interval
                  `[0, output_dim)`. Automatically generated from `seed_h_2`
                  if is None.

        rand_s_2: (Optional) an 1D numpy array of 1 and -1, having the same shape
                  as `rand_h_2`. Automatically generated from `seed_s_2` if is
                  None.
    """

    def __init__(self, input_dim1, input_dim2, output_dim,
                 sum_pool=True, cuda=True,
                 rand_h_1=None, rand_s_1=None, rand_h_2=None, rand_s_2=None):
        super(CompactBilinearPooling, self).__init__()
        self.input_dim1 = input_dim1
        self.input_dim2 = input_dim2
        self.output_dim = output_dim
        self.sum_pool = sum_pool

        if rand_h_1 is None:
            np.random.seed(1)
            rand_h_1 = np.random.randint(output_dim, size=self.input_dim1)
        if rand_s_1 is None:
            np.random.seed(3)
            rand_s_1 = 2 * np.random.randint(2, size=self.input_dim1) - 1

        self.sparse_sketch_matrix1 = Variable(self.generate_sketch_matrix(
            rand_h_1, rand_s_1, self.output_dim))

        if rand_h_2 is None:
            np.random.seed(5)
            rand_h_2 = np.random.randint(output_dim, size=self.input_dim2)
        if rand_s_2 is None:
            np.random.seed(7)
            rand_s_2 = 2 * np.random.randint(2, size=self.input_dim2) - 1

        self.sparse_sketch_matrix2 = Variable(self.generate_sketch_matrix(
            rand_h_2, rand_s_2, self.output_dim))

        if cuda:
            self.sparse_sketch_matrix1 = self.sparse_sketch_matrix1.cuda()
            self.sparse_sketch_matrix2 = self.sparse_sketch_matrix2.cuda()

    def forward(self, bottom1, bottom2):
        """
        bottom1: 1st input, 4D Tensor of shape [batch_size, input_dim1, height, width].
        bottom2: 2nd input, 4D Tensor of shape [batch_size, input_dim2, height, width].
        """
        assert bottom1.size(1) == self.input_dim1 and \
               bottom2.size(1) == self.input_dim2

        batch_size, _, height, width = bottom1.size()

        bottom1_flat = bottom1.permute(0, 2, 3, 1).contiguous().view(-1, self.input_dim1)
        bottom2_flat = bottom2.permute(0, 2, 3, 1).contiguous().view(-1, self.input_dim2)

        sketch_1 = bottom1_flat.mm(self.sparse_sketch_matrix1)
        sketch_2 = bottom2_flat.mm(self.sparse_sketch_matrix2)

        # sketch_1 = sketch_1.float()
        # sketch_2 = sketch_2.float()

        fft1 = torch.fft.fft(sketch_1)
        fft2 = torch.fft.fft(sketch_2)

        fft_product = fft1 * fft2

        cbp_flat = torch.fft.ifft(fft_product).real

        # 将输出转换回半精度
        # cbp_flat = cbp_flat.half()

        cbp = cbp_flat.view(batch_size, height, width, self.output_dim)

        if self.sum_pool:
            cbp = cbp.sum(dim=1).sum(dim=1)

        return cbp

    @staticmethod
    def generate_sketch_matrix(rand_h, rand_s, output_dim):
        """
        Return a sparse matrix used for tensor sketch operation in compact bilinear
        pooling
        Args:
            rand_h: an 1D numpy array containing indices in interval `[0, output_dim)`.
            rand_s: an 1D numpy array of 1 and -1, having the same shape as `rand_h`.
            output_dim: the output dimensions of compact bilinear pooling.
        Returns:
            a sparse matrix of shape [input_dim, output_dim] for tensor sketch.
        """

        # Generate a sparse matrix for tensor count sketch
        rand_h = rand_h.astype(np.int64)
        rand_s = rand_s.astype(np.float32)
        assert (rand_h.ndim == 1 and rand_s.ndim ==
                1 and len(rand_h) == len(rand_s))
        assert (np.all(rand_h >= 0) and np.all(rand_h < output_dim))

        input_dim = len(rand_h)
        indices = np.concatenate((np.arange(input_dim)[..., np.newaxis],
                                  rand_h[..., np.newaxis]), axis=1)
        indices = torch.from_numpy(indices)
        rand_s = torch.from_numpy(rand_s)
        sparse_sketch_matrix = torch.sparse.FloatTensor(
            indices.t(), rand_s, torch.Size([input_dim, output_dim]))
        return sparse_sketch_matrix.to_dense()


def binlinearpooling_func(x):
    x = torch.sign(x) * torch.sqrt(torch.abs(x))
    # pdb.set_trace()
    return x / torch.norm(x, 'fro', dim=1).unsqueeze(1)


def binlinear_pooling_module(feat1, feat2):
    n, c, p = feat1.size()
    result = feat1.unsqueeze(2) * feat2.unsqueeze(1)

    # result = result.permute(0, 3, 1, 2).contiguous()

    # downsampled_result = F.avg_pool2d(result, kernel_size=16, stride=16)
    result = result.view(n, c * c, p)

    result = binlinearpooling_func(result)
    return result


# 使用示例

class getfullLayer(nn.Module):
    def __init__(self, backbone_cfg):

        super(getfullLayer, self).__init__()
        self.layers_cfg = backbone_cfg['layers_cfg']
        self.in_channels = backbone_cfg['in_channels']
        self.layer = []

    def __call__(self, *args, **kwargs):

        self.layer.append(SimpleConvLayerBN(self.layers_cfg[0], self.in_channels, 3, 1, 1))
        self.in_channels = int(self.layers_cfg[0].split('-')[1])

        for cfg in self.layers_cfg[1:]:
            typ = cfg.split('-')[0]
            if cfg == 'M':
                self.layer.append(maxpool())

            elif typ == 'BC':
                self.layer.append(SimpleConvLayer(cfg, self.in_channels, 3, 1, 1))
                self.in_channels = int(cfg.split('-')[1])
            elif typ == 'BCN':
                self.layer.append(SimpleConvLayerBN(cfg, self.in_channels, 3, 1, 1))
                self.in_channels = int(cfg.split('-')[1])
            elif typ == 'BCNM':
                self.layer.append(SimpleConvLayerBN(cfg, self.in_channels, 3, 2, 1))
                self.in_channels = int(cfg.split('-')[1])
            elif typ == 'BCN3D':

                self.layer.append(SimpleConvLayerBN_3D(cfg, self.in_channels, 3, 1, 1))
                self.in_channels = int(cfg.split('-')[1])
            elif typ == 'BCN3D*2':
                self.layer.append(Separate_BCN3D(cfg, self.in_channels, 3, 1, 1))
            elif typ == 'Mean_Variance':
                self.layer.append(Mean_Variance(cfg, self.in_channels, 3, 1, 1))
            # elif typ == 'BC3D':
            #     self.layer.append(SimpleConvLayer_3D(cfg, self.in_channels, 3, 1, 1))
            #     self.in_channels = int(cfg.split('-')[1])
        return self.layer


class Separate_BCN3D(nn.Module):
    def __init__(self, cfg, in_channel, kernel_size, stride, padding):
        super(Separate_BCN3D, self).__init__()
        self.out_channel = int(cfg.split('-')[1])
        self.conv3d1_branch1 = BasicConv3d(in_channel, self.out_channel, kernel_size=kernel_size, stride=stride,
                                           padding=padding)
        self.conv3d2_branch1 = BasicConv3d(self.out_channel, self.out_channel, kernel_size=kernel_size, stride=stride,
                                           padding=padding)
        self.conv3d3_branch1 = BasicConv3d(self.out_channel, self.out_channel, kernel_size=kernel_size, stride=stride,
                                           padding=padding)
        self.activation = nn.LeakyReLU(inplace=True)
        self.bn1_branch1 = nn.BatchNorm3d(self.out_channel)
        self.bn2_branch1 = nn.BatchNorm3d(self.out_channel)
        self.bn3_branch1 = nn.BatchNorm3d(self.out_channel)
        self.conv3d1_branch2 = BasicConv3d(in_channel, self.out_channel, kernel_size=kernel_size, stride=stride,
                                           padding=padding)
        self.conv3d2_branch2 = BasicConv3d(self.out_channel, self.out_channel, kernel_size=kernel_size, stride=stride,
                                           padding=padding)
        self.conv3d3_branch2 = BasicConv3d(self.out_channel, self.out_channel, kernel_size=kernel_size, stride=stride,
                                           padding=padding)

        self.bn1_branch2 = nn.BatchNorm3d(self.out_channel)
        self.bn2_branch2 = nn.BatchNorm3d(self.out_channel)
        self.bn3_branch2 = nn.BatchNorm3d(self.out_channel)

    def forward(self, seqs, seqL):
        feat1 = self.conv3d1_branch1(seqs)
        feat1 = self.bn1_branch1(feat1)
        feat1 = self.activation(feat1)
        feat1 = self.conv3d2_branch1(feat1)
        feat1 = self.bn2_branch1(feat1)
        feat1 = self.activation(feat1)
        feat1 = self.conv3d3_branch1(feat1)
        feat1 = self.bn3_branch1(feat1)
        feat1 = self.activation(feat1)

        feat2 = self.conv3d1_branch2(seqs)
        feat2 = self.bn1_branch2(feat2)
        feat2 = self.activation(feat2)
        feat2 = self.conv3d2_branch2(feat2)
        feat2 = self.bn2_branch2(feat2)
        feat2 = self.activation(feat2)
        feat2 = self.conv3d3_branch2(feat2)
        feat2 = self.bn3_branch2(feat2)
        feat2 = self.activation(feat2)
        return feat1, feat2


class Mean_Variance(nn.Module):

    def __init__(self, cfg, in_channel, kernel_size, stride, padding):
        super(Mean_Variance, self).__init__()
        self.out_channel = int(cfg.split('-')[1])
        self.conv3d_mean = BasicConv3d(in_channel, self.out_channel, kernel_size=kernel_size, stride=stride,
                                       padding=padding)
        self.conv3d_mean1 = BasicConv3d(in_channel, self.out_channel, kernel_size=kernel_size, stride=stride,
                                        padding=padding)

        self.leakyrelu = nn.LeakyReLU(inplace=True)
        self.relu = nn.ReLU(inplace=True)
        self.bn_mean = nn.BatchNorm3d(self.out_channel)
        self.bn_mean1 = nn.BatchNorm3d(self.out_channel)
        self.conv3d_variance = BasicConv3d(in_channel, self.out_channel, kernel_size=kernel_size, stride=stride,
                                           padding=padding)
        self.bn_variance = nn.BatchNorm3d(self.out_channel)
        self.conv3d_variance1 = BasicConv3d(in_channel, self.out_channel, kernel_size=kernel_size, stride=stride,
                                            padding=padding)
        self.bn_variance1 = nn.BatchNorm3d(self.out_channel)

    def forward(self, seqs, seqL):
        mean = self.conv3d_mean(seqs)

        mean = self.bn_mean(mean)
        mean = self.leakyrelu(mean)
        mean = self.conv3d_mean1(mean)

        mean = self.bn_mean1(mean)

        mean = self.relu(mean)

        variance = self.conv3d_variance(seqs)
        variance = self.bn_variance(variance)
        variance = self.leakyrelu(variance)
        variance = self.conv3d_variance1(variance)
        variance = self.bn_variance1(variance)

        variance = self.leakyrelu(variance)

        # seqs = seqs.reshape(_, -1, h, w)
        return mean, variance


class SimpleConvLayer(nn.Module):
    def __init__(self, cfg, in_channel, kernel_size, stride, padding):
        super(SimpleConvLayer, self).__init__()

        self.out_channel = int(cfg.split('-')[1])
        self.conv = BasicConv2d(in_channel, self.out_channel, kernel_size=kernel_size, stride=stride, padding=padding)
        self.activation = nn.LeakyReLU(inplace=True)

    def forward(self, seqs, views):
        seqs = self.conv(seqs)
        seqs = self.activation(seqs)
        return seqs


class SimpleConvLayerBN_3D(nn.Module):
    def __init__(self, cfg, in_channel, kernel_size, stride, padding):
        super(SimpleConvLayerBN_3D, self).__init__()
        self.out_channel = int(cfg.split('-')[1])
        self.conv3d = BasicConv3d(in_channel, self.out_channel, kernel_size=kernel_size, stride=stride, padding=padding)
        self.activation = nn.LeakyReLU(inplace=True)
        self.bn = nn.BatchNorm3d(self.out_channel)
        # self.bn = nn.BatchNorm2d(self.out_channel)

    def forward(self, seqs, seqL):
        seqs = self.conv3d(seqs)

        seqs = self.bn(seqs)
        seqs = self.activation(seqs)
        # seqs = seqs.reshape(_, -1, h, w)

        return seqs


class SimpleConvLayerBN(nn.Module):
    def __init__(self, cfg, in_channel, kernel_size, stride, padding):
        super(SimpleConvLayerBN, self).__init__()

        self.out_channel = int(cfg.split('-')[1])
        self.conv = BasicConv2d(in_channel, self.out_channel, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(self.out_channel)
        self.activation = nn.LeakyReLU(inplace=True)

    def forward(self, seqs, views):
        n, c, s, h, w = seqs.shape
        seqs = trans(seqs)
        seqs = self.conv(seqs)
        seqs = self.bn(seqs)
        seqs = self.activation(seqs)
        seqs = trans_out(seqs, n, s)
        return seqs


def trilinear_attention(x):
    n, c, h, w = x.size()
    x = x.reshape(n, c, h * w)
    x_T = x.transpose(1, 2)
    attention = F.softmax(torch.matmul(x_T, x), dim=-1)
    output = torch.matmul(attention, x)
    output = output.reshape(n, c, h, w)
    return output


class maxpool(nn.Module):
    def __init__(self):
        super(maxpool, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, seqs, views):
        # pdb.set_trace()
        n, c, s, h, w = seqs.size()
        seqs = seqs.reshape(n, c * s, h, w)
        seqs = self.maxpool(seqs)
        seqs = seqs.reshape(n, c, s, int(h / 2), int(w / 2))
        return seqs


class MultiheadAttention(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads):
        super(MultiheadAttention, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.attention = nn.MultiheadAttention(embed_dim=in_channels, num_heads=num_heads)

    def forward(self, x):
        n, c, h, w = x.size()
        # 重塑: [n, c, h, w] -> [h*w, n, c]
        x = x.permute(2, 3, 0, 1).reshape(h * w, n, c)

        # 应用多头注意力
        attn_output, _ = self.attention(x, x, x)

        # 重塑回原始维度: [h*w, n, c] -> [n, c, h, w]
        attn_output = attn_output.reshape(h, w, n, c).permute(2, 3, 0, 1)

        return attn_output


class High_order_DDD(BaseModel):

    def build_network(self, model_cfg):
        # self.Backbone = self.get_backbone(model_cfg['backbone_cfg'])
        # self.Backbone = SetBlockWrapper(self.Backbone)
        self.backbone = getfullLayer(backbone_cfg=model_cfg['backbone_cfg'])
        self.layers = self.backbone()
        self.layers = nn.Sequential(*self.layers)
        self.FCs = SeparateFCs(**model_cfg['SeparateFCs'])
        self.BNNecks = SeparateBNNecks(**model_cfg['SeparateBNNecks'])
        self.TP = PackSequenceWrapper(torch.max)
        # self.bilinear = CompactBilinearPooling(256, 256, 256)
        self.HPP = HorizontalPoolingPyramid(bin_num=model_cfg['bin_num'])
        # self.attention = MultiheadAttention(256, 256, 1)
        # self.layernorm_mean = nn.LayerNorm(320)
        # self.layernorm_variance = nn.LayerNorm(256)
        # self.relu = nn.ReLU(inplace=True)
        # self.larange = larange(inchannels=128, top_k=model_cfg['top_k'])

    def forward(self, inputs):

        # global motion_feat, feat2, feat1
        ipts, labs, _, views, seqL = inputs

        sils = ipts[0]
        # for 3 channels
        # pdb.set_trace()
        if len(sils.size()) == 4:
            sils = sils.unsqueeze(1)
        else:
            sils = sils.permute(0, 4, 1, 2, 3).contiguous()
        del ipts

        # n, _, s, _, _ = sils.shape
        # sils_1 = trans(sils)  # [n, c, s, h, w]
        sils_1 = sils
        n, c, s, h, w = sils_1.size()

        for i, layer in enumerate(self.layers):
            sils = layer(sils, seqL)

        feat = self.TP(sils, seqL, options={"dim": 2})[0]  # [n, c, h, w]

        feat1 = trilinear_attention(feat)

        feat1 = self.HPP(feat1)

        embed_1 = self.FCs(feat1)  # [n, c, p]

        embed_2, logits = self.BNNecks(embed_1)  # [n, c, p]

        retval = {
            'training_feat': {
                'triplet': {'embeddings': embed_1, 'labels': labs},

                'softmax': {'logits': logits, 'labels': labs},

            },
            'visual_summary': {
                'image/sils': sils_1.view(n * s, 1, h, w)
            },
            'inference_feat': {
                'embeddings': embed_1
            }
        }
        return retval
