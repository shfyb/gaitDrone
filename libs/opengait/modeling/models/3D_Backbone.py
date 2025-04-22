import torch
import torch.nn as nn
from ..base_model import BaseModel
from ..modules import SetBlockWrapper, HorizontalPoolingPyramid, PackSequenceWrapper, SeparateFCs, SeparateBNNecks, \
    BasicConv2d, BasicConv3d
import torch.nn.functional as F
import pdb


def trans(x):
    n, c, s, h, w = x.size()
    x = x.transpose(1, 2).reshape(-1, c, h, w)
    return x


def trans_out(x, n, s):
    output_size = x.size()
    return x.reshape(n, s, *output_size[1:]).transpose(1, 2).contiguous()


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, input_size, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = input_size // num_heads

        self.q_linear = torch.nn.Linear(input_size, input_size)
        self.v_linear = torch.nn.Linear(input_size, input_size)
        self.k_linear = torch.nn.Linear(input_size, input_size)

        self.out_linear = torch.nn.Linear(input_size, input_size)

    def forward(self, x):
        # 注意力操作

        q = self.q_linear(x).view(x.size(0), -1, self.num_heads, self.head_dim)
        k = self.k_linear(x).view(x.size(0), -1, self.num_heads, self.head_dim)
        v = self.v_linear(x).view(x.size(0), -1, self.num_heads, self.head_dim)

        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        att_score = F.softmax(
            torch.matmul(q, k.permute(0, 1, 3, 2)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32)),
            dim=-1)
        out = torch.matmul(att_score, v).permute(0, 2, 1, 3).contiguous().view(x.size(0), -1,
                                                                               self.head_dim * self.num_heads)

        # 输出线性层
        out = self.out_linear(out)
        return out


class SelfAttentionLayer(torch.nn.Module):
    def __init__(self, in_channels):
        super(SelfAttentionLayer, self).__init__()
        self.in_channels = in_channels

        # Query, Key, Value 的线性映射
        self.query_conv = torch.nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = torch.nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        # Query, Key, Value 映射
        query = self.query_conv(x)
        key = self.key_conv(x)
        value = self.value_conv(x)

        # 将空间维度拉平
        n, c, h, w = x.size()
        query = query.view(n, -1, h * w).permute(0, 2, 1)
        key = key.view(n, -1, h * w)

        # 计算注意力分数
        energy = torch.bmm(query, key)
        attention = F.softmax(energy, dim=-1)

        # 通过注意力分数加权求和
        value = value.view(n, -1, h * w)
        out = torch.bmm(value, attention.permute(0, 2, 1)).view(n, c, h, w)

        return out


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

            elif typ == 'Mean_Variance':
                self.layer.append(Mean_Variance(cfg, self.in_channels, 3, 1, 1))
            # elif typ == 'BC3D':
            #     self.layer.append(SimpleConvLayer_3D(cfg, self.in_channels, 3, 1, 1))
            #     self.in_channels = int(cfg.split('-')[1])
        return self.layer


class Mean_Variance(nn.Module):

    def __init__(self, cfg, in_channel, kernel_size, stride, padding):
        super(Mean_Variance, self).__init__()
        self.out_channel = int(cfg.split('-')[1])
        self.conv3d_mean = BasicConv2d(in_channel, self.out_channel, kernel_size=kernel_size, stride=stride,
                                       padding=padding)
        self.conv3d_mean1 = BasicConv2d(self.out_channel, self.out_channel, kernel_size=kernel_size, stride=stride,
                                        padding=padding)

        self.leakyrelu = nn.LeakyReLU(inplace=True)
        self.relu = nn.ReLU(inplace=True)
        self.bn_mean = nn.BatchNorm2d(self.out_channel)
        self.bn_mean1 = nn.BatchNorm2d(self.out_channel)
        self.conv3d_variance = BasicConv2d(in_channel, self.out_channel, kernel_size=kernel_size, stride=stride,
                                           padding=padding)
        self.bn_variance = nn.BatchNorm2d(self.out_channel)
        self.conv3d_variance1 = BasicConv2d(self.out_channel, self.out_channel, kernel_size=kernel_size, stride=stride,
                                            padding=padding)
        self.bn_variance1 = nn.BatchNorm2d(self.out_channel)

    def forward(self, seqs, seqL):
        n, c, s, h, w = seqs.shape
        seqs = trans(seqs)
        mean = self.conv3d_mean(seqs)

        mean = self.bn_mean(mean)
        mean = self.leakyrelu(mean)
        mean = self.conv3d_mean1(mean)

        mean = self.bn_mean1(mean)

        mean = self.relu(mean)
        mean = trans_out(mean,n,s)
        variance = self.conv3d_variance(seqs)
        variance = self.bn_variance(variance)
        variance = self.leakyrelu(variance)
        variance = self.conv3d_variance1(variance)
        variance = self.bn_variance1(variance)

        variance = self.leakyrelu(variance)
        variance = trans_out(variance,n,s)

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


class LayerNorm(nn.Module):
    def __init__(self, layer_channels):
        super(LayerNorm, self).__init__()
        self.layernorm = nn.LayerNorm(layer_channels)

    def forward(self, x):
        x = x.permute(0, 2, 1).contiguous()
        x = self.layernorm(x)
        x = x.permute(0, 2, 1).contiguous()
        return x


class DDD_backbone(BaseModel):

    def build_network(self, model_cfg):
        # self.Backbone = self.get_backbone(model_cfg['backbone_cfg'])
        # self.Backbone = SetBlockWrapper(self.Backbone)
        self.backbone = getfullLayer(backbone_cfg=model_cfg['backbone_cfg'])
        self.layers = self.backbone()
        self.layers = nn.Sequential(*self.layers)
        self.FCs = SeparateFCs(**model_cfg['SeparateFCs'])
        self.BNNecks = SeparateBNNecks(**model_cfg['SeparateBNNecks'])
        self.TP = PackSequenceWrapper(torch.max)
        self.HPP = HorizontalPoolingPyramid(bin_num=model_cfg['bin_num'])
        self.layernorm_mean = LayerNorm(512)
        self.layernorm_variance = LayerNorm(512)
        # self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs):

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
        for layer in self.layers[:-1]:
            sils = layer(sils, seqL)
        mean, variance = self.layers[-1](sils, seqL)

        # for layer in self.layers:
        #     sils = layer(sils, seqL)

        # for vis
        # feature_path1 = '/data2/liaoqi/liaoqi/OpenGait_feature_map/folder1/sils1.pt'
        # torch.save(sils_1, feature_path1)
        # feature_path2 = '/data2/liaoqi/liaoqi/OpenGait_feature_map/folder1/sils2.pt'
        # sils, _ = torch.max(sils, dim=1)
        # torch.save(sils, feature_path2)


        outs = self.TP(mean, seqL, options={"dim": 2})[0]  # [n, c, h, w]
        outs = self.TP(mean, seqL, options={"dim": 2})[0]  # [n, c, h, w]
        variance = self.TP(variance, seqL, options={"dim": 2})[0]
        feat = self.HPP(outs)  # [n, c, p]
        feat = self.layernorm_mean(feat)
        # feat = feat.permute(0, 2, 1).contiguous()
        
        # feat = feat.permute(0, 2, 1).contiguous()

        #
        variance = self.HPP(variance)
        variance = self.layernorm_variance(variance)
        n1, c1, p1 = feat.size()
        epsilon = torch.randn(n1, c1, p1).cuda()
        # 使用均值和方差构建高斯分布

        variance_loss = torch.norm(variance, p=2) ** 2

        #

        #
        # 生成高斯噪声，维度与均值和方差相同
        gaussian_noise = epsilon * variance + feat

        embed_1 = self.FCs(feat)  # [n, c, p]
        embed_1_noise = self.FCs(gaussian_noise)
        embed_2, logits = self.BNNecks(embed_1)  # [n, c, p]
        _, logits_noise = self.BNNecks(embed_1_noise)
        
        retval = {
            'training_feat': {
                'triplet': {'embeddings': embed_1, 'labels': labs},
                'triplet_1': {'embeddings': embed_1_noise, 'labels': labs},
                'softmax': {'logits': logits, 'labels': labs},
                'softmax_1': {'logits': logits_noise, 'labels': labs},
                'variance_loss': {'variance': variance_loss},
            },
            'visual_summary': {
                'image/sils': sils_1.view(n * s, 1, h, w)
            },
            'inference_feat': {
                'embeddings': embed_1
            }
        }
        return retval
