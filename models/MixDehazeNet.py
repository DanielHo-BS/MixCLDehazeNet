import torch
import torch.nn as nn
import torch.nn.functional as F


class MixStructureBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.norm1 = nn.BatchNorm2d(dim)
        self.norm2 = nn.BatchNorm2d(dim)

        self.conv1 = nn.Conv2d(dim, dim, kernel_size=1)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=5, padding=2, padding_mode='reflect')
        self.conv3_19 = nn.Conv2d(dim, dim, kernel_size=7, padding=9, groups=dim, dilation=3, padding_mode='reflect')
        self.conv3_13 = nn.Conv2d(dim, dim, kernel_size=5, padding=6, groups=dim, dilation=3, padding_mode='reflect')
        self.conv3_7 = nn.Conv2d(dim, dim, kernel_size=3, padding=3, groups=dim, dilation=3, padding_mode='reflect')

        # Simple Channel Attention
        self.Wv = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=3 // 2, groups=dim, padding_mode='reflect')
        )
        self.Wg = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim, 1),
            nn.Sigmoid()
        )

        # Channel Attention
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim, 1, padding=0, bias=True),
            nn.GELU(),
            # nn.ReLU(True),
            nn.Conv2d(dim, dim, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

        # Pixel Attention
        self.pa = nn.Sequential(
            nn.Conv2d(dim, dim // 8, 1, padding=0, bias=True),
            nn.GELU(),
            # nn.ReLU(True),
            nn.Conv2d(dim // 8, 1, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

        self.mlp = nn.Sequential(
            nn.Conv2d(dim * 3, dim * 4, 1),
            nn.GELU(),
            # nn.ReLU(True),
            nn.Conv2d(dim * 4, dim, 1)
        )
        self.mlp2 = nn.Sequential(
            nn.Conv2d(dim * 3, dim * 4, 1),
            nn.GELU(),
            # nn.ReLU(True),
            nn.Conv2d(dim * 4, dim, 1)
        )

    def forward(self, x):
        identity = x
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.cat([self.conv3_19(x), self.conv3_13(x), self.conv3_7(x)], dim=1)
        x = self.mlp(x)
        x = identity + x

        identity = x
        x = self.norm2(x)
        x = torch.cat([self.Wv(x) * self.Wg(x), self.ca(x) * x, self.pa(x) * x], dim=1)
        x = self.mlp2(x)
        x = identity + x
        return x


class BasicLayer(nn.Module):
    def __init__(self, dim, depth):
        super().__init__()
        self.dim = dim
        self.depth = depth

        # build blocks
        self.blocks = nn.ModuleList(
            [MixStructureBlock(dim=dim) for i in range(depth)])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


class PatchEmbed(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, kernel_size=None):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if kernel_size is None:
            kernel_size = patch_size

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=patch_size,
                              padding=(kernel_size - patch_size + 1) // 2, padding_mode='reflect')

    def forward(self, x):
        x = self.proj(x)
        return x


class PatchUnEmbed(nn.Module):
    def __init__(self, patch_size=4, out_chans=3, embed_dim=96, kernel_size=None):
        super().__init__()
        self.out_chans = out_chans
        self.embed_dim = embed_dim

        if kernel_size is None:
            kernel_size = 1

        self.proj = nn.Sequential(
            nn.Conv2d(embed_dim, out_chans * patch_size ** 2, kernel_size=kernel_size,
                      padding=kernel_size // 2, padding_mode='reflect'),
            nn.PixelShuffle(patch_size)
        )

    def forward(self, x):
        x = self.proj(x)
        return x


class SKFusion(nn.Module):
    def __init__(self, dim, height=2, reduction=8):
        super(SKFusion, self).__init__()

        self.height = height
        d = max(int(dim / reduction), 4)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, d, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(d, dim * height, 1, bias=False)
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, in_feats):
        B, C, H, W = in_feats[0].shape

        in_feats = torch.cat(in_feats, dim=1)
        in_feats = in_feats.view(B, self.height, C, H, W)

        feats_sum = torch.sum(in_feats, dim=1)
        attn = self.mlp(self.avg_pool(feats_sum))
        attn = self.softmax(attn.view(B, self.height, C, 1, 1))

        out = torch.sum(in_feats * attn, dim=1)
        return out


class TextEmb(nn.Module):
    def __init__(self, embed_text=512, embed_dim=24, H=256, W=256):
        super().__init__()
        self.weight = 0.1
        self.proj = nn.Linear(embed_text, embed_dim*H*W)

    def forward(self, x, x1):
        x1 = self.proj(x1)
        x1 = x1.view(x.shape[0], x.shape[1], x.shape[2], x.shape[3])
        x = x + x1 * self.weight
        return x


class CrossModalAttention(nn.Module):
    def __init__(self):
        super(CrossModalAttention, self).__init__()

    def forward(self, F_s, F_t):
        # F_s ∈ R^{HW×C}
        # F_t ∈ R^{K×C}
        B,C,H,W = F_s.shape
        # transpose F_s from C×H×W to H×W×C
        F_s_t = F_s.permute(0, 2, 3, 1)
        _, _, dim_t = F_t.shape
        # reshape F_s into a 1D vector sequence
        F_s_t = F_s_t.contiguous().view(B, -1, dim_t)

        # A denotes the cross-modal attention map
        # A ∈ R^{HW×K}
        A = torch.matmul(F_s_t, F_t.permute(0,2,1))
        # scale A
        A = A / (A.shape[-1] ** 0.5)

        #  bidirectionally update both textual and visual features
        #  F_s' = softmax(A)F_t^T
        #  F_t' = softmax(A^T)F_s

        A = F.softmax(A, dim=-1)

        # update F_t by adding the cross-modal attention
        F_t_updated = torch.matmul(A.permute(0, 2, 1), F_s_t)

        # update F_s_t by adding the cross-modal attention
        F_s_t_update = torch.matmul(A, F_t)
        # reshape F_s_t_update into a 3D tensor
        F_s_t_update = F_s_t_update.view(B, H, W, C)
        # transpose F_s_t_update from H×W×C to C×H×W
        F_s_t_update = F_s_t_update.permute(0, 3, 1, 2).contiguous()
        # F_s = F_s + F_s_t_update
        F_s_updated = F_s + F_s_t_update

        return F_s_updated , F_t_updated


class CrossModalAttention2(nn.Module):
    def __init__(self):
        super(CrossModalAttention2, self).__init__()

    def check_image_size(self, x):
        # check image size: HxW should be divisible by dim_t (default: 512=2^9)
        # if not, pad the image to make it divisible 2^(a+b)
        # a and b are determined by the ratio of H and W
        
        _, _, h, w = x.size()
        if h * w % 512 != 0:
            a = round(9 * h / (h+w))
            b = 9 - a
            mod_pad_h = (2**a - h % (2**a))
            mod_pad_w = (2**b - w % (2**b))
            x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x
    
    def forward(self, F_s, F_t):
        F_s_tmp = self.check_image_size(F_s)
        B,C,H,W = F_s.shape
        _, _, H_tmp, W_tmp = F_s_tmp.shape
        _, _, dim_t = F_t.shape  # F_t ∈ R^{K×C}
        # transpose F_s from C×H×W to H×W×C, and reshape F_s into a 1D vector sequence
        F_s_tmp = F_s_tmp.permute(0, 2, 3, 1).contiguous().view(B, -1, dim_t)  # F_s ∈ R^{HWC}

        # A denotes the cross-modal attention map
        A = torch.matmul(F_s_tmp, F_t.permute(0,2,1))  # A ∈ R^{HW×K}
        # scale A and softmax
        A = F.softmax(A / (A.shape[1] ** 0.5), dim=1)

        # bidirectionally update both textual and visual features
        # update F_t by adding the cross-modal attention, F_t' = softmax(A^T)F_s
        F_t_updated = torch.matmul(A.permute(0, 2, 1), F_s_tmp)
        # update F_s_tmp by adding the cross-modal attention, F_s' = softmax(A)F_t^T
        F_s_tmp_update = torch.matmul(A, F_t)
        # reshape F_s_tmp_update into a 3D tensor, and transpose F_s_tmp_update from H×W×C to C×H×W
        F_s_tmp_update = F_s_tmp_update.view(B, H_tmp, W_tmp, C).permute(0, 3, 1, 2).contiguous()
        # F_s = F_s + F_s_tmp_update
        F_s_updated = F_s + F_s_tmp_update[:,:,:H,:W]

        return F_s_updated , F_t_updated


class CrossModalAttention3(nn.Module):
    def __init__(self):
        super(CrossModalAttention3, self).__init__()

    def check_image_size(self, x):
        # check image size: HxW should be divisible by dim_t (default: 512=2^9)
        # if not, pad the image to make it divisible 2^(a+b)
        # a and b are determined by the ratio of H and W
        
        _, _, h, w = x.size()
        if h * w % 512 != 0:
            a = round(9 * h / (h+w))
            b = 9 - a
            mod_pad_h = (2**a - h % (2**a))
            mod_pad_w = (2**b - w % (2**b))
            x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x
    
    def forward(self, F_s, F_t):
        F_s_tmp = self.check_image_size(F_s)
        B,C,H,W = F_s.shape
        _, _, H_tmp, W_tmp = F_s_tmp.shape
        _, _, dim_t = F_t.shape  # F_t ∈ R^{K×C}
        # transpose F_s from C×H×W to H×W×C, and reshape F_s into a 1D vector sequence
        F_s_tmp = F_s_tmp.permute(0, 2, 3, 1).contiguous().view(B, -1, dim_t)  # F_s ∈ R^{HWC}

        # A denotes the cross-modal attention map
        A = torch.matmul(F_s_tmp, F_t.permute(0,2,1))  # A ∈ R^{HW×K}
        # scale A and softmax
        A = F.softmax(A / (A.shape[1] ** 0.5), dim=1)

        # bidirectionally update both textual and visual features
        # update F_t by adding the cross-modal attention, F_t' = softmax(A^T)F_s
        F_t_updated = torch.matmul(A.permute(0, 2, 1), F_s_tmp)
        # update F_s_tmp by adding the cross-modal attention, F_s' = softmax(A)F_t^T
        F_s_tmp_update = torch.matmul(A, F_t)
        # reshape F_s_tmp_update into a 3D tensor, and transpose F_s_tmp_update from H×W×C to C×H×W
        F_s_tmp_update = F_s_tmp_update.view(B, H_tmp, W_tmp, C).permute(0, 3, 1, 2).contiguous()
        F_s_updated = F_s + (1 - F_s_tmp_update[:,:,:H,:W])

        return F_s_updated , F_t_updated


class MixDehazeNet(nn.Module):
    def __init__(self, in_chans=3, out_chans=4,
                 embed_dims=[24, 48, 96, 48, 24],
                 depths=[1, 1, 2, 1, 1]):
        super(MixDehazeNet, self).__init__()

        # setting
        self.patch_size = 4

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=1, in_chans=in_chans, embed_dim=embed_dims[0], kernel_size=3)
        
        # text embedding
        # self.text_emb1 = TextEmb(embed_text=512, embed_dim=embed_dims[0], H=256, W=256)
        # self.text_emb = CrossModalAttention()
        self.text_emb = CrossModalAttention2()

        # backbone
        self.layer1 = BasicLayer(dim=embed_dims[0], depth=depths[0])

        self.patch_merge1 = PatchEmbed(
            patch_size=2, in_chans=embed_dims[0], embed_dim=embed_dims[1], kernel_size=3)

        self.skip1 = nn.Conv2d(embed_dims[0], embed_dims[0], 1)

        self.layer2 = BasicLayer(dim=embed_dims[1], depth=depths[1])

        self.patch_merge2 = PatchEmbed(
            patch_size=2, in_chans=embed_dims[1], embed_dim=embed_dims[2], kernel_size=3)

        self.skip2 = nn.Conv2d(embed_dims[1], embed_dims[1], 1)

        self.layer3 = BasicLayer(dim=embed_dims[2], depth=depths[2])

        self.patch_split1 = PatchUnEmbed(
            patch_size=2, out_chans=embed_dims[3], embed_dim=embed_dims[2])

        assert embed_dims[1] == embed_dims[3]
        self.fusion1 = SKFusion(embed_dims[3])

        self.layer4 = BasicLayer(dim=embed_dims[3], depth=depths[3])

        self.patch_split2 = PatchUnEmbed(
            patch_size=2, out_chans=embed_dims[4], embed_dim=embed_dims[3])

        assert embed_dims[0] == embed_dims[4]
        self.fusion2 = SKFusion(embed_dims[4])

        self.layer5 = BasicLayer(dim=embed_dims[4], depth=depths[4])

        # merge non-overlapping patches into image
        self.patch_unembed = PatchUnEmbed(
            patch_size=1, out_chans=out_chans, embed_dim=embed_dims[4], kernel_size=3)

    def check_image_size(self, x):
        # NOTE: for I2I test
        _, _, h, w = x.size()
        mod_pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        mod_pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward_features(self, x, x1): # x(B, 3, 256, 256)
        x, x1 = self.text_emb(x, x1)
        x = self.patch_embed(x) # x(B, 24, 256, 256)
        x = self.layer1(x)
        skip1 = x

        x = self.patch_merge1(x) # x(B, 48, 128, 128)
        x = self.layer2(x)
        skip2 = x

        x = self.patch_merge2(x) # x(B, 96, 64, 64)
        x = self.layer3(x)
        x = self.patch_split1(x) # x(B, 48, 128, 128)

        x = self.fusion1([x, self.skip2(skip2)]) + x
        x = self.layer4(x)
        x = self.patch_split2(x) # x(B, 24, 256, 256)

        x = self.fusion2([x, self.skip1(skip1)]) + x 
        x = self.layer5(x)
        x = self.patch_unembed(x) # x(B, 4, 256, 256)
        return x # x(B, 4, 256, 256)

    def forward(self, x, x1):
        H, W = x.shape[2:]
        x = self.check_image_size(x)

        feat = self.forward_features(x, x1)
        # 2022/11/26
        K, B = torch.split(feat, (1, 3), dim=1)

        x = K * x - B + x
        x = x[:, :, :H, :W]
        return x


def MixDehazeNet_t():
    return MixDehazeNet(
        embed_dims=[24, 48, 96, 48, 24],
        depths=[1, 1, 2, 1, 1])

def MixDehazeNet_s():
    return MixDehazeNet(
        embed_dims=[24, 48, 96, 48, 24],
        depths=[2, 2, 4, 2, 2])

def MixDehazeNet_b():
    return MixDehazeNet(
        embed_dims=[24, 48, 96, 48, 24],
        depths=[4, 4, 8, 4, 4])

def MixDehazeNet_l():
    return MixDehazeNet(
        embed_dims=[24, 48, 96, 48, 24],
        depths=[8, 8, 16, 8, 8])


