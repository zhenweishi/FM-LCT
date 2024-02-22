import torch
import torch.nn as nn
from timm.models.layers import Mlp

__all__ = ["ConcatClassifier", "SumClassifier"]

class SumClassifier(nn.Module):
    def __init__(self, x_dim, shape_dim, hidden_dim, num_classes, drop=0., *args, **kwargs):
        super().__init__(*args, **kwargs)
        # x_dim to hidden_dim
        # shape_dim to hidden_dim
        # sum and then to num_classes
        self.mlp_x = Mlp(x_dim, hidden_dim, hidden_dim, drop=drop, act_layer=nn.GELU)
        self.mlp_shape = Mlp(shape_dim, hidden_dim, hidden_dim, drop=drop, act_layer=nn.GELU)
        self.mlp_sum = Mlp(hidden_dim, hidden_dim, num_classes, drop=drop, act_layer=nn.GELU)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, shape_x):
        x = self.mlp_x(x)
        shape_x = self.mlp_shape(shape_x)
        x = x + shape_x
        # x = self.bn(x)
        x = self.mlp_sum(x)
        return x

class ConcatClassifier(nn.Module):
    def __init__(self, x_dim, shape_dim, hidden_dim, num_classes, drop=0., *args, **kwargs):
        super().__init__(*args, **kwargs)
        input_dim = x_dim + shape_dim
        self.mlp = Mlp(input_dim, hidden_dim, num_classes, drop=drop, act_layer=nn.ReLU)
        # self.tmp = Mlp(shape_dim, 10, num_classes, drop=drop, act_layer=nn.ReLU)
        self.apply(self._init_weights)

    def forward(self, x, shape_x):
        x = torch.cat((x, shape_x), dim=1)
        x = self.mlp(x)
        return x

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
if __name__ == '__main__':
    x_dim = 768
    shape_dim = 20
    hidden_dim = 512
    num_classes = 2
    drop = 0.1
    model = ConcatClassifier(x_dim, shape_dim, hidden_dim, num_classes, drop)

    batch_size = 32
    x = torch.randn(batch_size, 768)
    shape_x = torch.randn(batch_size, 20)

    output = model(x, shape_x)
    print(output.shape) # torch.Size([32, 2])