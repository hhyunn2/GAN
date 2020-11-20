# make conv layer
def conv2d(params_list, instance_norm = True):
    channel_in, channel_out, kernel_size, stride, padding, activation = params_list
    layers = []
    if instance_norm:
        layers += [nn.Conv2d(channel_in, channel_out, kernel_size, stride, padding, bias=False),
                   nn.InstanceNorm2d(channel_out)]
        nn.init.xavier_uniform_(layers[0].weight)
    else:
        layers += [nn.Conv2d(channel_in, channel_out, kernel_size, stride, padding, bias=False)]
        nn.init.xavier_uniform_(layers[0].weight)
        
    if activation.lower() == 'relu':
        layers += [nn.ReLU(inplace=True)]
    if activation.lower() == 'leakyrelu':
        layers += [nn.LeakyReLU(0.1, inplace=True)]
    if activation.lower() == 'tanh':
        layers += [nn.Tanh()]
    if activation.lower() == 'sigmoid':
        layers += [nn.Sigmoid()]
    
        
    return nn.Sequential(*layers)

# make deconv layer
def upconv2d(params_list, instance_norm = True):
    channel_in, channel_out, kernel_size, stride, padding, activation = params_list
    layers = []
    if instance_norm:
        layers += [nn.ConvTranspose2d(channel_in, channel_out, kernel_size, stride, padding, bias=False)]
        nn.init.xavier_uniform_(layers[0].weight)
    else:
        layers += [nn.ConvTranspose2d(channel_in, channel_out, kernel_size, stride, padding, bias=False)]
        nn.init.xavier_uniform_(layers[0].weight)
        
    if activation.lower() == 'relu':
        layers += [nn.ReLU(inplace=True)]
    if activation.lower() == 'leakyrelu':
        layers += [nn.LeakyReLU(0.1, inplace=True)]
    if activation.lower() == 'tanh':
        layers += [nn.Tanh()]
    if activation.lower() == 'sigmoid':
        layers += [nn.Sigmoid()]
        
    return nn.Sequential(*layers)

def transpose(ndarray):
    return np.transpose(ndarray, [0,2,3,1])

# in_channel, out_channel, kernel_size, stride, padding, activation setting
# Encoder and Decoder
cfg_g_enc = [[64, 128, 4, 2, 1, 'relu'], [128, 256, 4, 2, 1, 'relu']]
cfg_g_block = [[256, 256, 3, 1, 1, 'relu']]
cfg_g_dec = [[256, 128, 4, 2, 1, 'relu'], [128, 64, 4, 2, 1, 'relu'], [64, 3, 7, 1, 3, 'tanh']]

# make block
class ResnetBlock(nn.Module):
    def __init__(self):
        super(ResnetBlock, self).__init__()
        self.conv1 = conv2d(cfg_g_block[0])
        self.conv2 = nn.Conv2d(256, 256, 3, 1, 1)

    def forward(self, x):
        shortcut = x
        out = self.conv1(x)
        out = self.conv2(out)

        out += shortcut
        return out

# Generator layers
class Generator(nn.Module):
    def __init__(self, num_label):
        super(Generator, self).__init__()
        self.num_label = num_label
        self.conv1 = nn.Conv2d(3 + self.num_label, 64, 7, 1, 3)
        self.inst_norm = nn.InstanceNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.enc = nn.Sequential(conv2d(cfg_g_enc[0]), conv2d(cfg_g_enc[1]))
        self.block = nn.Sequential(ResnetBlock(),ResnetBlock(),ResnetBlock(),ResnetBlock(),ResnetBlock(),ResnetBlock(),ResnetBlock())
        self.dec = nn.Sequential(upconv2d(cfg_g_dec[0]), upconv2d(cfg_g_dec[1]), conv2d(cfg_g_dec[2], instance_norm=False))

    def forward(self, x, c):
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1,1,x.size(2), x.size(3))

        out = torch.cat([x, c], dim=1)
        out = self.conv1(out)
        out = self.inst_norm(out)
        out = self.relu(out)

        out = self.enc(out)
        out = self.block(out)
        out = self.dec(out)

        return out

