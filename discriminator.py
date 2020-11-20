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
    
# in_channel, out_channel, kernel_size, stride, padding, activation
# discriminator layers setting   
cfg_d = [[3, 64, 4, 2, 1, 'leakyrelu'], [64, 128, 4, 2, 1, 'leakyrelu'], [128, 256, 4, 2, 1, 'leakyrelu'], [256, 512, 4, 2, 1, 'leakyrelu'], 
         [512, 1024, 4, 2, 1, 'leakyrelu'], [1024, 2048, 4, 2, 1, 'leakyrelu']]
         
class Discriminator(nn.Module):
    def __init__(self, num_label):
        super(Discriminator, self).__init__()
        self.num_label = num_label
        self.conv = nn.Sequential(conv2d(cfg_d[0]), conv2d(cfg_d[1]), conv2d(cfg_d[2]), 
                                  conv2d(cfg_d[3]), conv2d(cfg_d[4]), conv2d(cfg_d[5]))
        
        self.src = nn.Conv2d(2048, 1, 3, 1, 1)
        self.cls = nn.Conv2d(2048, int(self.num_label), 2, 1, 0)

    def forward(self, x):
        out = self.conv(x)
        src_out = self.src(out)
        cls_out = self.cls(out)
        cls_out = cls_out.view(cls_out.size(0), cls_out.size(1))
        return src_out, cls_out
