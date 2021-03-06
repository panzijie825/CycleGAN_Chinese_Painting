import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
   
    def __init__(self, in_features):
 
        super(ResidualBlock, self).__init__()
 
 
        conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features)  ]
 
        self.conv_block = nn.Sequential(*conv_block)
 
    def forward(self, x):
 
        return x + self.conv_block(x)


#Generator Resnet_9

class Generator(nn.Module):
    def __init__(self, input_nc,output_nc, n_residual_blocks=9):
        super(Generator, self).__init__()

        # Initial convolution block
        
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
        ]
        in_features = 64
        out_features = in_features

        # Downsampling
        for _ in range(2):
            out_features *=2
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features


        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(out_features)]

        # Upsampling
        
        
        for _ in range(2):
            out_features //= 2
            model += [
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_features, out_features, 3, stride=1, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Output layer
        model += [nn.ReflectionPad2d(3), 
                  nn.Conv2d(64, output_nc, 7), 
                  nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


##############################
#        Discriminator
##############################


class Discriminator(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()
        
        model = [nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
                 nn.LeakyReLU(0.2,inplace = True)]
        
        
        model+=[nn.Conv2d(64, 128, 4, stride=2, padding=1),
                nn.InstanceNorm2d(128),
                nn.LeakyReLU(0.2,inplace = True)]
        
        model+=[nn.Conv2d(128, 256, 4, stride=2, padding=1),
                nn.InstanceNorm2d(236),
                nn.LeakyReLU(0.2,inplace = True)]
        
        model+=[nn.Conv2d(256, 512, 4, padding=1),
                nn.InstanceNorm2d(512), 
                nn.LeakyReLU(0.2, inplace=True)]
        
        # Classification layer
        model+=[nn.Conv2d(512,1,4,padding = 1)]
        
        self.model = nn.Sequential(*model)
        
#        channels, height, width = input_shape
#
#        # Calculate output shape of image discriminator (PatchGAN)
#        self.output_shape = (1, height // 2 ** 4, width // 2 ** 4)
#
#        def discriminator_block(in_filters, out_filters, normalize=True):
#            """Returns downsampling layers of each discriminator block"""
#            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
#            if normalize:
#                layers.append(nn.InstanceNorm2d(out_filters))
#            layers.append(nn.LeakyReLU(0.2, inplace=True))
#            return layers
#
#        self.model = nn.Sequential(
#            *discriminator_block(channels, 64, normalize=False),
#            *discriminator_block(64, 128),
#            *discriminator_block(128, 256),
#            *discriminator_block(256, 512),
#            nn.ZeroPad2d((1, 0, 1, 0)),
#            nn.Conv2d(512, 1, 4, padding=1)
#        )

    def forward(self, x):
        x=self.model(x)
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)