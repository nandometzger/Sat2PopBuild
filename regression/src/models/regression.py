import torchvision.models as models
import torch.nn as nn
import torch

from models.linresnet import linear_resnet50
# import copy
import segmentation_models_pytorch as smp
from torch.nn.functional import upsample_nearest, interpolate

from utils.utils import plot_2dmatrix

class JacobsUNet(nn.Module):
    '''
    PomeloUNet
    '''
    def __init__(self, input_channels, num_classes):
        super(JacobsUNet, self).__init__()

        ic = input_channels
        
        self.encoder = nn.Sequential(
            nn.Sequential(
                    nn.Conv2d(ic, 32, kernel_size=3, padding='same'),  nn.Softplus(),
                    nn.Conv2d(32, 32, kernel_size=3, padding='same'), nn.Softplus() ),
            nn.Sequential(
                    nn.MaxPool2d(2,2),
                    nn.Conv2d(32, 64, kernel_size=3, padding='same'),  nn.Softplus(),
                    nn.Conv2d(64, 64, kernel_size=3, padding='same'), nn.Softplus() ),
            nn.Sequential(
                    nn.MaxPool2d(2,2),
                    nn.Conv2d(64, 128, kernel_size=3, padding='same'),  nn.Softplus(),
                    nn.Conv2d(128, 128, kernel_size=3, padding='same'),  nn.Softplus() ),
            nn.Sequential(
                    nn.MaxPool2d(2,2),
                    nn.Conv2d(128, 256, kernel_size=3, padding='same'),  nn.Softplus(),
                    nn.Conv2d(256, 256, kernel_size=3, padding='same'),  nn.Softplus() ) 
        )

        self.encoder2 = nn.Sequential(
            nn.Sequential(
                    nn.Conv2d(ic, 32, kernel_size=3, padding='same'),  nn.Softplus(),
                    nn.Conv2d(32, 32, kernel_size=3, padding='same'), nn.Softplus() ),
            nn.Sequential(
                    nn.MaxPool2d(2,2),
                    nn.Conv2d(32, 64, kernel_size=3, padding='same'),  nn.Softplus(),
                    nn.Conv2d(64, 64, kernel_size=3, padding='same'), nn.Softplus() ),
            nn.Sequential(
                    nn.MaxPool2d(2,2),
                    nn.Conv2d(64, 128, kernel_size=3, padding='same'),  nn.Softplus(),
                    nn.Conv2d(128, 128, kernel_size=3, padding='same'),  nn.Softplus() ),
            nn.Sequential(
                    nn.MaxPool2d(2,2),
                    nn.Conv2d(128, 256, kernel_size=3, padding='same'),  nn.Softplus(),
                    nn.Conv2d(256, 256, kernel_size=3, padding='same'),  nn.Softplus() ) 
        )

        self.decoder = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(384, 128, kernel_size=3, padding='same'),  nn.Softplus(),
                nn.Conv2d(128, 128, kernel_size=3, padding='same'),  nn.Softplus()),
            nn.Sequential(
                nn.Conv2d(192, 64, kernel_size=3, padding='same'),  nn.Softplus(),
                nn.Conv2d(64, 64, kernel_size=3, padding='same'),  nn.Softplus()),
            nn.Sequential(
                nn.Conv2d(96, 32, kernel_size=3, padding='same'),  nn.Softplus(),
                nn.Conv2d(32, 32, kernel_size=3, padding='same'),  nn.Softplus()),
        )
        
        self.decoder2 = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(384, 128, kernel_size=3, padding='same'),  nn.Softplus(),
                nn.Conv2d(128, 128, kernel_size=3, padding='same'),  nn.Softplus()),
            nn.Sequential(
                nn.Conv2d(192, 64, kernel_size=3, padding='same'),  nn.Softplus(),
                nn.Conv2d(64, 64, kernel_size=3, padding='same'),  nn.Softplus()),
            nn.Sequential(
                nn.Conv2d(96, 32, kernel_size=3, padding='same'),  nn.Softplus(),
                nn.Conv2d(32, 32, kernel_size=3, padding='same'),  nn.Softplus()),
        )

        self.head = nn.Sequential(nn.Conv2d(32, 2, kernel_size=3, padding=1), nn.Softplus())

        self.unetmodel = nn.Sequential(
            smp.Unet( encoder_name="resnet18", encoder_weights="imagenet", decoder_channels=(64, 32, 16),
                encoder_depth=3, in_channels=input_channels,  classes=32 ),
            nn.Softplus()
        )

        

    def forward(self, inputs):

        deactivated = True

        if deactivated:
            p2d = (2, 2, 2, 2)
            x = nn.functional.pad(inputs, p2d)
            x = self.unetmodel(x)[:,:,2:-2,2:-2]
        else:
                
            s = [ inputs.shape[2]//4, inputs.shape[2]//2, inputs.shape[2] ]
            s2 = [ inputs.shape[2]//2, inputs.shape[2] ]

            x = inputs
            
            #Encoding
            fmaps = []
            for layer in self.encoder:
                x = layer(x)
                fmaps.append(x)

            # remove this fmap, since it is the same as "x"
            del fmaps[-1]

            # Decoding
            for i, layer in enumerate(self.decoder):
                decodermap = torch.concatenate([ interpolate(x,(s[i],s[i])), fmaps[-(i+1)] ],1)
                x = layer(decodermap)

        x = self.head(x)
        Popmap = x[:,0]
        # Popmap = torch.exp(x[:,0])
        Popcount = Popmap.sum((1,2))

        builtmap = x[:,0]
        builtcount = builtmap.sum((1,2))

        return Popcount, {"Popmap": Popmap, "builtmap": builtmap, "builtcount": builtcount}




class PomeloUNet(nn.Module):
    '''
    PomeloUNet
    '''
    def __init__(self, input_channels, num_classes):
        super(PomeloUNet, self).__init__()
        
        self.PomeloEncoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding='same'),  nn.SELU(),
            nn.Conv2d(32, 32, kernel_size=3, padding='same'), nn.SELU(),
            nn.Conv2d(32, 32, kernel_size=3, padding='same'), nn.SELU(),
            nn.Conv2d(32, 32, kernel_size=3, padding='same'), nn.SELU(),
            nn.Conv2d(32, 32, kernel_size=3, padding='same'), nn.SELU(),
        )

        self.PomeloDecoder = nn.Sequential(
            nn.Conv2d(32+1, 128, kernel_size=3, padding='same'), nn.SELU(),
            nn.Conv2d(128, 128, kernel_size=1, padding='same'), nn.SELU(),
            nn.Conv2d(128, 128, kernel_size=1, padding='same'), nn.SELU(),
            nn.Conv2d(128, 128, kernel_size=1, padding='same'), nn.SELU(), 
            nn.Conv2d(128, 1, kernel_size=3, padding='same'), nn.SELU(),
        )

        ## set model features
        self.unetmodel = smp.Unet( encoder_name="resnet18", encoder_weights="imagenet",
                                  encoder_depth=3, in_channels=input_channels,  classes=num_classes )
        
        self.gumbeltau = torch.nn.Parameter(torch.tensor([2/3]), requires_grad=True)
        

    def forward(self, inputs):

        #Encoding
        encoding = self.PomeloEncoder(inputs)

        #Decode Buildings
        unetout = self.unetmodel(encoding)
        built_hard = torch.nn.functional.gumbel_softmax(unetout[0], tau=self.gumbeltau, hard=True, eps=1e-10, dim=1)[:,0]
        count = unetout[1]
        
        sparse_buildings = built_hard*count 

        with torch.no_grad():
            no_grad_sparse_buildings = sparse_buildings*1.0
            no_grad_count = count*1.0

        # Decode for OccRate (Population)
        OccRate = self.PomeloDecoder(torch.concatenate([encoding,no_grad_sparse_buildings],1))

        # Get population map and total count
        Popmap = OccRate * sparse_buildings
        Popcount = Popmap.sum()
        builtcount = builtcount
        
        return Popcount, {"Popmap": Popmap, "built_map": sparse_buildings, "builtcount": builtcount}

class EOResNet(nn.Module):
    '''
    Earth Observation ResNet 50
    '''
    def __init__(self, input_channels, num_classes):
        super(EOResNet, self).__init__()

        ## set model features
        self.model = models.resnet50(pretrained=False) # pretrained=False just for debug reasons
        first_conv_layer = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        first_conv_layer = [first_conv_layer]
        

        first_conv_layer.extend(list(self.model.children())[1:-1])
        self.model = nn.Sequential(*first_conv_layer) 

        self.clf_layer = nn.Linear(in_features=2048, out_features=num_classes)
        self.clf_layer.apply(init_weights)        

    def forward(self, inputs):
        outputs = self.model(inputs)
        outputs = outputs.view(outputs.shape[0],-1)
        outputs = self.clf_layer(outputs)
        return outputs
    

class EO2ResNet_OSM(nn.Module):
    '''
    Also using OSM Data
    '''
    def __init__(self, input_channels, num_classes, scale_factor=1):
        super(EO2ResNet_OSM, self).__init__()
        self.scale_factor = scale_factor
        num_features = [1024, 2048, 4096]
        self.cnn = EOResNet(input_channels, num_classes).model
        self.linear = linear_resnet50(scale_factor=scale_factor)
        if self.scale_factor != 1:
            self.lin_scale = nn.Sequential(
                nn.BatchNorm1d(num_features[1]*scale_factor),
                nn.Linear(num_features[1]*scale_factor, num_features[1]),
            )

        ## final depends on cnn output concat with linear output
        self.final = nn.Sequential(
            nn.Linear(in_features=num_features[2], out_features=num_features[1]),
            nn.LeakyReLU(inplace=True),
            nn.Linear(in_features=num_features[1], out_features=num_features[0]),
            nn.LeakyReLU(inplace=True)
            )
        self.clf_layer = nn.Linear(in_features=num_features[0],out_features=num_classes)
        self.final.apply(init_weights)
        self.clf_layer.apply(init_weights)

    def forward(self, inputs, osm_in):
        outputs = self.cnn(inputs)
        outputs = outputs.view(outputs.shape[0],-1)
        osm_in = osm_in.view(osm_in.shape[0],-1)
        osm_out = self.linear(osm_in)
        if self.scale_factor != 1:
            osm_out = self.lin_scale(osm_out)
        outputs = torch.cat((outputs, osm_out), dim=1)
        outputs = self.final(outputs)
        outputs = self.clf_layer(outputs)
        return outputs


def init_weights(layer, method = 'xavier normal'):
    if isinstance(layer, (nn.Linear, nn.Conv2d)):
        if method == 'xavier normal':
            nn.init.xavier_normal_(layer.weight)
        elif method == 'kaiming normal':
            nn.init.kaiming_normal_(layer.weight)