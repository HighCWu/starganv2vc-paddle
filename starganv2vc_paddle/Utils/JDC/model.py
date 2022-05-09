"""
Implementation of model from:
Kum et al. - "Joint Detection and Classification of Singing Voice Melody Using
Convolutional Recurrent Neural Networks" (2019)
Link: https://www.semanticscholar.org/paper/Joint-Detection-and-Classification-of-Singing-Voice-Kum-Nam/60a2ad4c7db43bace75805054603747fcd062c0d
"""
import paddle
from paddle import nn
        
class JDCNet(nn.Layer):
    """
    Joint Detection and Classification Network model for singing voice melody.
    """
    def __init__(self, num_class=722, seq_len=31, leaky_relu_slope=0.01):
        super().__init__()
        self.seq_len = seq_len  # 31
        self.num_class = num_class

        # input = (b, 1, 31, 513), b = batch size
        self.conv_block = nn.Sequential(
            nn.Conv2D(in_channels=1, out_channels=64, kernel_size=3, padding=1, bias_attr=False),  # out: (b, 64, 31, 513)
            nn.BatchNorm2D(num_features=64),
            nn.LeakyReLU(leaky_relu_slope),
            nn.Conv2D(64, 64, 3, padding=1, bias_attr=False),  # (b, 64, 31, 513)
        )

        # res blocks
        self.res_block1 = ResBlock(in_channels=64, out_channels=128)  # (b, 128, 31, 128)
        self.res_block2 = ResBlock(in_channels=128, out_channels=192)  # (b, 192, 31, 32)
        self.res_block3 = ResBlock(in_channels=192, out_channels=256)  # (b, 256, 31, 8)

        # pool block
        self.pool_block = nn.Sequential(
            nn.BatchNorm2D(num_features=256),
            nn.LeakyReLU(leaky_relu_slope),
            nn.MaxPool2D(kernel_size=(1, 4)),  # (b, 256, 31, 2)
            nn.Dropout(p=0.5),
        )

        # maxpool layers (for auxiliary network inputs)
        # in = (b, 128, 31, 513) from conv_block, out = (b, 128, 31, 2)
        self.maxpool1 = nn.MaxPool2D(kernel_size=(1, 40))
        # in = (b, 128, 31, 128) from res_block1, out = (b, 128, 31, 2)
        self.maxpool2 = nn.MaxPool2D(kernel_size=(1, 20))
        # in = (b, 128, 31, 32) from res_block2, out = (b, 128, 31, 2)
        self.maxpool3 = nn.MaxPool2D(kernel_size=(1, 10))

        # in = (b, 640, 31, 2), out = (b, 256, 31, 2)
        self.detector_conv = nn.Sequential(
            nn.Conv2D(640, 256, 1, bias_attr=False),
            nn.BatchNorm2D(256),
            nn.LeakyReLU(leaky_relu_slope),
            nn.Dropout(p=0.5),
        )

        # input: (b, 31, 512) - resized from (b, 256, 31, 2)
        self.bilstm_classifier = nn.LSTM(
            input_size=512, hidden_size=256,
            time_major=False, direction='bidirectional')  # (b, 31, 512)

        # input: (b, 31, 512) - resized from (b, 256, 31, 2)
        self.bilstm_detector = nn.LSTM(
            input_size=512, hidden_size=256,
            time_major=False, direction='bidirectional')  # (b, 31, 512)

        # input: (b * 31, 512)
        self.classifier = nn.Linear(in_features=512, out_features=self.num_class)  # (b * 31, num_class)

        # input: (b * 31, 512)
        self.detector = nn.Linear(in_features=512, out_features=2)  # (b * 31, 2) - binary classifier

        # initialize weights
        self.apply(self.init_weights)

    def get_feature_GAN(self, x):
        seq_len = x.shape[-2]
        x = x.astype(paddle.float32).transpose([0,1,3,2] if len(x.shape) == 4 else [0,2,1])
        
        convblock_out = self.conv_block(x)
        
        resblock1_out = self.res_block1(convblock_out)
        resblock2_out = self.res_block2(resblock1_out)
        resblock3_out = self.res_block3(resblock2_out)
        poolblock_out = self.pool_block[0](resblock3_out)
        poolblock_out = self.pool_block[1](poolblock_out)
        
        return poolblock_out.transpose([0,1,3,2] if len(poolblock_out.shape) == 4 else [0,2,1])
        
    def forward(self, x):
        """
        Returns:
            classification_prediction, detection_prediction
            sizes: (b, 31, 722), (b, 31, 2)
        """
        ###############################
        # forward pass for classifier #
        ###############################
        x = x.astype(paddle.float32).transpose([0,1,3,2] if len(x.shape) == 4 else [0,2,1])
        
        convblock_out = self.conv_block(x)
        
        resblock1_out = self.res_block1(convblock_out)
        resblock2_out = self.res_block2(resblock1_out)
        resblock3_out = self.res_block3(resblock2_out)
        
        
        poolblock_out = self.pool_block[0](resblock3_out)
        poolblock_out = self.pool_block[1](poolblock_out)
        GAN_feature = poolblock_out.transpose([0,1,3,2] if len(poolblock_out.shape) == 4 else [0,2,1])
        poolblock_out = self.pool_block[2](poolblock_out)
        
        # (b, 256, 31, 2) => (b, 31, 256, 2) => (b, 31, 512)
        classifier_out = poolblock_out.transpose([0, 2, 1, 3]).reshape((-1, self.seq_len, 512))
        self.bilstm_classifier.flatten_parameters()
        classifier_out, _ = self.bilstm_classifier(classifier_out)  # ignore the hidden states

        classifier_out = classifier_out.reshape((-1, 512))  # (b * 31, 512)
        classifier_out = self.classifier(classifier_out)
        classifier_out = classifier_out.reshape((-1, self.seq_len, self.num_class))  # (b, 31, num_class)
        
        # sizes: (b, 31, 722), (b, 31, 2)
        # classifier output consists of predicted pitch classes per frame
        # detector output consists of: (isvoice, notvoice) estimates per frame
        return paddle.abs(classifier_out.squeeze()), GAN_feature, poolblock_out

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.initializer.KaimingUniform()(m.weight)
            if m.bias is not None:
                nn.initializer.Constant(0)(m.bias)
        elif isinstance(m, nn.Conv2D):
            nn.initializer.XavierNormal()(m.weight)
        elif isinstance(m, nn.LSTM) or isinstance(m, nn.LSTMCell):
            for p in m.parameters():
                if len(p.shape) >= 2 and float('.'.join(paddle.__version__.split('.')[:2])) >= 2.3:
                    nn.initializer.Orthogonal()(p)
                else:
                    nn.initializer.Normal()(p)
                    

class ResBlock(nn.Layer):
    def __init__(self, in_channels: int, out_channels: int, leaky_relu_slope=0.01):
        super().__init__()
        self.downsample = in_channels != out_channels

        # BN / LReLU / MaxPool layer before the conv layer - see Figure 1b in the paper
        self.pre_conv = nn.Sequential(
            nn.BatchNorm2D(num_features=in_channels),
            nn.LeakyReLU(leaky_relu_slope),
            nn.MaxPool2D(kernel_size=(1, 2)),  # apply downsampling on the y axis only
        )

        # conv layers
        self.conv = nn.Sequential(
            nn.Conv2D(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=3, padding=1, bias_attr=False),
            nn.BatchNorm2D(out_channels),
            nn.LeakyReLU(leaky_relu_slope),
            nn.Conv2D(out_channels, out_channels, 3, padding=1, bias_attr=False),
        )

        # 1 x 1 convolution layer to match the feature dimensions
        self.conv1by1 = None
        if self.downsample:
            self.conv1by1 = nn.Conv2D(in_channels, out_channels, 1, bias_attr=False)

    def forward(self, x):
        x = self.pre_conv(x)
        if self.downsample:
            x = self.conv(x) + self.conv1by1(x)
        else:
            x = self.conv(x) + x
        return x