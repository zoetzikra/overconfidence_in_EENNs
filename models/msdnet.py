import torch.nn as nn
import torch
import math
import pdb
import numpy as np

class ConvBasic(nn.Module):
    def __init__(self, nIn, nOut, kernel=3, stride=1,
                 padding=1):
        super(ConvBasic, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(nIn, nOut, kernel_size=kernel, stride=stride,
                      padding=padding, bias=False),
            nn.BatchNorm2d(nOut),
            nn.ReLU()
        )

    def forward(self, x):
        out = self.net(x)
        return out


class ConvBN(nn.Module):
    def __init__(self, nIn, nOut, type: str, bottleneck,
                 bnWidth):
        """
        a basic conv in MSDNet, two type
        :param nIn:
        :param nOut:
        :param type: normal or down
        :param bottleneck: use bottleneck or not
        :param bnWidth: bottleneck factor
        """
        super(ConvBN, self).__init__()
        layer = []
        nInner = nIn
        if bottleneck is True:
            nInner = min(nInner, bnWidth * nOut)
            layer.append(nn.Conv2d(
                nIn, nInner, kernel_size=1, stride=1, padding=0, bias=False))
            layer.append(nn.BatchNorm2d(nInner))
            layer.append(nn.ReLU())

        if type == 'normal':
            layer.append(nn.Conv2d(nInner, nOut, kernel_size=3,
                                   stride=1, padding=1, bias=False))
        elif type == 'down':
            layer.append(nn.Conv2d(nInner, nOut, kernel_size=3,
                                   stride=2, padding=1, bias=False))
        else:
            raise ValueError

        layer.append(nn.BatchNorm2d(nOut))
        layer.append(nn.ReLU())

        self.net = nn.Sequential(*layer)
        #self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        return out


class ConvDownNormal(nn.Module):
    def __init__(self, nIn1, nIn2, nOut, bottleneck, bnWidth1, bnWidth2):
        super(ConvDownNormal, self).__init__()
        self.conv_down = ConvBN(nIn1, nOut // 2, 'down',
                                bottleneck, bnWidth1)
        self.conv_normal = ConvBN(nIn2, nOut // 2, 'normal',
                                   bottleneck, bnWidth2)

    def forward(self, x):
        res = [x[1],
               self.conv_down(x[0]),
               self.conv_normal(x[1])]
        return torch.cat(res, dim=1)


class ConvNormal(nn.Module):
    def __init__(self, nIn, nOut, bottleneck, bnWidth):
        super(ConvNormal, self).__init__()
        self.conv_normal = ConvBN(nIn, nOut, 'normal',
                                   bottleneck, bnWidth)

    def forward(self, x):
        if not isinstance(x, list):
            x = [x]
        res = [x[0],
               self.conv_normal(x[0])]

        return torch.cat(res, dim=1)

class MSDNFirstLayer(nn.Module):
    def __init__(self, nIn, nOut, args):
        super(MSDNFirstLayer, self).__init__()
        self.layers = nn.ModuleList()
        if args.data.startswith('cifar'):
            self.layers.append(ConvBasic(nIn, nOut * args.grFactor[0],
                                         kernel=3, stride=1, padding=1))
        elif args.data == 'caltech256':
            conv = nn.Sequential(
                    nn.Conv2d(nIn, nOut * args.grFactor[0], 7, 2, 3),
                    nn.BatchNorm2d(nOut * args.grFactor[0]),
                    nn.ReLU(),
                    nn.MaxPool2d(3, 2, 1))
            self.layers.append(conv)
        elif args.data == 'ImageNet':
            conv = nn.Sequential(
                    nn.Conv2d(nIn, nOut * args.grFactor[0], 7, 2, 3),
                    nn.BatchNorm2d(nOut * args.grFactor[0]),
                    nn.ReLU(),
                    nn.MaxPool2d(3, 2, 1))
            self.layers.append(conv)

        nIn = nOut * args.grFactor[0]

        for i in range(1, args.nScales):
            self.layers.append(ConvBasic(nIn, nOut * args.grFactor[i],
                                         kernel=3, stride=2, padding=1))
            nIn = nOut * args.grFactor[i]

    def forward(self, x):
        res = []
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            res.append(x)

        return res

class MSDNLayer(nn.Module):
    def __init__(self, nIn, nOut, args, inScales=None, outScales=None):
        super(MSDNLayer, self).__init__()
        self.nIn = nIn
        self.nOut = nOut
        self.inScales = inScales if inScales is not None else args.nScales
        self.outScales = outScales if outScales is not None else args.nScales

        self.nScales = args.nScales
        self.discard = self.inScales - self.outScales

        self.offset = self.nScales - self.outScales
        self.layers = nn.ModuleList()

        if self.discard > 0:
            nIn1 = nIn * args.grFactor[self.offset - 1]
            nIn2 = nIn * args.grFactor[self.offset]
            _nOut = nOut * args.grFactor[self.offset]
            self.layers.append(ConvDownNormal(nIn1, nIn2, _nOut, args.bottleneck,
                                              args.bnFactor[self.offset - 1],
                                              args.bnFactor[self.offset]))
        else:
            self.layers.append(ConvNormal(nIn * args.grFactor[self.offset],
                                          nOut * args.grFactor[self.offset],
                                          args.bottleneck,
                                          args.bnFactor[self.offset]))

        for i in range(self.offset + 1, self.nScales):
            nIn1 = nIn * args.grFactor[i - 1]
            nIn2 = nIn * args.grFactor[i]
            _nOut = nOut * args.grFactor[i]
            self.layers.append(ConvDownNormal(nIn1, nIn2, _nOut, args.bottleneck,
                                              args.bnFactor[i - 1],
                                              args.bnFactor[i]))

    def forward(self, x):
        if self.discard > 0:
            inp = []
            for i in range(1, self.outScales + 1):
                inp.append([x[i - 1], x[i]])
        else:
            inp = [[x[0]]]
            for i in range(1, self.outScales):
                inp.append([x[i - 1], x[i]])

        res = []
        for i in range(self.outScales):
            res.append(self.layers[i](inp[i]))

        return res


class ParallelModule(nn.Module):
    """
    This module is similar to luatorch's Parallel Table
    input: N tensor
    network: N module
    output: N tensor
    """
    def __init__(self, parallel_modules):
        super(ParallelModule, self).__init__()
        self.m = nn.ModuleList(parallel_modules)

    def forward(self, x):
        res = []
        for i in range(len(x)):
            res.append(self.m[i](x[i]))

        return res

class ClassifierModule(nn.Module):
    def __init__(self, m, channel, num_classes):
        super(ClassifierModule, self).__init__()
        self.m = m                                  # This is a sequential module (likely a sequence of convolutional or pooling layers) 
                                                    # that processes the input tensor before it’s fed into the linear classifier.
            
        self.linear = nn.Linear(channel, num_classes) # linear layer that maps the processed features to the number of classes 
                                                      # input: the flattened feature vector from the previous layers
                                                      # channel = the number of channels in the final output of the m module after processing the convolutional features

    def forward(self, x):
        res = self.m(x[-1])             # takes the last tensor in the list x. In MSDNet, x is likely a list of intermediate layer outputs, 
                                        # and x[-1] represents the final set of features after all the layers have processed the input.
                                        # res = self.m(x[-1]) applies the module self.m to the final output features from the previous layers.
        res = res.view(res.size(0), -1)     # This flattens the output from the self.m module, converting it to a 2D tensor where 
                                            # each row represents a sample in the batch. (flattened size is needed for compatibility with the linear layer)
        out = self.linear(res)
        return out

    ''' m: This parameter is a pre-built module (such as a sequence of convolutional and pooling layers) that 
        reduces the spatial dimensions of the features while retaining useful information. '''
        
class MLPProbe(nn.Module):
    def __init__(self, m, channel, num_classes):
        super(MLPProbe, self).__init__()
        self.m = m 
        self.linear = nn.Linear(channel, num_classes)

    def forward(self, x):
        res = self.m(x[-1])
        res = res.view(res.size(0), -1)
        out = self.linear(res)
        return out


# class MLPProbe(nn.Module):
#     def __init__(self, input_dim, output_dim): # input_dim should match the size of the intermediate layer output.
#         super(MLPProbe, self).__init__()       # output_dim is typically the number of classes
#         self.probe = nn.Sequential(
#             nn.Linear(input_dim, input_dim // 2), # fully connected layer that reduces the dimensionality of the input features
#             nn.ReLU(),                            # non-linearity
#             nn.Linear(input_dim // 2, output_dim) # fully connected layer that maps the input_dim // 2 features to the output_dim
#         )
    
#     def __init__(self, output_dim, pool_size=(4, 4)):
#         super(MLPProbe, self).__init__()
#         self.pool_size = pool_size
#         self.output_dim = output_dim
#         self.probe = None  # Initialize this after we know flattened_dim

    
#     def forward(self, x):
#         for i, tensor in enumerate(x):
#             print(f"Shape of tensor {i}: {tensor.shape}")
#         return self.probe(x)

#     def forward(self, x):
#         # If x is a list, concatenate or average along a suitable dimension
#         if isinstance(x, list):
#             for i, tensor in enumerate(x):
#                 print(f"Shape of tensor {i}: {tensor.shape}")
#             # Example: Concatenate along channel dimension (dim=1) if appropriate
#             x = [torch.nn.functional.adaptive_avg_pool2d(t, (self.common_height, self.common_width)) for t in x]
#             x = torch.cat(x, dim=1)  # or use torch.stack if they are batch items

#         # Flatten x to match input dimensions of Linear layer
#         x = x.view(x.size(0), -1)  # Flatten for Linear layer

#         # Forward through the probe layers
#         return self.probe(x)

#     def forward(self, x):
#         # Check if x is a list of tensors
#         if isinstance(x, list):
#             # Apply adaptive pooling to each tensor in the list individually
#             pooled_tensors = [torch.nn.functional.adaptive_avg_pool2d(tensor, self.pool_size) for tensor in x]
#             # Concatenate pooled tensors along the channel dimension
#             x = torch.cat(pooled_tensors, dim=1)
#         else:
#             # Apply adaptive pooling if x is a single tensor
#             x = torch.nn.functional.adaptive_avg_pool2d(x, self.pool_size)
        
#         # Dynamically calculate flattened dimension for Linear layer
#         batch_size = x.size(0)
#         flattened_dim = x.size(1) * x.size(2) * x.size(3)
        
#         # Initialize `self.probe` if it hasn’t been initialized yet
#         if self.probe is None:
#             self.probe = nn.Sequential(
#                 nn.Linear(flattened_dim, flattened_dim // 2),
#                 nn.ReLU(),
#                 nn.Linear(flattened_dim // 2, self.output_dim)
#             )
#             self.probe = self.probe.to(x.device)  # Ensure `probe` is on the correct device

#         # Flatten x to match input dimensions of Linear layer
#         x = x.view(batch_size, -1)
#         return self.probe(x)


class MSDNet(nn.Module):
    def __init__(self, args):
        super(MSDNet, self).__init__()
        self.blocks = nn.ModuleList()
        self.classifier = nn.ModuleList()
        self.probes = nn.ModuleList()  # Add probes
        self.nBlocks = args.nBlocks
        self.steps = [args.base]
        self.args = args
        
        n_layers_all, n_layer_curr = args.base, 0
        for i in range(1, self.nBlocks):
            self.steps.append(args.step if args.stepmode == 'even'
                             else args.step * i + 1)
            n_layers_all += self.steps[-1]

        print("building network of steps: ")
        print(self.steps, n_layers_all)

        nIn = args.nChannels
        for i in range(self.nBlocks):
            print(' ********************** Block {} '
                  ' **********************'.format(i + 1))
            m, nIn = \
                self._build_block(nIn, args, self.steps[i],
                                  n_layers_all, n_layer_curr)
            self.blocks.append(m)
            n_layer_curr += self.steps[i]

            if args.data.startswith('cifar100'):
                self.classifier.append(self._build_classifier_cifar(nIn * args.grFactor[-1], 100))
                
                self.probes.append(self._build_probe_cifar(nIn * args.grFactor[-1], 100))
                # self.probes.append(MLPProbe(nIn * args.grFactor[-1], 100))  # Add probe for CIFAR-100
                
                self.n_classes = 100
            
            elif args.data.startswith('cifar10'):
                self.classifier.append(
                    self._build_classifier_cifar(nIn * args.grFactor[-1], 10))
                self.n_classes = 10
            elif args.data == 'ImageNet':
                self.classifier.append(
                    self._build_classifier_imagenet(nIn * args.grFactor[-1], 1000))
                self.n_classes = 1000
            elif args.data == 'caltech256':
                self.classifier.append(
                    self._build_classifier_imagenet(nIn * args.grFactor[-1], 257))
                self.n_classes = 257
            else:
                raise NotImplementedError

        for m in self.blocks:
            if hasattr(m, '__iter__'):
                for _m in m:
                    self._init_weights(_m)
            else:
                self._init_weights(m)

        for m in self.classifier:
            if hasattr(m, '__iter__'): # checks if a module (m) has an iterable property i.e., it contains submodules
                for _m in m:
                    self._init_weights(_m)
            else:
                self._init_weights(m)
        
        for m in self.probes:  # Initialize probes
            if hasattr(m, '__iter__'):
                print("Would you look at that, the module has submodules")
                for _m in m:
                    self._init_weights(_m)
            else:
                self._init_weights(m)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.bias.data.zero_()

    def _build_block(self, nIn, args, step, n_layer_all, n_layer_curr):

        layers = [MSDNFirstLayer(3, nIn, args)] \
            if n_layer_curr == 0 else []
        for i in range(step):
            n_layer_curr += 1
            inScales = args.nScales
            outScales = args.nScales
            if args.prune == 'min':
                inScales = min(args.nScales, n_layer_all - n_layer_curr + 2)
                outScales = min(args.nScales, n_layer_all - n_layer_curr + 1)
            elif args.prune == 'max':
                interval = math.ceil(1.0 * n_layer_all / args.nScales)
                inScales = args.nScales - math.floor(1.0 * (max(0, n_layer_curr - 2)) / interval)
                outScales = args.nScales - math.floor(1.0 * (n_layer_curr - 1) / interval)
            else:
                raise ValueError

            layers.append(MSDNLayer(nIn, args.growthRate, args, inScales, outScales))
            print('|\t\tinScales {} outScales {} inChannels {} outChannels {}\t\t|'.format(inScales, outScales, nIn, args.growthRate))

            nIn += args.growthRate
            if args.prune == 'max' and inScales > outScales and \
                    args.reduction > 0:
                offset = args.nScales - outScales
                layers.append(
                    self._build_transition(nIn, math.floor(1.0 * args.reduction * nIn),
                                           outScales, offset, args))
                _t = nIn
                nIn = math.floor(1.0 * args.reduction * nIn)
                print('|\t\tTransition layer inserted! (max), inChannels {}, outChannels {}\t|'.format(_t, math.floor(1.0 * args.reduction * _t)))
            elif args.prune == 'min' and args.reduction > 0 and \
                    ((n_layer_curr == math.floor(1.0 * n_layer_all / 3)) or
                     n_layer_curr == math.floor(2.0 * n_layer_all / 3)):
                offset = args.nScales - outScales
                layers.append(self._build_transition(nIn, math.floor(1.0 * args.reduction * nIn),
                                                     outScales, offset, args))

                nIn = math.floor(1.0 * args.reduction * nIn)
                print('|\t\tTransition layer inserted! (min)\t|')
            print("")

        return nn.Sequential(*layers), nIn

    def _build_transition(self, nIn, nOut, outScales, offset, args):
        net = []
        for i in range(outScales):
            net.append(ConvBasic(nIn * args.grFactor[offset + i],
                                 nOut * args.grFactor[offset + i],
                                 kernel=1, stride=1, padding=0))
        return ParallelModule(net)

    def _build_classifier_cifar(self, nIn, num_classes):
        interChannels1, interChannels2 = 128, 128
        conv = nn.Sequential(
            ConvBasic(nIn, interChannels1, kernel=3, stride=2, padding=1),
            ConvBasic(interChannels1, interChannels2, kernel=3, stride=2, padding=1),
            nn.AvgPool2d(2),
        )
        return ClassifierModule(conv, interChannels2, num_classes)
    
    def _build_probe_cifar(self, nIn, num_classes):
        interChannels1, interChannels2 = 128, 128
        conv = nn.Sequential(
            ConvBasic(nIn, interChannels1, kernel=3, stride=2, padding=1),
            ConvBasic(interChannels1, interChannels2, kernel=3, stride=2, padding=1),
            nn.AvgPool2d(2),
        )
        return MLPProbe(conv, interChannels2, num_classes)

    def _build_classifier_imagenet(self, nIn, num_classes):
        conv = nn.Sequential(
            ConvBasic(nIn, nIn, kernel=3, stride=2, padding=1),
            ConvBasic(nIn, nIn, kernel=3, stride=2, padding=1),
            nn.AvgPool2d(2)
        )
        return ClassifierModule(conv, nIn, num_classes)

    # def forward(self, x):
    #     res = []
    #     for i in range(self.nBlocks):           # Each block self.blocks[i] is a sequence of layers that transforms x (the input or the output from the previous block)
    #         x = self.blocks[i](x)               # The updated x is then passed to the next block in the following iteration.
    #         res.append(self.classifier[i](x))   # res collects predictions from each intermediate classifier in the network.
    #     return res

    def forward(self, x, collect_intermediate=True):
        # if calculate_flops:
        #     collect_intermediate = False
    
        res = []
        probe_outputs = []

        for i in range(self.nBlocks):
            x = self.blocks[i](x)
            res.append(self.classifier[i](x))
            probe_outputs.append(self.probes[i](x))  # Get probe output
            # print(f"Shape of intermediate output at block {i}: {x[-1].shape}")

        if collect_intermediate:
            return res, probe_outputs  # Return both intermediate and probe outputs
        else:
            return res 

    def predict(self, x):
        """
        This function was coded for MC integration of LA inference
        it outputs final output (res) and mean feature of the image (phi_out) 
        """
        res = []     # Will store classifier outputs (logits)
        phi_out = [] # Will store feature representations before the final linear layer
        for i in range(self.nBlocks):
            # Forward pass through current block
            x = self.blocks[i](x)
            # Extract features using the classifier's convolutional layers
            phi = self.classifier[i].m(x[-1])
            # Flatten the features for the linear layer
            phi = phi.view(phi.size(0), -1)
            # Store both:
            res.append(self.classifier[i].linear(phi))
            phi_out.append(phi)
        return res, phi_out
    '''
    Feature Extraction:
        The learned intermediate features phi are crucial for LA because they're used to compute the posterior distribution
        By storing phi, we can later: Compute uncertainty estimates, Sample from the posterior over weights, and Calculate confidence scores
    Monte Carlo integration of LA:
        The stored features (phi_out) allow for MC sampling
        Multiple forward passes can be performed with different weight samples
        This helps estimate the posterior predictive distribution
    '''
        
    def predict_until(self, x, until_block):
        """
        This function was coded for MC integration of LA inference
        it outputs final output (res) and mean feature of the image (phi_out) 
        """
        res = []
        phi_out = []
        for i in range(until_block):
            x = self.blocks[i](x)
            # classifier module forward
            phi = self.classifier[i].m(x[-1])
            phi = phi.view(phi.size(0), -1)
            res.append(self.classifier[i].linear(phi))
            phi_out.append(phi)
        return res, phi_out


    def compute_max_softmax_confidence(self, logits):
        """Compute maximum softmax probability for each sample."""
        softmax_probs = torch.nn.functional.softmax(logits, dim=1)
        max_probs, _ = torch.max(softmax_probs, dim=1)
        
        return max_probs


    def compute_confidence_scores(self, x):
        """Compute confidence scores for each exit"""
        res = []
        confidences = []
        
        for i in range(self.nBlocks):
            x = self.blocks[i](x)
            logits = self.classifier[i](x)
            # logits = self.probes[i](x)
            confidence = self.compute_max_softmax_confidence(logits)
            res.append(logits)
            confidences.append(confidence)
            
        return res, confidences
