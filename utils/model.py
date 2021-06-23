import torch
from torch import nn
import math
import copy

class swinBlocks(object):
    """
        Class to create and handle swin block
    """
    def __init__(self, imageSize, C=96, patchesSize=4, windowSize=4):
        self.__patchSize = 4*4*3
        self.__C = C
        self.__patchesSize = patchesSize
        self.generateWeights(imageSize)
        self.__layerNorm = nn.functional.layer_norm
        self.__windowSize = windowSize

    def generateWeights(self, imageSize):
        """
            Method to generate randomly the weights
        """
        weights = []
        # Linear Embedding
        weights.append(torch.Tensor(self.__C, self.__patchSize).uniform_(0, 1/math.sqrt(self.__patchSize)).requires_grad_())
        weights.append(torch.Tensor(self.__C).zero_().requires_grad_())
        # SWINBLOCK 1
        k = 1
        # Linear 1 Embedding Attention
        weights.append(torch.Tensor(self.__C*3, self.__C).uniform_(0, 1/math.sqrt(self.__C)).requires_grad_())
        weights.append(torch.Tensor(self.__C*3).zero_().requires_grad_())
        # Scale self attention 1
        weights.append(torch.Tensor(1).zero_().requires_grad_())
        # MLP 1
        weights.append(torch.Tensor(self.__C, self.__C).uniform_(0, 1/math.sqrt(self.__C)).requires_grad_())
        weights.append(torch.Tensor(self.__C).zero_().requires_grad_())
        weights.append(torch.Tensor(self.__C, self.__C).uniform_(0, 1/math.sqrt(self.__C)).requires_grad_())
        weights.append(torch.Tensor(self.__C).zero_().requires_grad_())
        # Linear 2 Embedding Attention
        weights.append(torch.Tensor(self.__C*3, self.__C).uniform_(0, 1/math.sqrt(self.__C)).requires_grad_())
        weights.append(torch.Tensor(self.__C*3).zero_().requires_grad_())
        # Scale self attention 2
        weights.append(torch.Tensor(1).zero_().requires_grad_())
        # MLP 2
        weights.append(torch.Tensor(self.__C, self.__C).uniform_(0, 1/math.sqrt(self.__C)).requires_grad_())
        weights.append(torch.Tensor(self.__C).zero_().requires_grad_())
        weights.append(torch.Tensor(self.__C, self.__C).uniform_(0, 1/math.sqrt(self.__C)).requires_grad_())
        weights.append(torch.Tensor(self.__C).zero_().requires_grad_())

        k = 2
        # Downsample 1

        weights.append(torch.Tensor(self.__C*k, self.__C*(k*2)).uniform_(0, 1/math.sqrt(self.__C*k)).requires_grad_())
        weights.append(torch.Tensor(self.__C*k).zero_().requires_grad_())

        # SWINBLOCK 2
        
        # Linear 1 Embedding Attention
        weights.append(torch.Tensor(self.__C*3*k, self.__C*k).uniform_(0, 1/math.sqrt(self.__C*k)).requires_grad_())
        weights.append(torch.Tensor(self.__C*3*k).zero_().requires_grad_())
        # Scale self attention 1
        weights.append(torch.Tensor(1).zero_().requires_grad_())
        # MLP 1
        weights.append(torch.Tensor(self.__C*k, self.__C*k).uniform_(0, 1/math.sqrt(self.__C*k)).requires_grad_())
        weights.append(torch.Tensor(self.__C*k).zero_().requires_grad_())
        weights.append(torch.Tensor(self.__C*k, self.__C*k).uniform_(0, 1/math.sqrt(self.__C*k)).requires_grad_())
        weights.append(torch.Tensor(self.__C*k).zero_().requires_grad_())
        # Linear 2 Embedding Attention
        weights.append(torch.Tensor(self.__C*3*k, self.__C*k).uniform_(0, 1/math.sqrt(self.__C*k)).requires_grad_())
        weights.append(torch.Tensor(self.__C*3*k).zero_().requires_grad_())
        # Scale self attention 2
        weights.append(torch.Tensor(1).zero_().requires_grad_())
        # MLP 2
        weights.append(torch.Tensor(self.__C*k, self.__C*k).uniform_(0, 1/math.sqrt(self.__C*k)).requires_grad_())
        weights.append(torch.Tensor(self.__C*k).zero_().requires_grad_())
        weights.append(torch.Tensor(self.__C*k, self.__C*k).uniform_(0, 1/math.sqrt(self.__C*k)).requires_grad_())
        weights.append(torch.Tensor(self.__C*k).zero_().requires_grad_())

        k = 4
        # Downsample 2

        weights.append(torch.Tensor(self.__C*k, self.__C*(k*2)).uniform_(0, 1/math.sqrt(self.__C*k)).requires_grad_())
        weights.append(torch.Tensor(self.__C*k).zero_().requires_grad_())

        # SWINBLOCK 3
        
        # Linear 1 Embedding Attention
        weights.append(torch.Tensor(self.__C*3*k, self.__C*k).uniform_(0, 1/math.sqrt(self.__C*k)).requires_grad_())
        weights.append(torch.Tensor(self.__C*3*k).zero_().requires_grad_())
        # Scale self attention 1
        weights.append(torch.Tensor(1).zero_().requires_grad_())
        # MLP 1
        weights.append(torch.Tensor(self.__C*k, self.__C*k).uniform_(0, 1/math.sqrt(self.__C*k)).requires_grad_())
        weights.append(torch.Tensor(self.__C*k).zero_().requires_grad_())
        weights.append(torch.Tensor(self.__C*k, self.__C*k).uniform_(0, 1/math.sqrt(self.__C*k)).requires_grad_())
        weights.append(torch.Tensor(self.__C*k).zero_().requires_grad_())
        # Linear 2 Embedding Attention
        weights.append(torch.Tensor(self.__C*3*k, self.__C*k).uniform_(0, 1/math.sqrt(self.__C*k)).requires_grad_())
        weights.append(torch.Tensor(self.__C*3*k).zero_().requires_grad_())
        # Scale self attention 2
        weights.append(torch.Tensor(1).zero_().requires_grad_())
        # MLP 2
        weights.append(torch.Tensor(self.__C*k, self.__C*k).uniform_(0, 1/math.sqrt(self.__C*k)).requires_grad_())
        weights.append(torch.Tensor(self.__C*k).zero_().requires_grad_())
        weights.append(torch.Tensor(self.__C*k, self.__C*k).uniform_(0, 1/math.sqrt(self.__C*k)).requires_grad_())
        weights.append(torch.Tensor(self.__C*k).zero_().requires_grad_())

        k = 8
        # Downsample 3

        weights.append(torch.Tensor(self.__C*k, self.__C*(k*2)).uniform_(0, 1/math.sqrt(self.__C*k)).requires_grad_())
        weights.append(torch.Tensor(self.__C*k).zero_().requires_grad_())

        # SWINBLOCK 4
        
        # Linear 1 Embedding Attention
        weights.append(torch.Tensor(self.__C*3*k, self.__C*k).uniform_(0, 1/math.sqrt(self.__C*k)).requires_grad_())
        weights.append(torch.Tensor(self.__C*3*k).zero_().requires_grad_())
        # Scale self attention 1
        weights.append(torch.Tensor(1).zero_().requires_grad_())
        # MLP 1
        weights.append(torch.Tensor(self.__C*k, self.__C*k).uniform_(0, 1/math.sqrt(self.__C*k)).requires_grad_())
        weights.append(torch.Tensor(self.__C*k).zero_().requires_grad_())
        weights.append(torch.Tensor(self.__C*k, self.__C*k).uniform_(0, 1/math.sqrt(self.__C*k)).requires_grad_())
        weights.append(torch.Tensor(self.__C*k).zero_().requires_grad_())
        # Linear 2 Embedding Attention
        weights.append(torch.Tensor(self.__C*3*k, self.__C*k).uniform_(0, 1/math.sqrt(self.__C*k)).requires_grad_())
        weights.append(torch.Tensor(self.__C*3*k).zero_().requires_grad_())
        # Scale self attention 2
        weights.append(torch.Tensor(1).zero_().requires_grad_())
        # MLP 2
        weights.append(torch.Tensor(self.__C*k, self.__C*k).uniform_(0, 1/math.sqrt(self.__C*k)).requires_grad_())
        weights.append(torch.Tensor(self.__C*k).zero_().requires_grad_())
        weights.append(torch.Tensor(self.__C*k, self.__C*k).uniform_(0, 1/math.sqrt(self.__C*k)).requires_grad_())
        weights.append(torch.Tensor(self.__C*k).zero_().requires_grad_())

        self.__weights = weights

    def updateWeights(self, newWeights):
        """
            Method to update new weights
        """
        try:
            for i in range(len(self.__weights)):
                shapeOriginal = self.__weights[i].shape
                shapeNew = newWeights[i].shape

                if shapeOriginal != shapeNew:
                    print(shapeOriginal)
                    print(shapeNew)
                    print("ERROR: Dimensions mismatch")
                    print("The new weights have different dimensions compared to the original weights")
                    return
        except:
            print("ERROR: Dimensions mismatch")
            print("The new weights have different dimensions compared to the original weights")
            return

        self.__weights = newWeights

    def getWeights(self):
        """
            Method to get current Weights
        """
        return copy.deepcopy(self.__weights)

    def __linearEmbeddingOnPatches(self, patches, weights):
        """
            Method to calculate linear embedding over the patches
        """
        x = torch.reshape(patches, (patches.shape[0]*patches.shape[1]*patches.shape[2],
                                    patches.shape[3]*patches.shape[4]*patches.shape[5]
                                    ))

        x = torch.nn.functional.linear(
                                       x,
                                       weight=weights[0],
                                       bias=weights[1]
                                      )

        return torch.reshape(x, (patches.shape[0], patches.shape[1], patches.shape[2], x.shape[1]))

    def __swinBlock(self, input, weights):
        """
            Method to execute a swinBlock
        """
        res = input
        # FIRST BLOCK
        # Norm layer
        x = self.__layerNorm(input, input.shape)
        # Linear Layer Attention
        x = torch.reshape(x, (x.shape[0]*x.shape[1]*x.shape[2],
                              x.shape[3]
                                    ))

        x = torch.nn.functional.linear(
                                       x,
                                       weight=weights[0],
                                       bias=weights[1]
                                      )

        x = torch.reshape(x, (input.shape[0], input.shape[1], input.shape[2], x.shape[1]))
        # Window picking
        x = x.unfold(1, self.__windowSize, self.__windowSize).unfold(2, self.__windowSize, self.__windowSize)
        
        x = torch.reshape(x, (x.shape[0], x.shape[1]*x.shape[2], int(x.shape[3]*x.shape[4]*x.shape[5]/3), 3)).permute(3,0,1,2)
        q = x[0]
        k = x[1]
        v = x[2]
        # Self attention
        attention = torch.reshape((nn.functional.softmax((q*weights[2])@k.transpose(-2,-1), dim=2)@v),(res.shape))
        # Residual
        x = attention + res
        res = x
        # Normal layer
        x = self.__layerNorm(x, x.shape)
        # MLP layer
        x = torch.reshape(x, (x.shape[0]*x.shape[1]*x.shape[2],
                              x.shape[3]
                             ))
        x = torch.nn.functional.linear(
                                       x,
                                       weight=weights[3],
                                       bias=weights[4]
                                      )
        x = torch.nn.functional.gelu(x)
        x = torch.nn.functional.linear(
                                       x,
                                       weight=weights[5],
                                       bias=weights[6]
                                      )
        x = torch.reshape(x, (res.shape))
        # Residual
        x = x + res
        res = x

        # SECOND BLOCK
        # Norm layer
        x = self.__layerNorm(input, input.shape)
        # Linear Layer Attention
        x = torch.reshape(x, (x.shape[0]*x.shape[1]*x.shape[2],
                              x.shape[3]
                                    ))

        x = torch.nn.functional.linear(
                                       x,
                                       weight=weights[7],
                                       bias=weights[8]
                                      )

        x = torch.reshape(x, (input.shape[0], input.shape[1], input.shape[2], x.shape[1]))
        # Window picking
        x = x.roll(2,2)
        x = x.unfold(1, self.__windowSize, self.__windowSize).unfold(2, self.__windowSize, self.__windowSize)
        
        x = torch.reshape(x, (x.shape[0], x.shape[1]*x.shape[2], int(x.shape[3]*x.shape[4]*x.shape[5]/3), 3)).permute(3,0,1,2)
        q = x[0]
        k = x[1]
        v = x[2]
        # Self attention
        attention = torch.reshape((nn.functional.softmax((q*weights[9])@k.transpose(-2,-1), dim=2)@v),(res.shape))
        # Residual
        x = attention + res
        res = x
        # Normal layer
        x = self.__layerNorm(x, x.shape)
        # MLP layer
        x = torch.reshape(x, (x.shape[0]*x.shape[1]*x.shape[2],
                              x.shape[3]
                             ))
        x = torch.nn.functional.linear(
                                       x,
                                       weight=weights[10],
                                       bias=weights[11]
                                      )
        x = torch.nn.functional.gelu(x)
        x = torch.nn.functional.linear(
                                       x,
                                       weight=weights[12],
                                       bias=weights[13]
                                      )
        x = torch.reshape(x, (res.shape))
        # Residual
        x = x + res
        res = x

        #print(x.shape)

        return x


    def forward(self, x, weights, training=True):
        """
            Method to execute forward of the base learner model using torch.functional
            this due that nn.Module does not keep track of the gradients and therefore
            functional is needed to be able to calculate the gradient of the loss respect
            the meta learner parameters, going through the base learner forward execution

            x shape must be: batchSize x H x W x 3
        """
        if weights == None:
            weights = self.__weights

        #x = torch.flatten(x, start_dim=0)

        #Patch Partition in: HxWx3, out: H/4xW/4x48
        size = self.__patchesSize # patch size
        stride = self.__patchesSize # patch stride
        patches = x.unfold(1, size, stride).unfold(2, size, stride)

        #Linear Embbedding
        x = self.__linearEmbeddingOnPatches(patches, weights=weights[0:2])

        # Swin Transformer Block 1
        x = self.__swinBlock(x, weights=weights[2:16])

        # Merge Patches 1
        size = 2
        stride = 2
        x = x.unfold(1, size, stride).unfold(2, size, stride)
        
        input = torch.reshape(x, (x.shape[0]*x.shape[1]*x.shape[2],
                              x.shape[3]*x.shape[4]*x.shape[5]
                             ))

        input = torch.nn.functional.linear(
                                           input,
                                           weight=weights[16],
                                           bias=weights[17]
                                          )

        x = torch.reshape(input, (x.shape[0], x.shape[1], x.shape[2], input.shape[1]))

        # Swin Transformer Block 2
        x = self.__swinBlock(x, weights=weights[18:32])

        # Merge Patches 2
        size = 2
        stride = 2
        x = x.unfold(1, size, stride).unfold(2, size, stride)
        
        input = torch.reshape(x, (x.shape[0]*x.shape[1]*x.shape[2],
                              x.shape[3]*x.shape[4]*x.shape[5]
                             ))

        input = torch.nn.functional.linear(
                                           input,
                                           weight=weights[32],
                                           bias=weights[33]
                                          )

        x = torch.reshape(input, (x.shape[0], x.shape[1], x.shape[2], input.shape[1]))

        # Swin Transformer Block 3
        x = self.__swinBlock(x, weights=weights[34:48])

        # Merge Patches 3
        size = 2
        stride = 2
        x = x.unfold(1, size, stride).unfold(2, size, stride)
        
        input = torch.reshape(x, (x.shape[0]*x.shape[1]*x.shape[2],
                              x.shape[3]*x.shape[4]*x.shape[5]
                             ))

        input = torch.nn.functional.linear(
                                           input,
                                           weight=weights[48],
                                           bias=weights[49]
                                          )

        x = torch.reshape(input, (x.shape[0], x.shape[1], x.shape[2], input.shape[1]))

        # Swin Transformer Block 4
        x = self.__swinBlock(x, weights=weights[50:64])

        #x = torch.nn.functional.relu(x)
        #Linear Block 1
        #x = torch.nn.functional.linear(x,
        #                               weight=weights[2],
        #                               bias=weights[3])
        # Separating elements by Lr and Re
        #x = x.reshape(2,int(len(x)/2))

        return x