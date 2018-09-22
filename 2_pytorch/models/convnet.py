import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, im_size, hidden_dim, kernel_size, n_classes):
        '''
        Create components of a CNN classifier and initialize their weights.

        Arguments:
            im_size (tuple): A tuple of ints with (channels, height, width)
            hidden_dim (int): Number of hidden activations to use
            kernel_size (int): Width and height of (square) convolution filters
            n_classes (int): Number of classes to score
        '''
        super(CNN, self).__init__()
        #############################################################################
        # TODO: Initialize anything you need for the forward pass
        #############################################################################
        self.conv1 = nn.Conv2d(im_size[0], hidden_dim*3, kernel_size, 1, 1)
        final_image_dim=self.compute_image_size(im_size[1:],(kernel_size,kernel_size),1,1)
        #self.max_pool1 = nn.MaxPool2d(3, 1)
        #final_image_dim=self.compute_image_size(final_image_dim,(3,3),0,1)
        self.batch_norm1 = nn.BatchNorm2d(hidden_dim*3)
        self.conv2 = nn.Conv2d(hidden_dim*3, hidden_dim*2, (5,5), 1, 1)
        final_image_dim = self.compute_image_size(final_image_dim, (5, 5), 1, 1)
        self.max_pool2 = nn.MaxPool2d(3, 1)
        final_image_dim = self.compute_image_size(final_image_dim, (3, 3), 0, 1)
        self.batch_norm2 = nn.BatchNorm2d(hidden_dim*2)
        self.conv3=nn.Conv2d(hidden_dim*2,hidden_dim,kernel_size,1,1)
        final_image_dim = self.compute_image_size(final_image_dim, (kernel_size, kernel_size), 1, 1)
        self.max_pool3 = nn.MaxPool2d(3, 1)
        final_image_dim = self.compute_image_size(final_image_dim, (3, 3), 0, 1)
        self.batch_norm3=nn.BatchNorm2d(hidden_dim)
        self.linear=nn.Linear(hidden_dim*final_image_dim[0]*final_image_dim[1],hidden_dim)
        self.linear1 = nn.Linear(hidden_dim, n_classes)
        torch.nn.init.xavier_uniform_(self.linear1.weight)
        torch.nn.init.xavier_uniform_(self.linear.weight)
        torch.nn.init.xavier_uniform_(self.conv1.weight)
        torch.nn.init.xavier_uniform_(self.conv2.weight)
        torch.nn.init.xavier_uniform_(self.conv3.weight)
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################

    def compute_image_size(self,image_shape,filter_shape,padding=0,stride=1):
        f_height=1+(image_shape[0]+2*padding-filter_shape[0])//stride
        f_width=1+(image_shape[1]+2*padding-filter_shape[1])//stride
        return (f_height,f_width)

    def forward(self, images):
        '''
        Take a batch of images and run them through the CNN to
        produce a score for each class.

        Arguments:
            images (Variable): A tensor of size (N, C, H, W) where
                N is the batch size
                C is the number of channels
                H is the image height
                W is the image width

        Returns:
            A torch Variable of size (N, n_classes) specifying the score
            for each example and category.
        '''
        scores = None
        #############################################################################
        # TODO: Implement the forward pass. This should take few lines of code.
        #############################################################################
        out1=F.relu(self.batch_norm1(self.conv1(images)))
        out2=self.batch_norm2(self.max_pool2(F.relu(self.conv2(out1))))
        out3=self.batch_norm3(self.max_pool3(F.relu(self.conv3(out2))))
        scores=F.relu(self.linear(out3.view(out3.shape[0],-1)))
        scores=F.softmax(self.linear1(scores),1)


        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return scores

