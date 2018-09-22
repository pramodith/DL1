#!/bin/sh
#############################################################################
# TODO: Modify the hyperparameters such as hidden layer dimensionality, 
#       number of epochs, weigh decay factor, momentum, batch size, learning 
#       rate mentioned here to achieve good performance
#############################################################################
python -u train.py \
    --model convnet \
    --kernel-size 3 \
    --hidden-dim 32 \
    --epochs 5 \
    --weight-decay 0.0001 \
    --momentum 0.0 \
    --batch-size 128 \
    --no-cuda\
    --lr 0.0001 | tee convnet.log
#############################################################################
#                             END OF YOUR CODE                              #
#############################################################################
