import torch
import torch.nn as nn


class SimpleNet(nn.Module):
    """Simple Network with atleast 2 conv2d layers and two linear layers."""

    def __init__(self):
        """
        Init function to define the layers and loss function

        Note: Use 'sum' reduction in the loss_criterion. Read Pytorch documention to understand what it means

        Hints:
        1. Refer to https://pytorch.org/docs/stable/nn.html for layers
        2. Remember to use non-linearities in your network. Network without
        non-linearities is not deep.
        3. You will get 3D tensor for an image input from self.cnn_layers. You need 
        to process it and make it a compatible tensor input for self.fc_layers.
        """
        super().__init__()

        self.conv_layers = nn.Sequential(nn.Conv2d(1, 10, 5, 1), nn.MaxPool2d(3,3), nn.ReLU(),
                                         nn.Conv2d(10,20, 5, 1, 2), nn.MaxPool2d(3,4), nn.ReLU(), nn.Flatten())  # conv2d and supporting layers here
        self.fc_layers = nn.Sequential(nn.Linear(500, 100), nn.Linear(100,15))  # linear and supporting layers here
        self.loss_criterion = nn.CrossEntropyLoss(reduction='sum')

        ############################################################################
        # Student code begin
        ############################################################################

        # raise NotImplementedError('SimpleNet class not implemented') 

        ############################################################################
        # Student code end
        ############################################################################

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Perform the forward pass with the net

        Note: do not perform soft-max or convert to probabilities in this function

        Args:
        -   x: the input image [Dim: (N,C,H,W)]
        Returns:
        -   y: the output (raw scores) of the net [Dim: (N,15)]
        """
        #The following are helper variables and are OPTIONAL to use. They are meant to help guide you to calculate model_output.

        conv_features = None  # output of x passed through convolution layers (4D tensor)
        flattened_conv_features = None  # conv_features reshaped into 2D tensor using .reshape(), , or nn.Flatten()
        model_output = None  # output of flattened_conv_features passed through fully connected layers
        ############################################################################
        # Student code begin
        ############################################################################
        # raise NotImplementedError('forward function of SimpleNet not implemented')
        conv_features = self.conv_layers(x)
        # flattened_conv_features = torch.Flatten(conv_features)
        model_output = self.fc_layers(conv_features)
        ############################################################################
        # Student code end
        ############################################################################

        return model_output
