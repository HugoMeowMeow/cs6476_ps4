import torch
import torch.nn as nn


class SimpleNetDropout(nn.Module):
    def __init__(self):
        """
        Init function to define the layers and loss function

        Note: Use 'sum' reduction in the loss_criterion. Read Pytorch documention to understand what it means
        """
        super().__init__()

        # self.conv_layers = nn.Sequential()
        # self.fc_layers = nn.Sequential()
        # self.loss_criterion = None
        self.conv_layers = nn.Sequential(nn.Conv2d(1, 10, 5, 1), nn.MaxPool2d(3,3), nn.ReLU(),nn.Dropout(0.1),
                                         nn.Conv2d(10,20, 5, 1, 2), nn.MaxPool2d(3,4), nn.ReLU(),nn.Dropout(0.1), nn.Flatten())  # conv2d and supporting layers here
        self.fc_layers = nn.Sequential(nn.Linear(500, 100), nn.ReLU(), nn.Dropout(0.1), nn.Linear(100,15))  # linear and supporting layers here
        self.loss_criterion = nn.CrossEntropyLoss(reduction='sum')


        ############################################################################
        # Student code begin
        ############################################################################

        # raise NotImplementedError('SimpleNetDropout not implemented')

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
        conv_features = None  # output of x passed through convolution layers (4D tensor)
        flattened_conv_features = None  # conv_features reshaped into 2D tensor using .reshape()
        model_output = None  # output of flattened_conv_features passed through fully connected layers
        ############################################################################
        # Student code begin
        ############################################################################
        # raise NotImplementedError('SimpleNetDropout not implemented')
        conv_features = self.conv_layers(x)
        # flattened_conv_features = torch.Flatten(conv_features)
        model_output = self.fc_layers(conv_features)
        ############################################################################
        # Student code end
        ############################################################################

        return model_output
