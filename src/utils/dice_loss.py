# Libraries Import
import torch
import torch.nn as nn
import torch.nn.functional as F




def soft_dice_loss(output, target, epsilon=1e-6):
    numerator = 2. * torch.sum(output * target, dim=(-2, -1))
    denominator = torch.sum(output + target, dim=(-2, -1))
    
    return (numerator + epsilon) / (denominator + epsilon)


class SoftDiceLoss(nn.Module):
    def __init__(self, reduction='none', use_softmax=True):
        """
        Args:
            use_softmax: Set it to False when use the function for testing purpose
        """
        super(SoftDiceLoss, self).__init__()
        self.use_softmax = use_softmax
        self.reduction = reduction

    def forward(self, output, target, epsilon=1e-6):
        """
        References:
        JeremyJordan's Implementation
        https://gist.github.com/jeremyjordan/9ea3032a32909f71dd2ab35fe3bacc08#file-soft_dice_loss-py

        Paper related to this function:
        Formula for binary segmentation case - A survey of loss functions for semantic segmentation
        https://arxiv.org/pdf/2006.14822.pdf

        Formula for multiclass segmentation cases - Segmentation of Head and Neck Organs at Risk Using CNN with Batch
        Dice Loss
        https://arxiv.org/pdf/1812.02427.pdf

        Args:
            output: Tensor shape (N, N_Class, H, W), torch.float
            target: Tensor shape (N, H, W)
            epsilon: Use this term to avoid undefined edge case

        Returns:

        """
        num_classes = output.shape[1]                                                   # ; print("Number of classes: ", num_classes)
        # Apply softmax to the output to present it in probability.
        if self.use_softmax:
            output = F.softmax(output, dim=1)
        one_hot_target = F.one_hot(target.to(torch.int64), num_classes=num_classes)     # ; print("One hot shape1: ", one_hot_target.shape)
        one_hot_target = one_hot_target.permute((0, 3, 1, 2)).to(torch.float)           # ; print("One hot shape: ", one_hot_target.shape)
        assert output.shape == one_hot_target.shape
        if self.reduction == 'none':
            return 1.0 - soft_dice_loss(output, one_hot_target)
        elif self.reduction == 'mean':
            return 1.0 - torch.mean(soft_dice_loss(output, one_hot_target))
        else:
            raise NotImplementedError(f"Invalid reduction mode: {self.reduction}")