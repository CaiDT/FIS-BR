import torch
from torch import nn
from copy import deepcopy

global REC, ATTENTION_LIST, Model
REC = False
ATTENTION_LIST = []

TemporalAttention_List = []
SpatialAttention_List = []

class LLL_Net(nn.Module):
    """Basic class for implementing networks"""

    def __init__(self, model, remove_existing_head=False):
        if hasattr(model, 'head_var'):
            head_var = model.head_var
        # assert type(head_var) == str
        # assert not remove_existing_head or hasattr(model, head_var), \
        #     "Given model does not have a variable called {}".format(head_var)
        # assert not remove_existing_head or type(getattr(model, head_var)) in [nn.Sequential, nn.Linear], \
        #     "Given model's head {} does is not an instance of nn.Sequential or nn.Linear".format(head_var)
        super(LLL_Net, self).__init__()

        self.model = model
        # last_layer = getattr(self.model, head_var)

        # if remove_existing_head:
        #     if type(last_layer) == nn.Sequential:
        #         self.out_size = last_layer[-1].in_features
        #         # strips off last linear layer of classifier
        #         del last_layer[-1]
        #     elif type(last_layer) == nn.Linear:
        #         self.out_size = last_layer.in_features
        #         # converts last layer into identity
        #         # setattr(self.model, head_var, nn.Identity())
        #         # WARNING: this is for when pytorch version is <1.2
        #         setattr(self.model, head_var, nn.Sequential())
        # else:
        #     self.out_size = last_layer.out_features

        # self.out_size = last_layer.out_features

        self.heads = nn.ModuleList()
        self.task_cls = []
        self.task_offset = []
        self._initialize_weights()

    def add_head(self, num_input, num_outputs=1):
        """Add a new head with the corresponding number of outputs. Also update the number of classes per task and the
        corresponding offsets
        """
        self.heads.append(nn.Linear(num_input, num_outputs))
        # # we re-compute instead of append in case an approach makes changes to the heads
        # self.task_cls = torch.tensor([head.out_features for head in self.heads])
        # self.task_offset = torch.cat([torch.LongTensor(1).zero_(), self.task_cls.cumsum(0)[:-1]])

    def forward(self, x, return_features=False):
        """Forward pass for regression-based AVEC setting."""

        if Model == 'STA':
            if return_features:
                predict, features = self.model(x, return_features=True)
                predict = predict * 63
                predict = predict.view(predict.size(0))
                return predict, features
            else:
                predict = self.model(x)
                predict = predict * 63
                predict = predict.view(predict.size(0))
                return predict

        elif Model == 'OVit_tiny_16_augreg_224':
            features = self.model(x)
            predict = self.heads[0](features)
            predict = predict * 63
            predict = predict.view(predict.size(0))

            if return_features:
                return predict, features

            return predict


    def get_copy(self):
        """Get weights from the model"""
        return deepcopy(self.state_dict())

    def set_state_dict(self, state_dict):
        """Load weights into the model"""
        self.load_state_dict(deepcopy(state_dict))
        return

    def freeze_all(self):
        """Freeze all parameters from the model, including the heads"""
        for param in self.parameters():
            param.requires_grad = False

    def freeze_backbone(self):
        """Freeze all parameters from the main model, but not the heads"""
        for param in self.model.parameters():
            param.requires_grad = False

    def freeze_bn(self):
        """Freeze all Batch Normalization layers from the model and use them in eval() mode"""
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def _initialize_weights(self):
        """Initialize weights using different strategies"""
        # TODO: add different initialization strategies
        pass
