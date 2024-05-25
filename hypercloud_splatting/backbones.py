from hypercloud.models.aae import HyperNetwork as Sphere2ModelDecoder, Encoder, TargetNetwork as Sphere2ModelTargetNetwork
import torch
import torch.nn as nn

TARGET_INPUT_SIZE = 108 # based on embedding dimension, create single source of truth later
TARGET_OUTPUT_SIZE = lambda config, gstype: sum([channels for _, channels in config['model'][f'GS_TN_{gstype}']['gs_params_out_channels'].items()], 0) 

class Face2GSShapeTargetNetwork(nn.Module):
    def __init__(self, config, weights):
        super().__init__()

        self.z_size = config['z_size']
        self.use_bias = config['model']['GS_TN_SHAPE']['use_bias']
        # target network layers out channels
        out_ch = config['model']['GS_TN_SHAPE']['layer_out_channels']

        layer_data, split_index = self._get_layer_data(start_index=0, end_index=out_ch[0] * TARGET_INPUT_SIZE,
                                                       shape=(out_ch[0], TARGET_INPUT_SIZE), weights=weights)
        self.layers = {"1": layer_data}

        for x in range(1, len(out_ch)):
            layer_data, split_index = self._get_layer_data(start_index=split_index,
                                                           end_index=split_index + (out_ch[x - 1] * out_ch[x]),
                                                           shape=(out_ch[x], out_ch[x - 1]), weights=weights)
            self.layers[str(x + 1)] = layer_data

        layer_data, split_index = self._get_layer_data(start_index=split_index,
                                                       end_index=split_index + (out_ch[-1] * TARGET_OUTPUT_SIZE(config, 'SHAPE')),
                                                       shape=(TARGET_OUTPUT_SIZE(config, 'SHAPE'), out_ch[-1]), weights=weights)
        self.output = layer_data
        self.activation = torch.nn.ReLU()
        assert split_index == len(weights)

    def forward(self, x):
        for layer_index in self.layers:
            x = torch.mm(x, torch.transpose(self.layers[layer_index]["weight"], 0, 1))
            if self.use_bias:
                assert "bias" in self.layers[layer_index]
                x = x + self.layers[layer_index]["bias"]
            x = self.activation(x)
        return torch.mm(x, torch.transpose(self.output["weight"], 0, 1)) + self.output.get("bias", 0)

    def _get_layer_data(self, start_index, end_index, shape, weights):
        layer_data = {"weight": weights[start_index:end_index].view(shape[0], shape[1])}
        if self.use_bias:
            layer_data["bias"] = weights[end_index:end_index + shape[0]]
            end_index = end_index + shape[0]
        return layer_data, end_index
    
class Face2GSColorTargetNetwork(nn.Module):
    def __init__(self, config, weights):
        super().__init__()

        self.z_size = config['z_size']
        self.use_bias = config['model']['GS_TN_COLOR']['use_bias']
        # target network layers out channels
        out_ch = config['model']['GS_TN_COLOR']['layer_out_channels']

        layer_data, split_index = self._get_layer_data(start_index=0, end_index=out_ch[0] * TARGET_INPUT_SIZE,
                                                       shape=(out_ch[0], TARGET_INPUT_SIZE), weights=weights)
        self.layers = {"1": layer_data}

        for x in range(1, len(out_ch)):
            layer_data, split_index = self._get_layer_data(start_index=split_index,
                                                           end_index=split_index + (out_ch[x - 1] * out_ch[x]),
                                                           shape=(out_ch[x], out_ch[x - 1]), weights=weights)
            self.layers[str(x + 1)] = layer_data

        layer_data, split_index = self._get_layer_data(start_index=split_index,
                                                       end_index=split_index + (out_ch[-1] * TARGET_OUTPUT_SIZE(config, 'COLOR')),
                                                       shape=(TARGET_OUTPUT_SIZE(config, 'COLOR'), out_ch[-1]), weights=weights)
        self.output = layer_data
        self.activation = torch.nn.ReLU()
        assert split_index == len(weights)

    def forward(self, x):
        for layer_index in self.layers:
            x = torch.mm(x, torch.transpose(self.layers[layer_index]["weight"], 0, 1))
            if self.use_bias:
                assert "bias" in self.layers[layer_index]
                x = x + self.layers[layer_index]["bias"]
            x = self.activation(x)
        return torch.mm(x, torch.transpose(self.output["weight"], 0, 1)) + self.output.get("bias", 0)

    def _get_layer_data(self, start_index, end_index, shape, weights):
        layer_data = {"weight": weights[start_index:end_index].view(shape[0], shape[1])}
        if self.use_bias:
            layer_data["bias"] = weights[end_index:end_index + shape[0]]
            end_index = end_index + shape[0]
        return layer_data, end_index
    
class Face2GSTargetNetwork(nn.Module):
    def __init__(self, config, weights):
        super().__init__()

        self.z_size = config['z_size']
        self.use_bias = config['model']['GS_TN_SHAPE']['use_bias']
        # target network layers out channels
        out_ch = config['model']['GS_TN_SHAPE']['layer_out_channels']

        TOTAL_OUTPUT_SIZE = TARGET_OUTPUT_SIZE(config, 'SHAPE') + TARGET_OUTPUT_SIZE(config, 'COLOR')

        layer_data, split_index = self._get_layer_data(start_index=0, end_index=out_ch[0] * TARGET_INPUT_SIZE,
                                                       shape=(out_ch[0], TARGET_INPUT_SIZE), weights=weights)
        self.layers = {"1": layer_data}

        for x in range(1, len(out_ch)):
            layer_data, split_index = self._get_layer_data(start_index=split_index,
                                                           end_index=split_index + (out_ch[x - 1] * out_ch[x]),
                                                           shape=(out_ch[x], out_ch[x - 1]), weights=weights)
            self.layers[str(x + 1)] = layer_data

        layer_data, split_index = self._get_layer_data(start_index=split_index,
                                                       end_index=split_index + (out_ch[-1] * TOTAL_OUTPUT_SIZE),
                                                       shape=(TOTAL_OUTPUT_SIZE, out_ch[-1]), weights=weights)
        self.output = layer_data
        self.activation = torch.nn.ReLU()
        assert split_index == len(weights)

    def forward(self, x):
        for layer_index in self.layers:
            x = torch.mm(x, torch.transpose(self.layers[layer_index]["weight"], 0, 1))
            if self.use_bias:
                assert "bias" in self.layers[layer_index]
                x = x + self.layers[layer_index]["bias"]
            x = self.activation(x)
        return torch.mm(x, torch.transpose(self.output["weight"], 0, 1)) + self.output.get("bias", 0)

    def _get_layer_data(self, start_index, end_index, shape, weights):
        layer_data = {"weight": weights[start_index:end_index].view(shape[0], shape[1])}
        if self.use_bias:
            layer_data["bias"] = weights[end_index:end_index + shape[0]]
            end_index = end_index + shape[0]
        return layer_data, end_index

# Takes architecture from SHAPE TN but has output channels for all necessary outputs
class Face2GSDecoder(nn.Module):
    def __init__(self, config, device):
        super().__init__()

        self.z_size = config['z_size']
        self.use_bias = config['model']['GS_HN']['use_bias']
        self.relu_slope = config['model']['GS_HN']['relu_slope']
        TOTAL_OUTPUT_SIZE = TARGET_OUTPUT_SIZE(config, 'SHAPE') + TARGET_OUTPUT_SIZE(config, 'COLOR')
        
        # target network layers out channels
        target_network_out_ch = [TARGET_INPUT_SIZE] + config['model']['GS_TN_SHAPE']['layer_out_channels'] + [TOTAL_OUTPUT_SIZE]
        target_network_use_bias = int(config['model']['GS_TN_SHAPE']['use_bias'])

        # self.model = nn.Sequential(
        #     nn.Linear(in_features=self.z_size, out_features=64, bias=self.use_bias),
        #     nn.ReLU(inplace=False),

        #     nn.Linear(in_features=64, out_features=128, bias=self.use_bias),
        #     nn.ReLU(inplace=False),

        #     nn.Linear(in_features=128, out_features=512, bias=self.use_bias),
        #     nn.ReLU(inplace=False),

        #     nn.Linear(in_features=512, out_features=1024, bias=self.use_bias),
        #     nn.ReLU(inplace=False),

        #     nn.Linear(in_features=1024, out_features=2048, bias=self.use_bias),
        # )

        self.linears = []
        self.linears.append(nn.Linear(in_features=self.z_size, out_features=64, bias=self.use_bias)) #czemu tak jak z_size to 2048
        self.linears.append(nn.Linear(in_features=64, out_features=128, bias=self.use_bias))
        self.linears.append(nn.Linear(in_features=128, out_features=512, bias=self.use_bias))
        self.linears.append(nn.Linear(in_features=512, out_features=1024, bias=self.use_bias))
        self.linears.append(nn.Linear(in_features=1024, out_features=2048, bias=self.use_bias))
        self.activation = nn.ReLU(inplace=True)

        self.linears = nn.ModuleList(self.linears)

        self.output = [
            nn.Linear(2048, (target_network_out_ch[x - 1] + target_network_use_bias) * target_network_out_ch[x],
                      bias=True).to(device)
            for x in range(1, len(target_network_out_ch))
        ]

        if not config['model']['GS_TN_SHAPE']['freeze_layers_learning']:
            self.output = nn.ModuleList(self.output)

    def forward(self, x):
        for i, layer in enumerate(self.linears):
            x = nn.functional.linear(x, layer.weight.clone(),layer.bias)
            if i != len(self.linears):
                x = self.activation(x)
        target_weights = [nn.functional.linear(x, target_network_layer.weight.clone(),target_network_layer.bias) for target_network_layer in self.output]
        return torch.cat(target_weights, 1)


class Face2GSShapeDecoder(nn.Module):
    def __init__(self, config, device):
        super().__init__()

        self.z_size = config['z_size']
        self.use_bias = config['model']['GS_HN']['use_bias']
        self.relu_slope = config['model']['GS_HN']['relu_slope']
        
        # target network layers out channels
        target_network_out_ch = [TARGET_INPUT_SIZE] + config['model']['GS_TN_SHAPE']['layer_out_channels'] + [TARGET_OUTPUT_SIZE(config, 'SHAPE')]
        target_network_use_bias = int(config['model']['GS_TN_SHAPE']['use_bias'])

        # self.model = nn.Sequential(
        #     nn.Linear(in_features=self.z_size, out_features=64, bias=self.use_bias),
        #     nn.ReLU(inplace=False),

        #     nn.Linear(in_features=64, out_features=128, bias=self.use_bias),
        #     nn.ReLU(inplace=False),

        #     nn.Linear(in_features=128, out_features=512, bias=self.use_bias),
        #     nn.ReLU(inplace=False),

        #     nn.Linear(in_features=512, out_features=1024, bias=self.use_bias),
        #     nn.ReLU(inplace=False),

        #     nn.Linear(in_features=1024, out_features=2048, bias=self.use_bias),
        # )

        self.linears = []
        self.linears.append(nn.Linear(in_features=self.z_size, out_features=64, bias=self.use_bias)) #czemu tak jak z_size to 2048
        self.linears.append(nn.Linear(in_features=64, out_features=128, bias=self.use_bias))
        self.linears.append(nn.Linear(in_features=128, out_features=512, bias=self.use_bias))
        self.linears.append(nn.Linear(in_features=512, out_features=1024, bias=self.use_bias))
        self.linears.append(nn.Linear(in_features=1024, out_features=2048, bias=self.use_bias))
        self.activation = nn.ReLU(inplace=True)

        self.linears = nn.ModuleList(self.linears)

        self.output = [
            nn.Linear(2048, (target_network_out_ch[x - 1] + target_network_use_bias) * target_network_out_ch[x],
                      bias=True).to(device)
            for x in range(1, len(target_network_out_ch))
        ]

        if not config['model']['GS_TN_SHAPE']['freeze_layers_learning']:
            self.output = nn.ModuleList(self.output)

    def forward(self, x):
        for i, layer in enumerate(self.linears):
            x = nn.functional.linear(x, layer.weight.clone(),layer.bias)
            if i != len(self.linears):
                x = self.activation(x)
        target_weights = [nn.functional.linear(x, target_network_layer.weight.clone(),target_network_layer.bias) for target_network_layer in self.output]
        return torch.cat(target_weights, 1)
    
class Face2GSColorDecoder(nn.Module):
    def __init__(self, config, device):
        super().__init__()

        self.z_size = config['z_size']
        self.use_bias = config['model']['GS_HN']['use_bias']
        self.relu_slope = config['model']['GS_HN']['relu_slope']
        
        # target network layers out channels
        target_network_out_ch = [TARGET_INPUT_SIZE] + config['model']['GS_TN_COLOR']['layer_out_channels'] + [TARGET_OUTPUT_SIZE(config, 'COLOR')]
        target_network_use_bias = int(config['model']['GS_TN_COLOR']['use_bias'])

        # self.model = nn.Sequential(
        #     nn.Linear(in_features=self.z_size, out_features=64, bias=self.use_bias),
        #     nn.ReLU(inplace=False),

        #     nn.Linear(in_features=64, out_features=128, bias=self.use_bias),
        #     nn.ReLU(inplace=False),

        #     nn.Linear(in_features=128, out_features=512, bias=self.use_bias),
        #     nn.ReLU(inplace=False),

        #     nn.Linear(in_features=512, out_features=1024, bias=self.use_bias),
        #     nn.ReLU(inplace=False),

        #     nn.Linear(in_features=1024, out_features=2048, bias=self.use_bias),
        # )

        self.linears = []
        self.linears.append(nn.Linear(in_features=self.z_size, out_features=64, bias=self.use_bias)) #czemu tak jak z_size to 2048
        self.linears.append(nn.Linear(in_features=64, out_features=128, bias=self.use_bias))
        self.linears.append(nn.Linear(in_features=128, out_features=512, bias=self.use_bias))
        self.linears.append(nn.Linear(in_features=512, out_features=1024, bias=self.use_bias))
        self.linears.append(nn.Linear(in_features=1024, out_features=2048, bias=self.use_bias))
        self.activation = nn.ReLU(inplace=True)

        self.linears = nn.ModuleList(self.linears)

        self.output = [
            nn.Linear(2048, (target_network_out_ch[x - 1] + target_network_use_bias) * target_network_out_ch[x],
                      bias=True).to(device)
            for x in range(1, len(target_network_out_ch))
        ]

        if not config['model']['GS_TN_COLOR']['freeze_layers_learning']:
            self.output = nn.ModuleList(self.output)

    def forward(self, x):
        for i, layer in enumerate(self.linears):
            x = nn.functional.linear(x, layer.weight.clone(),layer.bias)
            if i != len(self.linears):
                x = self.activation(x)
        target_weights = [nn.functional.linear(x, target_network_layer.weight.clone(),target_network_layer.bias) for target_network_layer in self.output]
        return torch.cat(target_weights, 1)