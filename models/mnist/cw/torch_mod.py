import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Function

from models.mnist.cw.memristor import V_STEPS, G_STEPS


class MemLinearFunction(Function):
    @staticmethod
    def forward(ctx, x, weight, bias, lookup_table, steps, table_size):
        # Save context for backward
        ctx.save_for_backward(x, weight, bias, lookup_table)
        ctx.steps = steps
        ctx.table_size = table_size

        # Quantize weights to steps
        quantized_weights = torch.round((weight + 1) * (steps - 1) / 2).long()
        quantized_weights = torch.clamp(quantized_weights, 0, steps - 1)

        # Quantize inputs to table_size
        quantized_inputs = torch.round((x + 1) * (table_size - 1) / 2).long()
        quantized_inputs = torch.clamp(quantized_inputs, 0, table_size - 1)

        # Fetch values from the lookup table
        output = torch.zeros(x.size(0), weight.size(0), device=x.device)
        for i in range(weight.size(0)):  # Iterate over output features
            for j in range(weight.size(1)):  # Iterate over input features
                table_values = lookup_table[
                    quantized_weights[i, j], quantized_inputs[:, j]
                ]
                output[:, i] += table_values

        # Add bias
        output += bias
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve saved tensors and parameters
        x, weight, _, lookup_table = ctx.saved_tensors
        steps = ctx.steps
        table_size = ctx.table_size

        # Gradients placeholders
        grad_x = None
        grad_weight = torch.zeros_like(weight)
        grad_bias = grad_output.sum(dim=0)
        grad_lookup_table = None

        # Gradient w.r.t. x
        grad_x = torch.zeros_like(x)
        quantized_weights = map_weight_index(weight, steps)
        quantized_inputs = map_table_index(x, table_size)

        for i in range(weight.size(0)):  # Iterate over output features
            for j in range(weight.size(1)):  # Iterate over input features
                weight_idx = quantized_weights[i, j]
                input_idx = quantized_inputs[:, j]  # Shape: [batch_size]

                # Use the lookup table to get corresponding values
                table_values_grad = lookup_table[
                    weight_idx, input_idx
                ]  # Shape: [batch_size]

                # Grad w.r.t. x
                grad_x[:, j] += grad_output[:, i] * table_values_grad

                # Grad w.r.t. weights
                grad_weight[i, j] += (grad_output[:, i] * table_values_grad).sum()

        return grad_x, grad_weight, grad_bias, grad_lookup_table, None, None


class memConv2dFunc(Function):
    @staticmethod
    def forward(ctx, x, weight, bias, lookup_table, stride, padding, steps, table_size):
        ctx.save_for_backward(x, weight, bias, lookup_table)
        ctx.steps = steps
        ctx.table_size = table_size
        ctx.stride = stride
        ctx.padding = padding

        x_unfold = F.unfold(
            x, kernel_size=weight.shape[2], stride=stride, padding=padding
        )

        # Shape of x_unfold: (batch_size, in_channels * kernel_size^2, num_patches)
        batch_size, num_features, num_patches = x_unfold.shape
        patch_size = weight.shape[2] * weight.shape[3]

        # Reshape the input patches to: (batch_size, num_patches, in_channels, kernel_size * kernel_size)
        x_unfold = x_unfold.view(
            batch_size, num_features // patch_size, patch_size, num_patches
        )

        # Quantize the weights to the number of steps (e.g., 256 levels between [-1, 1])
        quantized_weights = map_weight_index(weight, steps)
        quantized_weights = quantized_weights.view(
            1, weight.shape[0], weight.shape[1], patch_size
        ).expand(batch_size, -1, -1, -1)

        # Quantize the inputs to the table_size (e.g., 100 levels between [-1, 1])
        quantized_inputs = map_table_index(x_unfold, table_size)

        # Perform lookup table operations for each patch
        output = torch.zeros(batch_size, weight.shape[0], num_patches, device=x.device)

        # Iterate through the output channels
        for o_channel in range(weight.shape[0]):  # Output channels
            # Iterate through input channels
            for i_channel in range(weight.shape[1]):  # Input channels
                for patch_id in range(num_patches):  # Iterate through each patch size
                    # Quantized inputs shape:  torch.Size([64, 6, 25, 64])
                    # Quantized weights shape:  torch.Size([64, 16, 6, 25])
                    # Get the current quantized weight for the given output channel and patch
                    weight_value = quantized_weights[
                        :, o_channel, i_channel, :
                    ]  # Shape: [(64, 25)]

                    # Ensure weight_value is within valid range for lookup_table
                    weight_value = weight_value.clamp(
                        0, lookup_table.size(0) - 1
                    ).long()  # Shape: [64]

                    # Access the quantized input values for this input channel and all patches
                    quantized_input_values = quantized_inputs[
                        :, i_channel, :, patch_id
                    ].squeeze(-1)  # Shape: [64, 25]

                    # Prepare the input for the lookup table
                    # Ensure weight_value is of shape [64, 1] for proper broadcasting

                    # Perform lookup using weight_value and quantized_input_values
                    # Accessing the lookup table
                    table_values = lookup_table[weight_value, quantized_input_values]

                    # Now we need to accumulate across the second dimension of table_values
                    # Sum the table values across the patch dimension (dim=1)
                    summed_values = table_values.sum(dim=1)  # Shape: [64]

                    # Accumulate values into the output tensor for this output channel and patch
                    output[:, o_channel, patch_id] += summed_values

        out_frame_size = (x.size(2) + 2 * padding - weight.shape[2]) // stride + 1

        output = output.reshape(
            batch_size, weight.shape[0], out_frame_size, out_frame_size
        )

        # Add bias
        output += bias.view(1, weight.shape[0], 1, 1)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias, _ = ctx.saved_tensors
        stride = ctx.stride
        padding = ctx.padding
        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = torch.nn.grad.conv2d_input(input.shape, weight, grad_output, stride, padding)
        if ctx.needs_input_grad[1]:
            grad_weight = torch.nn.grad.conv2d_weight(input, weight.shape, grad_output, stride, padding) 
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum((0,2,3)).squeeze(0)

        return grad_input, grad_weight, grad_bias, None, None, None, None, None

    # @staticmethod
    # def backward(ctx, grad_output):
    #     # Retrieve saved tensors and parameters
    #     x, weight, _, lookup_table = ctx.saved_tensors
    #     stride, padding, steps, table_size = (
    #         ctx.stride,
    #         ctx.padding,
    #         ctx.steps,
    #         ctx.table_size,
    #     )

    #     # Prepare gradient placeholders
    #     grad_x = None
    #     grad_weight = None
    #     grad_bias = grad_output.sum(dim=(0, 2, 3))
    #     grad_lookup_table = None

    #     # Step 1: Compute gradient of output w.r.t. unfolded input
    #     batch_size, _, _, _ = grad_output.shape
    #     kernel_height, kernel_width = weight.shape[2:]

    #     # Step 2: Quantize weights (gradient approximation for quantized weights)
    #     quantized_weights = map_weight_index(weight, steps)

    #     # Prepare for lookup table gradient if required
    #     unfolded_x = F.unfold(
    #         x, kernel_size=(kernel_height, kernel_width), stride=stride, padding=padding
    #     )

    #     _, _, num_patches = unfolded_x.shape

    #     # Quantize inputs
    #     quantized_inputs = map_table_index(
    #         unfolded_x.view(
    #             batch_size, weight.shape[1], num_patches, kernel_height, kernel_width
    #         ),
    #         table_size,
    #     )

    #     grad_x_unfold = torch.zeros_like(unfolded_x)
    #     quantized_weights = quantized_weights.expand(batch_size, -1, -1, -1, -1)

    #     # Step 3: Compute gradient for each patch
    #     for o_channel in range(weight.shape[0]):  # Output channels
    #         for i_channel in range(weight.shape[1]):  # Input channels
    #             for patch_id in range(num_patches):
    #                 weight_value = quantized_weights[:, o_channel, i_channel, :, :]
    #                 quantized_input_values = quantized_inputs[
    #                     :, i_channel, patch_id, :, :
    #                 ]

    #                 # Lookup table values
    #                 table_values_grad = lookup_table[
    #                     weight_value, quantized_input_values
    #                 ]

    #                 # Backpropagate gradient
    #                 grad_x_unfold += torch.sum(
    #                     table_values_grad
    #                 )  # Adjust this logic for the lookup table operation

    #     # Step 4: Fold gradients back to the input tensor shape
    #     grad_x = F.fold(
    #         grad_x_unfold,
    #         output_size=(x.size(2), x.size(3)),
    #         kernel_size=(kernel_height, kernel_width),
    #         stride=stride,
    #         padding=padding,
    #     )

    #     return grad_x, grad_weight, grad_bias, grad_lookup_table, None, None, None, None


class memLinear(nn.Linear):
    def __init__(
        self,
        in_features,
        out_features,
        lookup_table,
        steps=G_STEPS,
        table_size=V_STEPS,
    ):
        super(memLinear, self).__init__(in_features, out_features)
        self.in_features = in_features
        self.out_features = out_features
        self.steps = steps
        self.table_size = table_size
        self.lookup_table = lookup_table.detach()
        self.lookup_table.requires_grad = False

        # Initialize the weights (quantized to a discrete range)
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        return MemLinearFunction.apply(
            x, self.weight, self.bias, self.lookup_table, self.steps, self.table_size
        )


class memConv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        lookup_table,
        stride=1,
        padding=0,
        steps=G_STEPS,
        table_size=V_STEPS,
    ):
        super(memConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding
        )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = (stride, stride)
        self.padding = (padding, padding)
        self.steps = steps
        self.table_size = table_size
        self.lookup_table = lookup_table.detach()
        self.lookup_table.requires_grad = False

        # Initialize the weights (quantized)
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x):
        # Extract input patches using F.unfold, which flattens patches into vectors
        # Input: (batch_size, in_channels, height, width)
        # Output: (batch_size, in_channels * kernel_size * kernel_size, num_patches)
        return memConv2dFunc.apply(
            x,
            self.weight,
            self.bias,
            self.lookup_table,
            self.stride[0],
            self.padding[0],
            self.steps,
            self.table_size,
        )


class memRelu(nn.ReLU):
    def __init__(self):
        super(memRelu, self).__init__()

    def forward(self, x):
        return torch.clamp(x, 0, 1)


def map_table_index(x, table_size=100):
    # Map the input x to the closest quantized input value
    quantized_x = torch.round((x + 1) * (table_size - 1) / 2).long()
    quantized_x = torch.clamp(quantized_x, 0, table_size - 1)
    return quantized_x


def map_weight_index(w, steps=256):
    # Map the weight w to the closest quantized weight index
    quantized_w = torch.round((w + 1) * (steps - 1) / 2).long()
    quantized_w = torch.clamp(quantized_w, 0, steps - 1)
    return quantized_w
