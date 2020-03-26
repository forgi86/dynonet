import torch

torch.autograd

# Inherit from Function
class LinearFunction(torch.autograd.Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, weight, bias=None):
        ctx.save_for_backward(input, weight, bias)
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias


if __name__ == '__main__':

     from torch.autograd import gradcheck

     linear = LinearFunction.apply
     # gradcheck takes a tuple of tensors as input, check if your gradient
     # evaluated with these tensors are close enough to numerical
     # approximations and returns True if they all verify this condition.

     batch_size = 128
     in_feat = 20
     out_feat = 30
     X = torch.randn(batch_size, in_feat, requires_grad=True)
     W = torch.randn(out_feat, in_feat, requires_grad=True)
     Y = linear(X, W)  # shape: (batch_size, out_feat)
     Y_sum = torch.sum(Y, dim=1)
     #Y.backward(torch.ones(batch_size, out_feat))
     Y_sum.backward(torch.ones(batch_size))
     #input = (X, W)
     #test = gradcheck(linear, input, eps=1e-6, atol=1e-4)
     #print(test)