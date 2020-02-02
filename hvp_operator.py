import torch


class Operator():
    def __init__(self, size):
        self._size = size

    def __matmul__(self, vec):
        return self.apply(vec)

    def apply(self, vec):
        raise NotImplementedError()

    def size(self):
        return self._size


class ModelHessianOperator(Operator):
    def __init__(self, model, criterion, data_input, data_target):
        size = int(sum(p.numel() for p in model.parameters()))
        super(ModelHessianOperator, self).__init__(size)
        self._model = model
        self._criterion = criterion
        self.set_model_data(data_input, data_target)

    def apply(self, vec):
        return to_vector(torch.autograd.grad(self._grad, self._model.parameters()
                                             , grad_outputs=vec, only_inputs=True, retain_graph=True))

    def set_model_data(self, data_input, data_target):
        self._data_input = data_input
        self._data_target = data_target
        self._output = self._model(self._data_input)
        self._loss = self._criterion(self._output, self._data_target)
        self._grad = to_vector(torch.autograd.grad(self._loss, self._model.parameters(), create_graph=True))

    def get_input(self):
        return self._data_input

    def get_target(self):
        return self._data_target


def to_vector(tensors):
    return torch.cat([t.contiguous().view(-1) for t in tensors])