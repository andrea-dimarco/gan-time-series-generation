from torch.nn import Module


# Just a function to count the number of parameters
def count_parameters(model: Module) -> int:
  """ Counts the number of trainable parameters of a module

  :param model: model that contains the parameters to count
  :returns: the number of parameters in the model
  """
  return sum(p.numel() for p in model.parameters() if p.requires_grad)