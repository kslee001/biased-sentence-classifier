from loadlibs import *
# from freezing import roberta_unfreeze_list, distilbert_unfreeze_list



class BasicClassifier(torch.nn.Module):
    def __init__(self, encoder, linear_dim=1024):
        super().__init__()
        self.encoder = encoder
        encoder.train()
        self.linear1 = torch.nn.Linear(linear_dim, linear_dim//2)
        self.linear2 = torch.nn.Linear(linear_dim//2, 2)
        self.activate = torch.nn.GELU(approximate='tanh')
        
    def forward(self, input_ids, attention_mask):
        x = self.encoder(input_ids=input_ids, attention_mask=attention_mask)['last_hidden_state'][:, 0, :] # logits / pooler_output
        x = self.activate(self.linear1(x))
        x = self.linear2(x)
        return x
        
        
class ExpClassifier(torch.nn.Module):
    def __init__(self, encoder, linear_dim=1024):
        super().__init__()
        self.activate = torch.nn.GELU(approximate='tanh')
        self.head_mask = [None]*encoder.config.num_hidden_layers
        
        self.embeddings = encoder.embeddings
        self.transformer = encoder.transformer
        self.CrossEntropyLinear = torch.nn.Linear(linear_dim, 2)
        self.ContrastiveLinear = torch.nn.Sequential(
            torch.nn.Linear(linear_dim, 256),
            self.activate,
            torch.nn.Linear(256, 64)
        )
        
        self.eps = 0.05
        
    def forward(self, input_ids, attention_mask, temp_embeddings=None, device=None):        
        # adversarial attack
        if temp_embeddings is not None:
            with torch.no_grad():
                temp_embeddings = temp_embeddings.to(device)
                x = temp_embeddings.to(device)(input_ids)
        else: 
            x = self.embeddings(input_ids)
            
        x = self.transformer(
            x=x, 
            attn_mask=attention_mask,
            head_mask=self.head_mask,
            output_attentions=[None],
            output_hidden_states=None,
            return_dict=True,
        )['last_hidden_state'][:, 0, :]
        
        yhat = self.CrossEntropyLinear(x)
        feature = self.ContrastiveLinear(x)
        
        return yhat, feature



        
        
class ContrastiveClassifier(torch.nn.Module):
    def __init__(self, encoder, linear_dim=1024):
        super().__init__()
        self.encoder = encoder
        self.linear1 = torch.nn.Linear(linear_dim, linear_dim//2)
        self.linear2 = torch.nn.Linear(linear_dim//2, 2)
        self.activate = torch.nn.GELU(approximate='tanh')
        
    # Fast Gradient Sign Method explanation
     # https://rain-bow.tistory.com/entry/%EC%A0%81%EB%8C%80%EC%A0%81-%EA%B3%B5%EA%B2%A9Adversarial-Attack-FGSMPGD
    # code : https://pytorch.org/tutorials/beginner/fgsm_tutorial.html



class FGSM(torch.nn.Module):
    def __init__(self, model, eps, target=None):
        super().__init__()
        self.model = model
        self.target = target
        self.eps = eps
    
    def forward(self, x, y):
        x_adv = x.detach().clone()
        x_adv.requires_grad = True
        self.model.zero_grad()
        logit = self.model(x_adv)
        if self.target is None:
            cost = -torch.nn.functional.cross_entropy(logit, y)
        else:
            cost = torch.nn.functional.cross_entropy(logit, self.target)
        
        if x_adv.grad is not None:
            x_adv.grad.data.fill_(0)
        cost.backward()
        
        x_adv.grad.sign_()
        x_adv = x_adv - self.eps*x_adv.grad
        x_adv = torch.clamp(x_adv, *self.clamp)
                
        return x_adv
        
        





# --------------------------------------------------------------------------    
class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, mode='train'):
        self.text = data.text.values
        if mode in ['train', 'valid']:
            self.gender = data.gender.astype(int).values.astype(int)
            self.toxic = data.toxic.astype(int).values.astype(int)
        else:
            self.gender = None
            self.toxic = None
        self.tokenizer = tokenizer
        self.mode = mode
        
    def __len__(self):
        return len(self.text)
    
    def __getitem__(self, idx):
        encoded = self.tokenizer(
            self.text[idx],
            add_special_tokens=True,
            max_length = 256,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        ids = encoded["input_ids"].squeeze(0)
        mask = encoded["attention_mask"].squeeze(0)
        encoded = {'input_ids':ids, 'attention_mask':mask}
        
        if self.mode in ['train', 'valid']:
            return (encoded, torch.tensor([self.gender[idx], self.toxic[idx]]))
        else:
            return encoded
        
        
class ExpDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, mode='train'):
        self.text = data.text.values
        if mode in ['train', 'valid']:
            self.gender = data.gender.astype(int).values.astype(int)
            self.toxic = data.toxic.astype(int).values.astype(int)
        else:
            self.gender = None
            self.toxic = None
        self.tokenizer = tokenizer
        self.mode = mode
        
    def __len__(self):
        return len(self.text)
    
    def __getitem__(self, idx):
        encoded = self.tokenizer(
            self.text[idx],
            add_special_tokens=True,
            max_length = 256,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        ids = encoded["input_ids"].squeeze(0)
        mask = encoded["attention_mask"].squeeze(0)
        encoded = {'input_ids':ids, 'attention_mask':mask}
        
        if self.mode in ['train', 'valid']:
            return (encoded, torch.tensor([self.gender[idx], self.toxic[idx]]))
        else:
            return encoded
        
        
# ----------------------------------------------------------------------
def clone_module(module, memo=None):
    """
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/utils.py)
    **Description**
    Creates a copy of a module, whose parameters/buffers/submodules
    are created using PyTorch's torch.clone().
    This implies that the computational graph is kept, and you can compute
    the derivatives of the new modules' parameters w.r.t the original
    parameters.
    **Arguments**
    * **module** (Module) - Module to be cloned.
    **Return**
    * (Module) - The cloned module.
    **Example**
    ~~~python
    net = nn.Sequential(Linear(20, 10), nn.ReLU(), nn.Linear(10, 2))
    clone = clone_module(net)
    error = loss(clone(X), y)
    error.backward()  # Gradients are back-propagate all the way to net.
    ~~~
    """
    # NOTE: This function might break in future versions of PyTorch.

    # TODO: This function might require that module.forward()
    #       was called in order to work properly, if forward() instanciates
    #       new variables.
    # TODO: We can probably get away with a shallowcopy.
    #       However, since shallow copy does not recurse, we need to write a
    #       recursive version of shallow copy.
    # NOTE: This can probably be implemented more cleanly with
    #       clone = recursive_shallow_copy(model)
    #       clone._apply(lambda t: t.clone())

    if memo is None:
        # Maps original data_ptr to the cloned tensor.
        # Useful when a Module uses parameters from another Module; see:
        # https://github.com/learnables/learn2learn/issues/174
        memo = {}

    # First, create a copy of the module.
    # Adapted from:
    # https://github.com/pytorch/pytorch/blob/65bad41cbec096aa767b3752843eddebf845726f/torch/nn/modules/module.py#L1171
    if not isinstance(module, torch.nn.Module):
        return module
    clone = module.__new__(type(module))
    clone.__dict__ = module.__dict__.copy()
    clone._parameters = clone._parameters.copy()
    clone._buffers = clone._buffers.copy()
    clone._modules = clone._modules.copy()

    # Second, re-write all parameters
    if hasattr(clone, '_parameters'):
        for param_key in module._parameters:
            if module._parameters[param_key] is not None:
                param = module._parameters[param_key]
                param_ptr = param.data_ptr
                if param_ptr in memo:
                    clone._parameters[param_key] = memo[param_ptr]
                else:
                    cloned = param.clone()
                    clone._parameters[param_key] = cloned
                    memo[param_ptr] = cloned

    # Third, handle the buffers if necessary
    if hasattr(clone, '_buffers'):
        for buffer_key in module._buffers:
            if clone._buffers[buffer_key] is not None and \
                    clone._buffers[buffer_key].requires_grad:
                buff = module._buffers[buffer_key]
                buff_ptr = buff.data_ptr
                if buff_ptr in memo:
                    clone._buffers[buffer_key] = memo[buff_ptr]
                else:
                    cloned = buff.clone()
                    clone._buffers[buffer_key] = cloned
                    memo[param_ptr] = cloned

    # Then, recurse for each submodule
    if hasattr(clone, '_modules'):
        for module_key in clone._modules:
            clone._modules[module_key] = clone_module(
                module._modules[module_key],
                memo=memo,
            )

    # Finally, rebuild the flattened parameters for RNNs
    # See this issue for more details:
    # https://github.com/learnables/learn2learn/issues/139
    if hasattr(clone, 'flatten_parameters'):
        clone = clone._apply(lambda x: x)
    return clone