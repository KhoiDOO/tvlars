from torch import optim

opt_map = {
    'adam' : optim.Adam,
    'adamw' : optim.AdamW,
    'adagrad' : optim.Adagrad,
    'rmsprop' : optim.RMSprop
}

def get_opt(opt_name: str = 'adam', params = None, learning_rate = 0.001, weight_decay = 1e-4):
    return opt_map[opt_name](
        params = params,
        weight_decay = weight_decay,
        lr = learning_rate
    )