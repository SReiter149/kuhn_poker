import torch
from abc import ABC, abstractmethod
from typing import Optional, Iterable

def get_seed(
    upper=1 << 31
) -> int:
    """
    Generates a random integer by the `torch` PRNG,
    to be used as seed in a stochastic function.

    Parameters
    ----------
    upper : int, optional
        Exclusive upper bound of the interval to generate integers from.
        Default: 1 << 31.

    Returns
    -------
    A random integer.
    """
    return int(torch.randint(upper, size=()))

class Linear(torch.nn.Module):
    """
    Ensemble-ready affine transformation `y = x^T W + b`.

    Arguments
    ---------
    config : `dict`
        Configuration dictionary. Required key-value pairs:
        `"device"` : `str`
            The device to store parameters on.
        `"ensemble_shape"` : `tuple[int]`
            The shape of the ensemble of affine transformations
            the model represents.
    in_features : `int`
        The number of input features
    out_features : `int`
        The number of output features.
    bias : `bool`, optional
        Whether the model should include bias. Default: `True`.
    init_multiplier : `float`, optional
        The weight parameter values are initialized following
        a normal distribution with center 0 and std
        `in_features ** -.5` times this value. Default: `1.`

    Calling
    -------
    Instance calls require one positional argument:
    features : `torch.Tensor`
        The input tensor. It is required to be one of the following shapes:
        1. `ensemble_shape + batch_shape + (in_features,)`
        2. `batch_shape + (in_features,)

        Upon a call, the model thinks we're in the first case
        if the first `len(ensemble_shape)` many entries of the
        shape of the input tensor is `ensemble_shape`.
    """
    def __init__(
        self,
        config: dict,
        in_features: int,
        out_features: int,
        bias=True,
        init_multiplier=1.
    ):
        super().__init__()

        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(
                config["ensemble_shape"] + (out_features,),
                device=config["device"],
                dtype=torch.float32
            ))
        else:
            self.bias = None

        self.weight = torch.nn.Parameter(torch.empty(
            config["ensemble_shape"] + (in_features, out_features),
            device=config["device"],
            dtype=torch.float32
        ).normal_(std=out_features ** -.5) * init_multiplier)


    def forward(
        self,
        features: torch.Tensor
    ) -> torch.Tensor:
        ensemble_shape = self.weight.shape[:-2]
        ensemble_dim = len(ensemble_shape)
        ensemble_input = features.shape[:ensemble_dim] == ensemble_shape
        batch_dim = len(features.shape) - 1 - ensemble_dim * ensemble_input
        
        # (*e, *b, i) @ (*e, *b[:-1], i, o)
        weight = self.weight.reshape(
            ensemble_shape
          + (1,) * (batch_dim - 1)
          + self.weight.shape[-2:]
        )
        features = features @ weight

        if self.bias is None:
            return features
        
        # (*e, *b, o) + (*e, *b, o)
        bias = self.bias.reshape(
            ensemble_shape
          + (1,) * batch_dim
          + self.bias.shape[-1:]
        )
        features = features + bias

        return features
    
def get_mlp(
    config: dict,
    in_features: int,
    out_features: int,
    hidden_layer_num: Optional[int] = None,
    hidden_layer_size: Optional[int] = None,
    hidden_layer_sizes: Optional[Iterable[int]] = None,
) -> torch.nn.Sequential:
    """
    Creates an MLP with ReLU activation functions.
    Can create a model ensemble.

    config : `dict`
        Configuration dictionary. Required key-value pairs:
        `"device"` : `str`
            The device to store parameters on.
        `"ensemble_shape"` : `tuple[int]`
            The shape of the ensemble of affine transformations
            the model represents.
    in_features : `int`
        The number of input features
    out_features : `int`
        The number of output features.
    hidden_layer_num : `int`, optional
        If `hidden_layer_sizes` is not given, we create an MLP with
        `hidden_layer_num` hidden layers of
        `hidden_layer_size` dimensions.
    hidden_layer_size : `int`, optional
        If `hidden_layer_sizes` is not given, we create an MLP with
        `hidden_layer_num` hidden layers of
        `hidden_layer_size` dimensions.
    hidden_layer_sizes: `Iterable[int]`, optional
        If given, each entry gives a hidden layer with the given size.
    """
    if hidden_layer_sizes is None:
        hidden_layer_sizes = (hidden_layer_size,) * hidden_layer_num

    layers = []
    layer_in_size = in_features
    for layer_out_size in hidden_layer_sizes:
        layers.extend([
            Linear(
                config,
                layer_in_size,
                layer_out_size,
                init_multiplier=2 ** .5
            ),
            torch.nn.ReLU()
        ])
        layer_in_size = layer_out_size
    
    layers.append(Linear(
        config,
        layer_in_size,
        out_features
    ))

    return torch.nn.Sequential(*layers)


class Optimizer(ABC):
    """
    Optimizer base class.
    Can optimize model ensembles
    with training defined by hyperparameter ensembles.

    Arguments
    ---------
    parameters : `Iterable[torch.nn.Parameter]`
        An iterable of `torch.nn.Parameter` to track.
        In a simple case of optimizing a single `model: torch.nn.Module`,
        this can be `model.parameters()`.
    config : `dict`, optional
        If given, the `update_config` method is called on it
        to initialize hyperparameters. Default: `None`.

    Class attributes
    ----------------
    keys : `tuple[str]`
        The collection of the hyperparameter keys to track
        in the configuration dictionary.

        We expect the hyperparameter values to be either
        `float` or `torch.Tensor`. In the latter case,
        we expect the shape to be a prefix of the shape of the parameters.
        The hyperparameter shapes are regarded as ensemble shapes.

        Required keys:
        `"learning_rate"`
        `"weight_decay"`

    Instance attributes
    -------------------
    config : `dict`
        The hyperparameter dictionary.
    parameters : `list[torch.nn.Parameter]`
        The list of tracked parameters.
    step_id : `int`
        Train step counter.
    """
    keys=(
        "learning_rate",
        "weight_decay"
    )
    def __init__(
        self,
        parameters: Iterable[torch.nn.Parameter],
        config=None
    ):
        self.config = dict()
        self.parameters = list(parameters)
        self.step_id = 0

        if config is not None:
            self.update_config(config)
    

    def get_parameters(self) -> Iterable[torch.Tensor]:
        """
        Get an iterable over tracked parameters
        and optimizer state tensors.
        """
        return iter(self.parameters)


    def get_hyperparameter(
        self,
        key: str,
        parameter: torch.Tensor
    ) -> torch.Tensor:
        """
        Take the hyperparameter with name `key`,
        transform it to `torch.Tensor` with the same
        `device` and `dtype` as `parameter`
        and reshape it to be broadcastable
        to `parameter` by postfixing to its shape
        an appropriate number of dimensions of 1.
        """        
        hyperparameter = torch.asarray(
            self.config[key],
            device=parameter.device,
            dtype=parameter.dtype
        )

        return hyperparameter.reshape(
            hyperparameter.shape
            + (
                len(parameter.shape)
                - len(hyperparameter.shape)
            )
            * (1,)
        )


    def step(self):
        """
        Update optimizer state, then apply parameter updates in-place.
        Assumes that backpropagation has already occurred by
        a call to the `backward` method of the loss tensor.
        """
        self.step_id += 1
        with torch.no_grad():
            for i, parameter in enumerate(self.parameters):
                self._update_parameter(parameter, i)


    def update_config(self, config: dict):
        """
        Update hyperparameters by the values in `config: dict`.
        """
        for key in self.keys:
            self.config[key] = config[key]


    def zero_grad(self):
        """
        Make the `grad` attribute of each tracked parameter `None`.
        """
        for parameter in self.parameters:
            parameter.grad = None


    def _apply_parameter_update(
        self,
        parameter: torch.nn.Parameter,
        parameter_update: torch.Tensor
    ):
        parameter += parameter_update


    @abstractmethod
    def _get_parameter_update(
        self,
        parameter: torch.nn.Parameter,
        parameter_id: int
    ) -> torch.Tensor:
        if self.config["weight_decay"] is None:
            return torch.zeros_like(parameter)
        
        return -(
            self.get_hyperparameter("learning_rate", parameter)
          * self.get_hyperparameter("weight_decay", parameter)
          * parameter
        )


    def _update_state(
        self,
        parameter: torch.nn.Parameter,
        parameter_id: int
    ):
        pass


    def _update_parameter(
        self,
        parameter: torch.nn.Parameter,
        parameter_id: int
    ):
        self._update_state(parameter, parameter_id)
        parameter_update = self._get_parameter_update(
            parameter,
            parameter_id
        )
        self._apply_parameter_update(
            parameter,
            parameter_update
        )
    


class AdamW(Optimizer):
    """
    Adam optimizer with optionally weight decay.
    Can optimize model ensembles
    with training defined by hyperparameter ensembles.

    Arguments
    ---------
    parameters : `Iterable[torch.nn.Parameter]`
        An iterable of `torch.nn.Parameter` to track.
        In a simple case of optimizing a single `model: torch.nn.Module`,
        this can be `model.parameters()`.
    config : `dict`, optional
        If given, the `update_config` method is called on it
        to initialize hyperparameters. Default: `None`.

    Class attributes
    ----------------
    keys : `tuple[str]`
        The collection of the hyperparameter keys to track
        in the configuration dictionary.

        We expect the hyperparameter values to be either
        `float` or `torch.Tensor`. In the latter case,
        we expect the shape to be a prefix of the shape of the parameters.
        The hyperparameter shapes are regarded as ensemble shapes.

        Required keys:
        `"epsilon"`,
        `"first_moment_decay"`,
        `"learning_rate"`
        `"second_moment_decay"`,
        `"weight_decay"`
    """
    keys = (
        "epsilon",
        "first_moment_decay",
        "learning_rate",
        "second_moment_decay",
        "weight_decay"
    )
    def __init__(
        self,
        parameters: Iterable[torch.nn.Parameter],
        config=None
    ):
        super().__init__(parameters, config)
        self.first_moments = [
            torch.zeros_like(parameter)
            for parameter in self.parameters
        ]
        self.second_moments = [
            torch.zeros_like(parameter)
            for parameter in self.parameters
        ]


    def get_parameters(self) -> Iterable[torch.Tensor]:
        yield from self.parameters
        yield from self.first_moments
        yield from self.second_moments


    def _get_parameter_update(
        self,
        parameter: torch.nn.Parameter,
        parameter_id: int
    ) -> torch.Tensor:
        parameter_update = super()._get_parameter_update(
            parameter,
            parameter_id
        )

        epsilon = self.get_hyperparameter(
            "epsilon",
            parameter
        )
        first_moment = self.first_moments[parameter_id]
        first_moment_decay = self.get_hyperparameter(
            "first_moment_decay",
            parameter
        )
        learning_rate = self.get_hyperparameter(
            "learning_rate",
            parameter
        )
        second_moment = self.second_moments[parameter_id]
        second_moment_decay = self.get_hyperparameter(
            "second_moment_decay",
            parameter
        )

        first_moment_debiased = (
            first_moment
          / (1 - first_moment_decay ** self.step_id)
        )
        second_moment_debiased = (
            second_moment
          / (1 - second_moment_decay ** self.step_id)
        )        

        parameter_update -= (
            learning_rate
          * first_moment_debiased
          / (
                second_moment_debiased.sqrt()
              + epsilon
            )
        )

        return parameter_update


    def _update_state(
        self,
        parameter: torch.nn.Parameter,
        parameter_id: int
    ):
        first_moment = self.first_moments[parameter_id]
        first_moment_decay = self.get_hyperparameter(
            "first_moment_decay",
            parameter
        )
        second_moment = self.second_moments[parameter_id]
        second_moment_decay = self.get_hyperparameter(
            "second_moment_decay",
            parameter
        )

        first_moment[:] = (
            first_moment_decay
          * first_moment
          + (1 - first_moment_decay)
          * parameter.grad
        )
        second_moment[:] = (
            second_moment_decay
          * second_moment
          + (1 - second_moment_decay)
          * parameter.grad.square()
        )

