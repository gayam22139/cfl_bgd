import torch
from torch.optim.optimizer import Optimizer


class BGD(Optimizer):
    """Implements BGD.
    A simple usage of BGD would be:
    for samples, labels in batches:
        for mc_iter in range(mc_iters):
            optimizer.randomize_weights()
            output = model.forward(samples)
            loss = cirterion(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.aggregate_grads()
        optimizer.step()
    """
    def __init__(self, params, std_init, mean_eta=1, mc_iters=10):
        """
        Initialization of BGD optimizer
        group["mean_param"] is the learned mean.
        group["std_param"] is the learned STD.
        :param params: List of model parameters
        :param std_init: Initialization value for STD parameter
        :param mean_eta: Eta value
        :param mc_iters: Number of Monte Carlo iteration. Used for correctness check.
                         Use None to disable the check.
        """
        super(BGD, self).__init__(params, defaults={})
        assert mc_iters is None or (type(mc_iters) == int and mc_iters > 0), "mc_iters should be positive int or None."
        self.std_init = std_init
        self.mean_eta = mean_eta
        self.mc_iters = mc_iters
        # Initialize mu (mean_param) and sigma (std_param)
        # breakpoint()
        for group in self.param_groups:
            assert len(group["params"]) == 1, "BGD optimizer does not support multiple params in a group"
            # group['params'][0] is the weights
            assert isinstance(group["params"][0], torch.Tensor), "BGD expect param to be a tensor"
            # We use the initialization of weights to initialize the mean.
            group["mean_param"] = group["params"][0].data.clone()
            #group["std_param"] = torch.zeros_like(group["params"][0].data).add_(self.std_init)

            '''The above line has been replaced by the below line - Shiva Bhai Recommendation'''
            group["std_param"] = torch.full_like(group["params"][0].data,self.std_init)

        self._init_accumulators()

    def get_mc_iters(self):
        return self.mc_iters

    def _init_accumulators(self):
        self.mc_iters_taken = 0
        for group in self.param_groups:
            group["eps"] = None
            group["grad_mul_eps_sum"] = torch.zeros_like(group["params"][0].data)
            group["grad_sum"] = torch.zeros_like(group["params"][0].data)

    def randomize_weights(self, force_std=-1):
        """
        Randomize the weights according to N(mean, std).
        :param force_std: If force_std>=0 then force_std is used for STD instead of the learned STD.
        :return: None
        """
        for group in self.param_groups:
            mean = group["mean_param"] # theta recieved from the server
            std = group["std_param"] # initialized   
            if force_std >= 0:
                std = std.mul(0).add(force_std)
            group["eps"] = torch.normal(torch.zeros_like(mean), 1)
            # Reparameterization trick (here we set the weights to their randomized value):
            group["params"][0].data.copy_(mean.add(std.mul(group["eps"])))

            '''The above line(Reparamterization) implements this equation -> θi = μi + εiσi'''

    def aggregate_grads(self, batch_size):
        """
        Aggregates a single Monte Carlo iteration gradients. Used in step() for the expectations calculations.
        optimizer.zero_grad() should be used before calling .backward() once again.
        :param batch_size: BGD is using non-normalized gradients, but PyTorch gives normalized gradients.
                            Therefore, we multiply the gradients by the batch size.
        :return: None
        """
        self.mc_iters_taken += 1
        groups_cnt = 0
        for group in self.param_groups:
            if group["params"][0].grad is None:
                continue
            assert group["eps"] is not None, "Must randomize weights before using aggregate_grads"
            groups_cnt += 1
            grad = group["params"][0].grad.data.mul(batch_size)
            group["grad_sum"].add_(grad)
             #The above grad_sum is used to estimate the expectation of gradient which inturn is used in updating μi
            group["grad_mul_eps_sum"].add_(grad.mul(group["eps"]))
            #The above grad_mul_eps_sum is used to estimate the expectation of gradient multiplied by epsilon which inturn is used in updating σi
            group["eps"] = None
        assert groups_cnt > 0, "Called aggregate_grads, but all gradients were None. Make sure you called .backward()"

    def step(self, closure=None):
        """
        Updates the learned mean and STD.
        :return:
        """
        # Makes sure that self.mc_iters had been taken.
        assert self.mc_iters is None or self.mc_iters == self.mc_iters_taken, "MC iters is set to " \
                                                                              + str(self.mc_iters) \
                                                                              + ", but took " + \
                                                                              str(self.mc_iters_taken) + " MC iters"
        # need global mu, sigma
        for group in self.param_groups:
            mean = group["mean_param"] # mu k n-1
            std = group["std_param"] # sigma k n-1 
            # Divide gradients by MC iters to get expectation
            e_grad = group["grad_sum"].div(self.mc_iters_taken)
            e_grad_eps = group["grad_mul_eps_sum"].div(self.mc_iters_taken)
            # Update mean and STD params

            g_mean = 1
            g_std = 1
            alpha_mg = 0.5
            var = std.pow(2)
            g_var = g_std.pow(2) 

            denominator = alpha_mg*(var).add( (1 - alpha_mg)*(g_var) )
            mean = (alpha_mg*(var.mul(g_mean))).add( (1-alpha_mg)*( g_var.mul(mean)) ).sub( (var.mul(g_var).mul(e_grad)))
            mean = mean.div(denominator)

            sqrt_term =  torch.mul( (g_var.mul(var)) , torch.sqrt( (alpha_mg*var).add((1-alpha_mg)*g_var).add( torch.pow(((0.5*g_std.mul(std)).mul(e_grad)), 2) ) ) )
            std = sqrt_term.sub(0.5*g_var.mul(var).mul(e_grad_eps))
            std = std.div(denominator)



            # mean.add_(-std.pow(2).mul(e_grad).mul(self.mean_eta))
            # sqrt_term = torch.sqrt(e_grad_eps.mul(std).div(2).pow(2).add(1)).mul(std)
            # std.copy_(sqrt_term.add(-e_grad_eps.mul(std.pow(2)).div(2)))
        self.randomize_weights(force_std=0)
        self._init_accumulators()