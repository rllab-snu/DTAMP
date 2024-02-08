import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions.normal import Normal


class NormalizedNormal(Normal):
    def __init__(self, loc, scale):
        super().__init__(loc, scale)
        self._normalized_mean = F.normalize(loc, p=2.0, dim=-1)

    def rsample(self, sample_shape=torch.Size()):
        x = super().rsample(sample_shape)
        return F.normalize(x, p=2.0, dim=-1)

    def sample(self, sample_shape=torch.Size()):
        x = super().sample(sample_shape)
        return F.normalize(x, p=2.0, dim=-1)

    @property
    def mean(self):
        return self._normalized_mean

    def detach(self):
        return NormalizedNormal(self.loc.detach(), self.scale.detach())


class SquashedNormal(Normal):
    def __init__(self, loc, scale):
        super().__init__(loc, scale)
        self._squashed_mean = torch.tanh(loc)

    def rsample(self, sample_shape=torch.Size()):
        x = super().rsample(sample_shape)
        return torch.tanh(x)

    def sample(self, sample_shape=torch.Size()):
        x = super().sample(sample_shape)
        return torch.tanh(x)

    @property
    def mean(self):
        return self._squashed_mean

    def detach(self):
        return SquashedNormal(self.loc.detach(), self.scale.detach())


def to_one_hot(tensor, n, fill_with=1.):
    # we perform one hot encore with respect to the last axis
    one_hot = torch.FloatTensor(tensor.size() + (n,)).zero_()
    if tensor.is_cuda:
        # one_hot = one_hot.cuda()
        one_hot = one_hot.to(tensor.device)
    one_hot.scatter_(len(tensor.size()), tensor.unsqueeze(-1), fill_with)
    return one_hot


class DiscreteMixLogistic(torch.distributions.Distribution):
    def __init__(self, mean, log_scale, logit_probs, num_classes=10, log_scale_min=-7.0, validate_args=False):
        assert mean.device == log_scale.device and mean.device == logit_probs.device, "all tensors must lie on same device!"
        batch_shape = log_scale.shape[:-1]
        event_shape = mean.shape[len(batch_shape) + 1:]
        super().__init__(batch_shape, event_shape, validate_args)
        self._mean = mean
        self._log_scale = log_scale
        self._logit_probs = logit_probs
        self._num_classes = num_classes
        self._log_scale_min = log_scale_min

    def log_prob(self, value):
        # reshape value to match convention
        B, n_mix = value.shape[:-1], self._log_scale.shape[-1]
        # unpack parameters. (B, T, num_mixtures) x 3
        logit_probs = self._logit_probs.reshape((*B, -1, n_mix))
        means = self._mean.reshape((*B, -1, n_mix))
        log_scales = torch.clamp(self._log_scale.reshape((*B, -1, n_mix)),
                                 min=self._log_scale_min)

        # B x T x 1 -> B x T x num_mixtures
        y = value.reshape((*B, -1, 1))

        centered_y = y - means
        inv_stdv = torch.exp(-log_scales)
        plus_in = inv_stdv * (centered_y + 1. / (self._num_classes - 1))
        cdf_plus = torch.sigmoid(plus_in)
        min_in = inv_stdv * (centered_y - 1. / (self._num_classes - 1))
        cdf_min = torch.sigmoid(min_in)

        # log probability for edge case of 0 (before scaling)
        # equivalent: torch.log(torch.sigmoid(plus_in))
        log_cdf_plus = plus_in - F.softplus(plus_in)

        # log probability for edge case of 255 (before scaling)
        # equivalent: (1 - torch.sigmoid(min_in)).log()
        log_one_minus_cdf_min = -F.softplus(min_in)

        # probability for all other cases
        cdf_delta = cdf_plus - cdf_min

        mid_in = inv_stdv * centered_y
        # log probability in the center of the bin, to be used in extreme cases
        # (not actually used in our code)
        log_pdf_mid = mid_in - log_scales - 2. * F.softplus(mid_in)

        # tf equivalent
        """
        log_probs = tf.where(x < -0.999, log_cdf_plus,
                            tf.where(x > 0.999, log_one_minus_cdf_min,
                                    tf.where(cdf_delta > 1e-5,
                                            tf.log(tf.maximum(cdf_delta, 1e-12)),
                                            log_pdf_mid - np.log(127.5))))
        """
        # TODO: cdf_delta <= 1e-5 actually can happen. How can we choose the value
        # for num_classes=65536 case? 1e-7? not sure..
        inner_inner_cond = (cdf_delta > 1e-5).float()

        inner_inner_out = inner_inner_cond * \
                          torch.log(torch.clamp(cdf_delta, min=1e-12)) + \
                          (1. - inner_inner_cond) * (log_pdf_mid - np.log((self._num_classes - 1) / 2))
        inner_cond = (y > 0.999).float()
        inner_out = inner_cond * log_one_minus_cdf_min + (1. - inner_cond) * inner_inner_out
        cond = (y < -0.999).float()
        log_probs = cond * log_cdf_plus + (1. - cond) * inner_out

        log_probs = log_probs + F.log_softmax(logit_probs, -1)
        return torch.logsumexp(log_probs, dim=-1).reshape(value.shape)

    def sample(self):
        B, n_mix = self._log_scale.shape[:-2], self._log_scale.shape[-1]
        logit_probs = self._logit_probs.reshape((*B, -1, n_mix))

        # sample mixture indicator from softmax
        temp = logit_probs.data.new(logit_probs.size()).uniform_(1e-5, 1.0 - 1e-5)
        temp = logit_probs.data - torch.log(- torch.log(temp))
        _, argmax = temp.max(dim=-1)

        # (B, T) -> (B, T, nr_mix)
        one_hot = to_one_hot(argmax, n_mix)
        # select logistic parameters
        means = self._mean.reshape((*B, -1, n_mix))
        means = torch.sum(means * one_hot, dim=-1)
        log_scales = self._log_scale.reshape((*B, -1, n_mix))
        log_scales = torch.sum(log_scales * one_hot, dim=-1)

        # sample from logistic & clip to interval
        # we don't actually round to the nearest 8bit value when sampling
        u = means.data.new(means.size()).uniform_(1e-5, 1.0 - 1e-5)
        x = means + torch.exp(log_scales) * (torch.log(u) - torch.log(1. - u))

        x = torch.clamp(torch.clamp(x, min=-1.), max=1.)

        return x.reshape(self._mean.shape[:-1])

    @property
    def mean(self):
        alphas = F.softmax(self._logit_probs, dim=-1)
        return torch.sum(self._mean * alphas, -1)
