#!/usr/bin/env python
"""
Summary
-------
Provide a subclass of random.Random using mrg32k3a as the generator
with stream/substream/subsubstream support.

Listing
-------
MRG32k3a : class
advance_stream : method
advance_substream : method
advance_subsubstream : method
reset_stream : method
reset_substream : method
reset_subsubstream : method
start_fixed_s_ss_sss : method
"""

# Code largely adopted from PyMOSO repository (https://github.com/pymoso/PyMOSO).

import numpy as np
import random
from math import log, ceil, sqrt, exp
from copy import deepcopy

from .matmodops import mat33_mat31_mult, mat33_mat33_mult, mat31_mod, mat33_mod, mat33_mat33_mod, mat33_power_mod

# Constants used in mrg32k3a and in substream generation.
# P. L'Ecuyer, ``Good Parameter Sets for Combined Multiple Recursive Random Number Generators'',
# Operations Research, 47, 1 (1999), 159--164.
# P. L'Ecuyer, R. Simard, E. J. Chen, and W. D. Kelton,
# ``An Objected-Oriented Random-Number Package with Many Long Streams and Substreams'',
# Operations Research, 50, 6 (2002), 1073--1075.

mrgnorm = 2.328306549295727688e-10
mrgm1 = 4294967087
mrgm2 = 4294944443
mrga12 = 1403580
mrga13n = 810728
mrga21 = 527612
mrga23n = 1370589

A1p0 = [[0, 1, 0],
        [0, 0, 1],
        [-mrga13n, mrga12, 0]
        ]

A2p0 = [[0, 1, 0],
        [0, 0, 1],
        [-mrga23n, 0, mrga21]
        ]

# A1p47 = mat33_power_mod(A1p0, 2**47, mrgm1).
A1p47 = [[1362557480, 3230022138, 4278720212],
         [3427386258, 3848976950, 3230022138],
         [2109817045, 2441486578, 3848976950]
         ]

# A2p47 = mat33_power_mod(A2p0, 2**47, mrgm2).
A2p47 = [[2920112852, 1965329198, 1177141043],
         [2135250851, 2920112852, 969184056],
         [296035385, 2135250851, 4267827987]
         ]

# A1p94 = mat33_power_mod(A1p0, 2**94, mrgm1).
A1p94 = [[2873769531, 2081104178, 596284397],
         [4153800443, 1261269623, 2081104178],
         [3967600061, 1830023157, 1261269623]
         ]

# A2p94 = mat33_power_mod(A2p0, 2**94, mrgm2).
A2p94 = [[1347291439, 2050427676, 736113023],
         [4102191254, 1347291439, 878627148],
         [1293500383, 4102191254, 745646810]
         ]

# A1p141 = mat33_power_mod(A1p0, 2**141, mrgm1).
A1p141 = [[3230096243, 2131723358, 3262178024],
          [2882890127, 4088518247, 2131723358],
          [3991553306, 1282224087, 4088518247]
          ]

# A2p141 = mat33_power_mod(A2p0, 2**141, mrgm2).
A2p141 = [[2196438580, 805386227, 4266375092],
          [4124675351, 2196438580, 2527961345],
          [94452540, 4124675351, 2825656399]
          ]

# Constants used in Beasley-Springer-Moro algorithm for approximating
# the inverse cdf of the standard normal distribution.
bsma = [2.50662823884, -18.61500062529, 41.39119773534, -25.44106049637]
bsmb = [-8.47351093090, 23.08336743743, -21.06224101826, 3.13082909833]
bsmc = [0.3374754822726147, 0.9761690190917186, 0.1607979714918209, 0.0276438810333863, 0.0038405729373609,
        0.0003951896511919, 0.0000321767881768, 0.0000002888167364, 0.0000003960315187]


# Adapted to pure Python from the P. L'Ecuyer code referenced above.
def mrg32k3a(state):
    """Generate a random number between 0 and 1 from a given state.

    Parameters
    ----------
    state : 'tuple' ['int']
        current state of the generator.

    Returns
    -------
    new_state : 'tuple' ['int']
        next state of the generator.
    u : 'float'
        pseudo uniform random variate.
    """
    # Component 1.
    p1 = mrga12 * state[1] - mrga13n * state[0]
    k1 = int(p1 / mrgm1)
    p1 -= k1 * mrgm1
    if p1 < 0.0:
        p1 += mrgm1
    # Component 2.
    p2 = mrga21 * state[5] - mrga23n * state[3]
    k2 = int(p2 / mrgm2)
    p2 -= k2 * mrgm2
    if p2 < 0.0:
        p2 += mrgm2
    # Combination.
    if p1 <= p2:
        u = (p1 - p2 + mrgm1) * mrgnorm
    else:
        u = (p1 - p2) * mrgnorm
    new_state = (state[1], state[2], int(p1), state[4], state[5], int(p2))
    return new_state, u


def bsm(u):
    """Approximate a quantile of the standard normal distribution via
    the Beasley-Springer-Moro algorithm.

    Parameters
    ----------
    u : 'float'
        probability value for the desired quantile (between 0 and 1).

    Returns
    -------
    z : 'float'
    """
    y = u - 0.5
    if abs(y) < 0.42:
        # Approximate from the center (Beasly-Springer 1977).
        r = pow(y, 2)
        r2 = pow(r, 2)
        r3 = pow(r, 3)
        r4 = pow(r, 4)
        asum = sum([bsma[0], bsma[1] * r, bsma[2] * r2, bsma[3] * r3])
        bsum = sum([1, bsmb[0] * r, bsmb[1] * r2, bsmb[2] * r3, bsmb[3] * r4])
        z = y * (asum / bsum)
    else:
        # Approximate from the tails (Moro 1995).
        if y < 0.0:
            signum = -1
            r = u
        else:
            signum = 1
            r = 1 - u
        s = log(-log(r))
        s0 = pow(s, 2)
        s1 = pow(s, 3)
        s2 = pow(s, 4)
        s3 = pow(s, 5)
        s4 = pow(s, 6)
        s5 = pow(s, 7)
        s6 = pow(s, 8)
        clst = [bsmc[0], bsmc[1] * s, bsmc[2] * s0, bsmc[3] * s1, bsmc[4] * s2, bsmc[5] * s3, bsmc[6] * s4,
                bsmc[7] * s5, bsmc[8] * s6]
        t = sum(clst)
        z = signum * t
    return z


class MRG32k3a(random.Random):
    """Implements mrg32k3a as the generator for a random.Random object.

    Attributes
    ----------
    _current_state : 'tuple' ['int']
        current state of mrg32k3a generator.
    ref_seed : 'tuple' ['int']
        seed from which to start the generator
        streams/substreams/subsubstreams are referenced w.r.t. ref_seed.
    s_ss_sss_index : 'list' ['int']
        triplet of the indices of the current stream-substream-subsubstream.
    stream_start : 'list' ['int']
        state corresponding to the start of the current stream.
    substream_start: 'list' ['int']
        state corresponding to the start of the current substream.
    subsubstream_start: 'list' ['int']
        state corresponding to the start of the current subsubstream.

    Parameters
    ----------
    ref_seed : 'tuple' ['int'], optional
        seed from which to start the generator.
    s_ss_sss_index : 'list' ['int']
        triplet of the indices of the stream-substream-subsubstream to start at.

    See also
    --------
    random.Random
    """

    def __init__(self, ref_seed=(12345, 12345, 12345, 12345, 12345, 12345), s_ss_sss_index=None):
        assert (len(ref_seed) == 6)
        self.version = 2
        self.generate = mrg32k3a
        self.ref_seed = ref_seed
        super().__init__(ref_seed)
        if s_ss_sss_index is None:
            s_ss_sss_index = [0, 0, 0]
        self.start_fixed_s_ss_sss(s_ss_sss_index)

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result

    def seed(self, new_state):
        """Set the state (or seed) of the generator and update the generator state.

        Parameters
        ----------
        new_state : 'tuple' ['int']
            new state to which to advance the generator.
        """
        assert (len(new_state) == 6)
        self._current_state = new_state
        # super().seed(new_state)

    def getstate(self):
        """Return the state of the generator.

        Returns
        -------
        _current_state : 'tuple' ['int']
            current state of the generator.
        random.Random.getstate() : 'tuple' ['int']
            Random.getstate output.

        See also
        --------
        random.Random
        """
        return self.get_current_state(), super().getstate()

    def setstate(self, state):
        """Set the internal state of the generator.

        Parameters
        ----------
        state : 'tuple'
            ''state[0]'' is new state for the generator.
            ''state[1]'' is random.Random.getstate().

        See also
        --------
        random.Random
        """
        self.seed(state[0])
        super().setstate(state[1])

    def random(self):
        """Generate a standard uniform variate and advance the generator
        state.

        Returns
        -------
        u : 'float'
            pseudo uniform random variate.
        """
        state = self._current_state
        new_state, u = self.generate(state)
        self.seed(new_state)
        return u

    def get_current_state(self):
        """Return the current state of the generator.

        Returns
        -------
        _current_state : 'tuple' ['int']
            current state of the generator.
        """
        return self._current_state

    def normalvariate(self, mu=0, sigma=1):
        """Generate a normal random variate.

        Parameters
        ----------
        mu : 'float'
            expected value of the normal distribution from which to
            generate.
        sigma : 'float'
            standard deviation of the normal distribution from which to
            generate.

        Returns
        -------
        'float'
            a normal random variate from the specified distribution.
        """
        u = self.random()
        z = bsm(u)
        return mu + sigma * z

    def lognormalvariate(self, lq=10, uq=200):
        """Generate a Lognormal random variate using 2.5% and 97.5% quantiles

        Parameters
        ----------
        lq : 'float'
            2.5% quantile of the lognormal distribution from which to
            generate.
        uq : 'float'
            97.5% quantile of the lognormal distribution from which to
            generate.

        Returns
        -------
        'float'
            a lognormal random variate from the specified distribution.
        """
        mu = (log(lq) + log(uq)) / 2
        sigma = (log(uq) - mu) / 1.96
        x = self.normalvariate(mu, sigma)
        return exp(x)

    def mvnormalvariate(self, mean_vec, cov, factorized=True):
        """Generate a normal random vector.

        Parameters
        ---------
        mean_vec : 'array'
            location parameters of the multivariate normal distribution
            from which to generate.
        cov : 'array'
            covariance matrix of the multivariate normal distribution
            from which to generate.
        factorized : 'bool'
            False : need to calculate chol based on covariance
            True : do not need to calculate chol since we already have it.
        Returns
        -------
        'list' ['float']
            a normal random multivariate from the specified distribution.
        """
        n_cols = len(cov)
        if not factorized:
            Chol = np.linalg.cholesky(cov)
        else:
            Chol = cov
        observations = [self.normalvariate(0, 1) for _ in range(n_cols)]
        return Chol.dot(observations).transpose() + mean_vec

    def poissonvariate(self, lmbda):
        """Generate a poisson random variate.

        Parameters
        ---------
        lmbda : 'float'
            expected value of the poisson distribution from which to
            generate.

        Returns
        -------
        'float'
            a poisson random variate from the specified distribution.
        """
        if lmbda < 35:
            n = 0
            p = self.random()
            threshold = exp(-lmbda)
            while p >= threshold:
                u = self.random()
                p = p * u
                n = n + 1
        else:
            z = self.normalvariate()
            n = max(ceil(lmbda + sqrt(lmbda) * z - 0.5), 0)
        return n

    def gumbelvariate(self, mu, beta):
        """Generate a gumbel random variate.

        Parameters
        ---------
        mu : 'float'
            location of the mode of the gumbel distribution from which to
            generate.
        beta : 'float'
            scale parameter of the gumbel distribution from which to
            generate.

        Returns
        -------
        'float'
            a gumbel random variate from the specified distribution.
        """
        u = self.random()
        q = mu - beta * np.log(-np.log(u))
        return q

    def advance_stream(self):
        """Advance the state of the generator to the start of the next stream.
        Streams are of length 2**141.
        """
        state = self.stream_start
        # Split the state into 2 components of length 3.
        st1 = state[0:3]
        st2 = state[3:6]
        # Efficiently advance state -> A*s % m for both state parts.
        nst1m = mat33_mat31_mult(A1p141, st1)
        nst2m = mat33_mat31_mult(A2p141, st2)
        nst1 = mat31_mod(nst1m, mrgm1)
        nst2 = mat31_mod(nst2m, mrgm2)
        nstate = tuple(nst1 + nst2)
        self.seed(nstate)
        # Increment the stream index.
        self.s_ss_sss_index[0] += 1
        # Reset index for substream and subsubstream.
        self.s_ss_sss_index[1] = 0
        self.s_ss_sss_index[2] = 0
        # Update state referencing.
        self.stream_start = nstate
        self.substream_start = nstate
        self.subsubstream_start = nstate

    def advance_substream(self):
        """Advance the state of the generator to the start of the next substream.
        Substreams are of length 2**94.
        """
        state = self.substream_start
        # Split the state into 2 components of length 3.
        st1 = state[0:3]
        st2 = state[3:6]
        # Efficiently advance state -> A*s % m for both state parts.
        nst1m = mat33_mat31_mult(A1p94, st1)
        nst2m = mat33_mat31_mult(A2p94, st2)
        nst1 = mat31_mod(nst1m, mrgm1)
        nst2 = mat31_mod(nst2m, mrgm2)
        nstate = tuple(nst1 + nst2)
        self.seed(nstate)
        # Increment the substream index.
        self.s_ss_sss_index[1] += 1
        # Reset index for subsubstream.
        self.s_ss_sss_index[2] = 0
        # Update state referencing.
        self.substream_start = nstate
        self.subsubstream_start = nstate

    def advance_subsubstream(self):
        """Advance the state of the generator to the start of the next subsubstream.
        Subsubstreams are of length 2**47.
        """
        state = self.subsubstream_start
        # Split the state into 2 components of length 3.
        st1 = state[0:3]
        st2 = state[3:6]
        # Efficiently advance state -> A*s % m for both state parts.
        nst1m = mat33_mat31_mult(A1p47, st1)
        nst2m = mat33_mat31_mult(A2p47, st2)
        nst1 = mat31_mod(nst1m, mrgm1)
        nst2 = mat31_mod(nst2m, mrgm2)
        nstate = tuple(nst1 + nst2)
        self.seed(nstate)
        # Increment the subsubstream index.
        self.s_ss_sss_index[2] += 1
        # Update state referencing.
        self.subsubstream_start = nstate

    def reset_stream(self):
        """Reset the state of the generator to the start of the current stream.
        """
        nstate = self.stream_start
        self.seed(nstate)
        # Update state referencing.
        self.substream_start = nstate
        self.subsubstream_start = nstate
        # Reset index for substream and subsubstream.
        self.s_ss_sss_index[1] = 0
        self.s_ss_sss_index[2] = 0

    def reset_substream(self):
        """Reset the state of the generator to the start of the current substream.
        """
        nstate = self.substream_start
        self.seed(nstate)
        # Update state referencing.
        self.subsubstream_start = nstate
        # Reset index for subsubstream.
        self.s_ss_sss_index[2] = 0

    def reset_subsubstream(self):
        """Reset the state of the generator to the start of the current subsubstream.
        """
        nstate = self.subsubstream_start
        self.seed(nstate)

    def start_fixed_s_ss_sss(self, s_ss_sss_triplet):
        """Set the rng to the start of a specified (stream, substream, subsubstream) triplet.

        Parameters
        ----------
        s_ss_sss_triplet : 'list' ['int']
            triplet of the indices of the current stream-substream-subsubstream.
        """
        state = self.ref_seed
        # Split the reference seed into 2 components of length 3.
        st1 = state[0:3]
        st2 = state[3:6]
        # Advance to start of specified stream.
        # Efficiently advance state -> A*s % m for both state parts.
        nst1m = mat33_mat31_mult(mat33_power_mod(A1p141, s_ss_sss_triplet[0], mrgm1), st1)
        nst2m = mat33_mat31_mult(mat33_power_mod(A2p141, s_ss_sss_triplet[0], mrgm2), st2)
        st1 = mat31_mod(nst1m, mrgm1)
        st2 = mat31_mod(nst2m, mrgm2)
        self.stream_start = tuple(st1 + st2)
        # Advance to start of specified substream.
        # Efficiently advance state -> A*s % m for both state parts.
        nst1m = mat33_mat31_mult(mat33_power_mod(A1p94, s_ss_sss_triplet[1], mrgm1), st1)
        nst2m = mat33_mat31_mult(mat33_power_mod(A2p94, s_ss_sss_triplet[1], mrgm2), st2)
        st1 = mat31_mod(nst1m, mrgm1)
        st2 = mat31_mod(nst2m, mrgm2)
        self.substream_start = tuple(st1 + st2)
        # Advance to start of specified subsubstream.
        # Efficiently advance state -> A*s % m for both state parts.
        nst1m = mat33_mat31_mult(mat33_power_mod(A1p47, s_ss_sss_triplet[2], mrgm1), st1)
        nst2m = mat33_mat31_mult(mat33_power_mod(A2p47, s_ss_sss_triplet[2], mrgm2), st2)
        st1 = mat31_mod(nst1m, mrgm1)
        st2 = mat31_mod(nst2m, mrgm2)
        self.subsubstream_start = tuple(st1 + st2)
        nstate = tuple(st1 + st2)
        self.seed(nstate)
        # Update index referencing.
        self.s_ss_sss_index = s_ss_sss_triplet