#!/usr/bin/python
import numpy as np

def random_factor_model(
                 n_instrs=500,
                 n_group=10,
                 n_style=10,
                 prop_group=0.3,
                 prop_style=0.1,
                 prop_market=0.1,
                 prob_conglomerate=0.1):
    """generate a simple new factor model. There are three types of
    factors: a market factor, n_group "industry group" factors, and
    n_style "style factors". The market factor has the same loadings
    for everything. The industry group factors are sparse: most
    intruments have one loading, though some are "conglomerates" and
    may have two or more (the probability of this is given by
    prob_conglomerate). The style factors have random normally
    distributed entries. prop_group controls the proportion of
    variance from industry groups, prop_style from style factors, and
    prop_market from the market factor. The remainder 1 - prop_group -
    prop_market - prop_style is the instrument specific variance.
    """
    assert prop_group + prop_style + prop_market < 1.0
    n_facs = n_group + n_style + 1
    loadings = np.zeros( (n_instrs, n_facs) )
    for i in range(0, n_instrs):
        if np.random.randn() < prob_conglomerate:
            # a small percentage of the universe is conglomerates,
            # that belong to several industries
            cgroups = int(np.random.exponential() + 2)
            l = np.zeros( (n_facs,) )
            for _ in range(0, cgroups):
                l[np.random.randint(n_group)] = np.random.lognormal()
            loadings[i, :] = l * np.sqrt(prop_group / np.sum(l * l))
        else:
            # most of the universe belongs to one industry only
            loadings[i, np.random.randint(n_group)] = np.sqrt(prop_group)
    # these are the style factors: these loadings have nice properties
    for i in range(n_group, n_group + n_style):
        loadings[:, i] = np.random.randn(n_instrs) * np.sqrt(prop_style / n_style)
    loadings[:,n_facs - 1] = np.sqrt(prop_market)
    spec_vol = np.ones( (n_instrs,) ) * np.sqrt(1.0 - prop_group - prop_style - prop_market)
    return ReturnGen(loadings, spec_vol)

class ReturnGen:
    """generate random returns. Instatiate using a loadings matrix and vector
    of specific volatilities. Use gen_samples to generate samples."""
    def __init__(self, loadings, specific_vols):
        self.loadings = loadings
        self.specific_vols = specific_vols
        self.n_instrs = len(specific_vols)
        self.n_factors = loadings.shape[1]
        assert loadings.shape[0] == self.n_instrs, "mismatched loadings!"
    def gen_samples(self, n_samples):
        """generate n_samples random correlated returns.
        The output is an n_samples x n_instruments matrix"""
        return np.random.randn(n_samples, self.n_instrs) * self.specific_vols + np.dot(np.random.randn(n_samples, self.n_factors), self.loadings.T)

def main():
    r = random_factor_model(n_instrs=10, n_group=10, n_style=10, prob_conglomerate=0.3)
    r.gen_samples(500)
    
if __name__ == "__main__":
    main()
