"""
Schelling coordination game with false belief:
Two spies, Alice and Bob, claim to want to meet.
Bob wants to meet Alice, but Alice actually wants to avoid Bob.

They must choose between two locations without communicating
by recursively reasoning about one another.

Taken from: http://forestdb.org/models/schelling-falsebelief.html
"""
import argparse
import torch
from search_inference import HashingMarginal, Search
import pyro
import pyro.poutine as poutine
from pyro.distributions import Bernoulli

def location(preference):
    if False:
        while True:
            i = 10
    '\n    Flips a weighted coin to decide between two locations to meet.\n    In this example, we assume that Alice and Bob share a prior preference\n    for one location over another, reflected in the value of preference below.\n    '
    return pyro.sample('loc', Bernoulli(preference))

def alice_fb(preference, depth):
    if False:
        return 10
    "\n    Alice's actual decision process:\n    Alice decides where to go by reasoning about Bob's choice\n    and choosing the other location.\n    "
    alice_prior = location(preference)
    with poutine.block():
        bob_marginal = HashingMarginal(Search(bob).run(preference, depth - 1))
    pyro.sample('bob_choice', bob_marginal, obs=alice_prior)
    return 1 - alice_prior

def alice(preference, depth):
    if False:
        return 10
    "\n    Alice decides where to go by reasoning about Bob's choice\n    "
    alice_prior = location(preference)
    with poutine.block():
        bob_marginal = HashingMarginal(Search(bob).run(preference, depth - 1))
    return pyro.sample('bob_choice', bob_marginal, obs=alice_prior)

def bob(preference, depth):
    if False:
        for i in range(10):
            print('nop')
    "\n    Bob decides where to go by reasoning about Alice's choice\n    "
    bob_prior = location(preference)
    if depth > 0:
        with poutine.block():
            alice_marginal = HashingMarginal(Search(alice).run(preference, depth))
        return pyro.sample('alice_choice', alice_marginal, obs=bob_prior)
    else:
        return bob_prior

def main(args):
    if False:
        print('Hello World!')
    shared_preference = torch.tensor([args.preference])
    alice_depth = args.depth
    num_samples = args.num_samples
    alice_decision = HashingMarginal(Search(alice_fb).run(shared_preference, alice_depth))
    alice_prob = sum([alice_decision() for i in range(num_samples)]) / float(num_samples)
    print('Empirical frequency of Alice choosing their favored location ' + 'given preference {} and recursion depth {}: {}'.format(shared_preference, alice_depth, alice_prob))
if __name__ == '__main__':
    assert pyro.__version__.startswith('1.8.6')
    parser = argparse.ArgumentParser(description='parse args')
    parser.add_argument('-n', '--num-samples', default=10, type=int)
    parser.add_argument('--depth', default=3, type=int)
    parser.add_argument('--preference', default=0.55, type=float)
    args = parser.parse_args()
    main(args)