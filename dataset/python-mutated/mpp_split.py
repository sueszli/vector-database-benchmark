import random
import math
from typing import List, Tuple, Dict, NamedTuple
from collections import defaultdict
from .lnutil import NoPathFound
PART_PENALTY = 1.0
MIN_PART_SIZE_MSAT = 10000000
EXHAUST_DECAY_FRACTION = 10
RELATIVE_SPLIT_SPREAD = 0.3
CANDIDATES_PER_LEVEL = 20
MAX_PARTS = 5
ChannelsFundsInfo = Dict[Tuple[bytes, bytes], int]

class SplitConfig(dict, Dict[Tuple[bytes, bytes], List[int]]):
    """maps a channel (channel_id, node_id) to a list of amounts"""

    def number_parts(self) -> int:
        if False:
            i = 10
            return i + 15
        return sum([len(v) for v in self.values() if sum(v)])

    def number_nonzero_channels(self) -> int:
        if False:
            return 10
        return len([v for v in self.values() if sum(v)])

    def number_nonzero_nodes(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        return len({nodeid for ((_, nodeid), amounts) in self.items() if sum(amounts)})

    def total_config_amount(self) -> int:
        if False:
            i = 10
            return i + 15
        return sum([sum(c) for c in self.values()])

    def is_any_amount_smaller_than_min_part_size(self) -> bool:
        if False:
            i = 10
            return i + 15
        smaller = False
        for amounts in self.values():
            if any([amount < MIN_PART_SIZE_MSAT for amount in amounts]):
                smaller |= True
        return smaller

class SplitConfigRating(NamedTuple):
    config: SplitConfig
    rating: float

def split_amount_normal(total_amount: int, num_parts: int) -> List[int]:
    if False:
        while True:
            i = 10
    'Splits an amount into about `num_parts` parts, where the parts are split\n    randomly (normally distributed around amount/num_parts with certain spread).'
    parts = []
    avg_amount = total_amount / num_parts
    while total_amount - sum(parts) > avg_amount:
        amount_to_add = int(abs(random.gauss(avg_amount, RELATIVE_SPLIT_SPREAD * avg_amount)))
        if sum(parts) + amount_to_add < total_amount:
            parts.append(amount_to_add)
    parts.append(total_amount - sum(parts))
    return parts

def remove_duplicates(configs: List[SplitConfig]) -> List[SplitConfig]:
    if False:
        for i in range(10):
            print('nop')
    unique_configs = set()
    for config in configs:
        config_sorted_values = {k: sorted(v) for (k, v) in config.items()}
        config_sorted_keys = {k: config_sorted_values[k] for k in sorted(config_sorted_values.keys())}
        hashable_config = tuple(((c, tuple(sorted(config[c]))) for c in config_sorted_keys))
        unique_configs.add(hashable_config)
    unique_configs = [SplitConfig({c[0]: list(c[1]) for c in config}) for config in unique_configs]
    return unique_configs

def remove_multiple_nodes(configs: List[SplitConfig]) -> List[SplitConfig]:
    if False:
        for i in range(10):
            print('nop')
    return [config for config in configs if config.number_nonzero_nodes() == 1]

def remove_single_part_configs(configs: List[SplitConfig]) -> List[SplitConfig]:
    if False:
        return 10
    return [config for config in configs if config.number_parts() != 1]

def remove_single_channel_splits(configs: List[SplitConfig]) -> List[SplitConfig]:
    if False:
        i = 10
        return i + 15
    filtered = []
    for config in configs:
        for v in config.values():
            if len(v) > 1:
                continue
            filtered.append(config)
    return filtered

def rate_config(config: SplitConfig, channels_with_funds: ChannelsFundsInfo) -> float:
    if False:
        print('Hello World!')
    'Defines an objective function to rate a configuration.\n\n    We calculate the normalized L2 norm for a configuration and\n    add a part penalty for each nonzero amount. The consequence is that\n    amounts that are equally distributed and have less parts are rated\n    lowest (best). A penalty depending on the total amount sent over a channel\n    counteracts channel exhaustion.'
    rating = 0
    total_amount = config.total_config_amount()
    for (channel, amounts) in config.items():
        funds = channels_with_funds[channel]
        if amounts:
            for amount in amounts:
                rating += amount * amount / (total_amount * total_amount)
                rating += PART_PENALTY * PART_PENALTY
            decay = funds / EXHAUST_DECAY_FRACTION
            rating += math.exp((sum(amounts) - funds) / decay)
    return rating

def suggest_splits(amount_msat: int, channels_with_funds: ChannelsFundsInfo, exclude_single_part_payments=False, exclude_multinode_payments=False, exclude_single_channel_splits=False) -> List[SplitConfigRating]:
    if False:
        print('Hello World!')
    'Breaks amount_msat into smaller pieces and distributes them over the\n    channels according to the funds they can send.\n\n    Individual channels may be assigned multiple parts. The split configurations\n    are returned in sorted order, from best to worst rating.\n\n    Single part payments can be excluded, since they represent legacy payments.\n    Split configurations that send via multiple nodes can be excluded as well.\n    '
    configs = []
    channels_order = list(channels_with_funds.keys())
    for _ in range(CANDIDATES_PER_LEVEL):
        for target_parts in range(1, MAX_PARTS):
            config = SplitConfig()
            split_amounts = split_amount_normal(amount_msat, target_parts)
            for amount in split_amounts:
                random.shuffle(channels_order)
                for c in channels_order:
                    if c not in config:
                        config[c] = []
                    if sum(config[c]) + amount <= channels_with_funds[c]:
                        config[c].append(amount)
                        break
                else:
                    distribute_amount = amount
                    for c in channels_order:
                        funds_left = channels_with_funds[c] - sum(config[c])
                        add_amount = min(funds_left, distribute_amount)
                        config[c].append(add_amount)
                        distribute_amount -= add_amount
                        if distribute_amount == 0:
                            break
            if config.total_config_amount() != amount_msat:
                raise NoPathFound('Cannot distribute payment over channels.')
            if target_parts > 1 and config.is_any_amount_smaller_than_min_part_size():
                continue
            assert config.total_config_amount() == amount_msat
            configs.append(config)
    configs = remove_duplicates(configs)
    if exclude_multinode_payments:
        configs = remove_multiple_nodes(configs)
    if exclude_single_part_payments:
        configs = remove_single_part_configs(configs)
    if exclude_single_channel_splits:
        configs = remove_single_channel_splits(configs)
    rated_configs = [SplitConfigRating(config=c, rating=rate_config(c, channels_with_funds)) for c in configs]
    rated_configs.sort(key=lambda x: x.rating)
    return rated_configs