from collections import defaultdict
from itertools import chain

class SimpleCampaign(object):

    def __init__(self, name, target_names, impressions):
        if False:
            while True:
                i = 10
        self.name = name
        self.target_names = target_names
        self.impressions = impressions

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        s = '<%s %s: %s impressions in %s>'
        return s % (self.__class__.__name__, self.name, self.impressions, ', '.join(self.target_names))

class SimpleTarget(object):

    def __init__(self, name, impressions):
        if False:
            for i in range(10):
                print('nop')
        self.name = name
        self.impressions = impressions

    def __repr__(self):
        if False:
            print('Hello World!')
        return '<%s %s: %s impressions>' % (self.__class__.__name__, self.name, self.impressions)

class System(object):
    """Take a set of campaigns and a set of targets and allocate the
    inventory of each target to the campaigns in such a way to maximize
    the free inventory in the priority target or targets.

    """

    def __init__(self, campaigns, targets, priority_target_names):
        if False:
            i = 10
            return i + 15
        self.priority_target_names = priority_target_names
        (self.campaigns, self.targets) = self.simplify(campaigns, targets)

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        max_names = ', '.join(self.priority_target_names)
        all_names = ', '.join(('%s (%s)' % (target.name, target.impressions) for target in self.targets))
        return '<%s: max %s in %s>' % (self.__class__.__name__, max_names, all_names)

    def combine_campaigns(self, campaigns):
        if False:
            return 10
        'Combine campaigns with the same target.'
        campaigns_by_target = defaultdict(list)
        for campaign in campaigns:
            target_names_tuple = tuple(sorted(campaign.target_names))
            campaigns_by_target[target_names_tuple].append(campaign)
        combined_campaigns = []
        changed = False
        for (target_names_tuple, campaigns) in campaigns_by_target.iteritems():
            if len(campaigns) > 1:
                changed = True
                name = ','.join((camp.name for camp in campaigns))
                target_names = list(target_names_tuple)
                impressions = sum((camp.impressions for camp in campaigns))
                combined = SimpleCampaign(name, target_names, impressions)
                combined_campaigns.append(combined)
            else:
                combined_campaigns.extend(campaigns)
        return (changed, combined_campaigns)

    def reduce_campaigns(self, campaigns, targets):
        if False:
            print('Hello World!')
        'Remove campaigns.\n\n        Find campaigns with only a single target and subtract their required\n        impressions from the target and remove the campaign.\n\n        '
        targets_by_name = {target.name: target for target in targets}
        changed = False
        reduced_campaigns = []
        for campaign in campaigns:
            if len(campaign.target_names) == 1:
                changed = True
                target_name = campaign.target_names[0]
                target_impressions = targets_by_name[target_name].impressions
                target_impressions -= campaign.impressions
                new_target = SimpleTarget(target_name, target_impressions)
                targets_by_name[target_name] = new_target
            else:
                reduced_campaigns.append(campaign)
        reduced_targets = targets_by_name.values()
        return (changed, reduced_campaigns, reduced_targets)

    def reduce_targets(self, campaigns, targets):
        if False:
            i = 10
            return i + 15
        "Remove targets.\n\n        Remove non-priority targets that have only a single campaign or that\n        have enough inventory to satisfy all their campaigns. As a result may\n        end up removing campaigns if they're fully satisfied.\n\n        "
        campaign_names_by_target = defaultdict(list)
        for campaign in campaigns:
            for target_name in campaign.target_names:
                if target_name not in self.priority_target_names:
                    campaign_names_by_target[target_name].append(campaign.name)
        campaigns_by_name = {campaign.name: campaign for campaign in campaigns}
        targets_by_name = {target.name: target for target in targets}
        changed = False
        for (target_name, campaign_names) in campaign_names_by_target.iteritems():
            target = targets_by_name[target_name]
            campaign_impressions = sum((campaigns_by_name[name].impressions for name in campaign_names))
            fully_satisfied = campaign_impressions <= target.impressions
            single_campaign = len(campaign_names) == 1
            if not (fully_satisfied or single_campaign):
                continue
            changed = True
            for campaign_name in campaign_names:
                campaign = campaigns_by_name[campaign_name]
                if fully_satisfied:
                    impressions = 0
                else:
                    target_impressions = max(0, target.impressions)
                    impressions = campaign_impressions - target_impressions
                target_names = campaign.target_names[:]
                target_names.remove(target_name)
                campaigns_by_name[campaign_name] = SimpleCampaign(campaign_name, target_names, impressions)
            del targets_by_name[target_name]
        reduced_campaigns = []
        for campaign in campaigns_by_name.itervalues():
            if campaign.impressions > 0:
                reduced_campaigns.append(campaign)
        reduced_targets = targets_by_name.values()
        return (changed, reduced_campaigns, reduced_targets)

    def simplify(self, campaigns, targets):
        if False:
            return 10
        changed = False
        first_run = True
        while changed or first_run:
            first_run = False
            (changed_1, campaigns) = self.combine_campaigns(campaigns)
            (changed_2, campaigns, targets) = self.reduce_campaigns(campaigns, targets)
            (changed_3, campaigns, targets) = self.reduce_targets(campaigns, targets)
            changed = changed_2 or changed_3
        return (campaigns, targets)

    def get_free_impressions(self):
        if False:
            for i in range(10):
                print('nop')
        "Run through algorithm to solve for maximum free impressions.\n\n        Choose how to allocate inventory to each campaign by first mapping out\n        the distance of each target from the targets we're trying to maximize\n        inventory of, and then assigning inventory to each campaign\n        preferring to choose the targets that are farthest away.\n\n        "
        campaigns_by_target = defaultdict(list)
        for campaign in self.campaigns:
            for target_name in campaign.target_names:
                campaigns_by_target[target_name].append(campaign)
        level = 0
        level_by_target_name = {}
        next_level_target_names = set(self.priority_target_names)
        while next_level_target_names:
            target_names = next_level_target_names
            for target_name in target_names:
                level_by_target_name[target_name] = level
            campaigns = chain.from_iterable((campaigns_by_target[target_name] for target_name in target_names))
            next_level_target_names = {target_name for campaign in campaigns for target_name in campaign.target_names if target_name not in level_by_target_name}
            level += 1
        for target in self.targets:
            if target.name not in level_by_target_name:
                level_by_target_name[target.name] = level
        target_names_by_level = defaultdict(list)
        for (target_name, level) in level_by_target_name.iteritems():
            target_names_by_level[level].append(target_name)
        unassigned_by_campaign = {campaign.name: campaign.impressions for campaign in self.campaigns}
        impressions_by_target = {target.name: target.impressions for target in self.targets}
        for level in sorted(target_names_by_level.iterkeys(), reverse=True):
            target_names = target_names_by_level[level]
            campaigns = chain.from_iterable((campaigns_by_target[target_name] for target_name in target_names))

            def sort_val(campaign):
                if False:
                    for i in range(10):
                        print('nop')
                val = sum((level_by_target_name[name] + 1 for name in campaign.target_names if level_by_target_name[name] <= level))
                return val
            for campaign in sorted(campaigns, key=sort_val):
                campaign_targets = [name for name in target_names if name in campaign.target_names]
                for target_name in campaign_targets:
                    unassigned = unassigned_by_campaign[campaign.name]
                    if unassigned > 0:
                        available = max(0, impressions_by_target[target_name])
                        assigned = min(unassigned, available)
                        unassigned_by_campaign[campaign.name] -= assigned
                        impressions_by_target[target_name] -= assigned
        penalty = 0
        for campaign in self.campaigns:
            unassigned = unassigned_by_campaign[campaign.name]
            if unassigned > 0:
                if not campaign.target_names:
                    continue
                target_name = min(campaign.target_names, key=lambda name: level_by_target_name[name])
                unassigned_by_campaign[campaign.name] -= unassigned
                impressions_by_target[target_name] -= unassigned
                penalty += unassigned
        free_impressions = sum((impressions_by_target[target_name] for target_name in self.priority_target_names))
        return free_impressions - penalty

def campaign_to_simple_campaign(campaign):
    if False:
        while True:
            i = 10
    name = campaign._fullname
    target_names = campaign.target.subreddit_names
    impressions = campaign.impressions / campaign.ndays
    return SimpleCampaign(name, target_names, impressions)

def get_maximized_pageviews(priority_sr_names, booked_by_target, pageviews_by_sr_name):
    if False:
        print('Hello World!')
    targets = [SimpleTarget(sr_name, pageviews) for (sr_name, pageviews) in pageviews_by_sr_name.iteritems()]
    campaigns = [SimpleCampaign(', '.join(sr_names), list(sr_names), impressions) for (sr_names, impressions) in booked_by_target.iteritems()]
    system = System(campaigns, targets, priority_sr_names)
    return system.get_free_impressions()

def run_tests():
    if False:
        for i in range(10):
            print('nop')
    pageviews_by_sr_name = {'leagueoflegends': 50000, 'dota2': 50000, 'hearthstone': 50000, 'games': 50000}
    targets = [SimpleTarget(sr_name, pageviews) for (sr_name, pageviews) in pageviews_by_sr_name.iteritems()]
    campaigns = [SimpleCampaign('c1', ['leagueoflegends'], 20000), SimpleCampaign('c2', ['dota2'], 40000), SimpleCampaign('c3', ['games'], 40000), SimpleCampaign('c4', ['hearthstone'], 40000), SimpleCampaign('c5', ['leagueoflegends', 'dota2', 'hearthstone', 'games'], 20000)]
    priority_target_names = ['leagueoflegends']
    system = System(campaigns, targets, priority_target_names)
    impressions = system.get_free_impressions()
    assert impressions == 30000
    priority_target_names = ['leagueoflegends', 'dota2', 'hearthstone', 'games']
    system = System(campaigns, targets, priority_target_names)
    impressions = system.get_free_impressions()
    assert impressions == 40000
    pageviews_by_sr_name = {'leagueoflegends': 25000, 'dota2': 25000, 'hearthstone': 25000, 'games': 25000, 'smashbros': 50000}
    targets = [SimpleTarget(sr_name, pageviews) for (sr_name, pageviews) in pageviews_by_sr_name.iteritems()]
    campaigns = [SimpleCampaign('c1', ['leagueoflegends', 'dota2'], 25000), SimpleCampaign('c2', ['hearthstone', 'games'], 25000), SimpleCampaign('c3', ['dota2', 'smashbros'], 50000), SimpleCampaign('c4', ['games', 'smashbros'], 50000)]
    priority_target_names = ['leagueoflegends', 'hearthstone']
    "\n    optimal distribution:\n    c4: 25000 from smashbros, 25000 from games\n    c3: 25000 from smashbros, 25000 from dota2\n    c2: 25000 from hearthstone\n    c2: 25000 from leagueoflegends\n\n    Current algorithm can't split smashbros because it's too greedy, the first\n    of c4 or c3 to be allocated will get all 50000\n\n    Subsequent improvements to the algorithm should allow splitting a target\n    and should prioritize campaigns for which the target is their lowest\n    level target. Also for campaigns for which the target is their highest\n    level target the algorithm should look forward to their lowest level target\n    and determin whether that has any chance of satisfying the campaign.\n\n    "
    system = System(campaigns, targets, priority_target_names)
    impressions = system.get_free_impressions()
    assert impressions == 0