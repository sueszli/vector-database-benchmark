class GossipManager(object):
    """ Keeps gossip and rankings that should be sent to other nodes or collected by Ranking """

    def __init__(self):
        if False:
            return 10
        ' Create new gossip keeper instance '
        self.gossips = []
        self.peers_that_stopped_gossiping = set()
        self.neighbour_loc_rank_buff = []

    def add_gossip(self, gossip):
        if False:
            return 10
        ' Add newly heard gossip to the gossip list\n        :param list gossip: list of gossips from one peer\n        '
        self.gossips.append(gossip)

    def pop_gossips(self):
        if False:
            print('Hello World!')
        ' Return all gathered gossips and clear gossip buffer\n        :return list: list of all gossips\n        '
        gossip = self.gossips
        self.gossips = []
        return gossip

    def register_that_peer_stopped_gossiping(self, id_):
        if False:
            while True:
                i = 10
        ' Register that holds peer ids that has stopped gossiping\n        :param str id_: id of a string that has stopped gossiping\n        '
        self.peers_that_stopped_gossiping.add(id_)

    def pop_peers_that_stopped_gossiping(self):
        if False:
            i = 10
            return i + 15
        " Return set of all peers that has stopped gossiping\n        :return set: set of peers id's\n        "
        stop = self.peers_that_stopped_gossiping
        self.peers_that_stopped_gossiping = set()
        return stop

    def add_neighbour_loc_rank(self, neigh_id, about_id, rank):
        if False:
            for i in range(10):
                print('nop')
        '\n        Add local rank from neighbour to the collection\n        :param str neigh_id: id of a neighbour - opinion giver\n        :param str about_id: opinion is about a node with this id\n        :param list rank: opinion that node <neigh_id> have about node <about_id>\n        :return:\n        '
        self.neighbour_loc_rank_buff.append([neigh_id, about_id, rank])

    def pop_neighbour_loc_ranks(self):
        if False:
            for i in range(10):
                print('nop')
        ' Return all local ranks that was collected in that round and clear the rank list\n        :return list: list of all neighbours local rank sent to this node\n        '
        nlr = self.neighbour_loc_rank_buff
        self.neighbour_loc_rank_buff = []
        return nlr