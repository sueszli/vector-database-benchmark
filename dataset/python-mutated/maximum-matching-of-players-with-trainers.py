class Solution(object):

    def matchPlayersAndTrainers(self, players, trainers):
        if False:
            while True:
                i = 10
        '\n        :type players: List[int]\n        :type trainers: List[int]\n        :rtype: int\n        '
        players.sort()
        trainers.sort()
        result = 0
        for x in trainers:
            if players[result] > x:
                continue
            result += 1
            if result == len(players):
                break
        return result