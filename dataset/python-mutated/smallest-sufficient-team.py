class Solution(object):

    def smallestSufficientTeam(self, req_skills, people):
        if False:
            i = 10
            return i + 15
        '\n        :type req_skills: List[str]\n        :type people: List[List[str]]\n        :rtype: List[int]\n        '
        lookup = {v: i for (i, v) in enumerate(req_skills)}
        dp = {0: []}
        for (i, p) in enumerate(people):
            his_skill_set = 0
            for skill in p:
                if skill in lookup:
                    his_skill_set |= 1 << lookup[skill]
            for (skill_set, people) in dp.items():
                with_him = skill_set | his_skill_set
                if with_him == skill_set:
                    continue
                if with_him not in dp or len(dp[with_him]) > len(people) + 1:
                    dp[with_him] = people + [i]
        return dp[(1 << len(req_skills)) - 1]