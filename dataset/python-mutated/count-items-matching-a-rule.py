class Solution(object):

    def countMatches(self, items, ruleKey, ruleValue):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type items: List[List[str]]\n        :type ruleKey: str\n        :type ruleValue: str\n        :rtype: int\n        '
        rule = {'type': 0, 'color': 1, 'name': 2}
        return sum((item[rule[ruleKey]] == ruleValue for item in items))