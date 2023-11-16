class Solution(object):

    def totalMoney(self, n):
        if False:
            print('Hello World!')
        '\n        :type n: int\n        :rtype: int\n        '

        def arithmetic_sequence_sum(a, d, n):
            if False:
                print('Hello World!')
            return (2 * a + (n - 1) * d) * n // 2
        (cost, day) = (1, 7)
        first_week_cost = arithmetic_sequence_sum(cost, cost, day)
        (week, remain_day) = divmod(n, day)
        return arithmetic_sequence_sum(first_week_cost, cost * day, week) + arithmetic_sequence_sum(cost * (week + 1), cost, remain_day)