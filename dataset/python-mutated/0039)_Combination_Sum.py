class Solution:

    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        if False:
            for i in range(10):
                print('nop')
        result = []
        candidates = sorted(candidates)

        def dfs(remain, stack):
            if False:
                print('Hello World!')
            if remain == 0:
                result.append(stack)
                return
            for item in candidates:
                if item > remain:
                    break
                if stack and item < stack[-1]:
                    continue
                else:
                    dfs(remain - item, stack + [item])
        dfs(target, [])
        return result