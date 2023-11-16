(start, end, height) = (0, 1, 2)

class Solution(object):

    def getSkyline(self, buildings):
        if False:
            while True:
                i = 10
        intervals = self.ComputeSkylineInInterval(buildings, 0, len(buildings))
        res = []
        last_end = -1
        for interval in intervals:
            if last_end != -1 and last_end < interval[start]:
                res.append([last_end, 0])
            res.append([interval[start], interval[height]])
            last_end = interval[end]
        if last_end != -1:
            res.append([last_end, 0])
        return res

    def ComputeSkylineInInterval(self, buildings, left_endpoint, right_endpoint):
        if False:
            print('Hello World!')
        if right_endpoint - left_endpoint <= 1:
            return buildings[left_endpoint:right_endpoint]
        mid = left_endpoint + (right_endpoint - left_endpoint) / 2
        left_skyline = self.ComputeSkylineInInterval(buildings, left_endpoint, mid)
        right_skyline = self.ComputeSkylineInInterval(buildings, mid, right_endpoint)
        return self.MergeSkylines(left_skyline, right_skyline)

    def MergeSkylines(self, left_skyline, right_skyline):
        if False:
            i = 10
            return i + 15
        (i, j) = (0, 0)
        merged = []
        while i < len(left_skyline) and j < len(right_skyline):
            if left_skyline[i][end] < right_skyline[j][start]:
                merged.append(left_skyline[i])
                i += 1
            elif right_skyline[j][end] < left_skyline[i][start]:
                merged.append(right_skyline[j])
                j += 1
            elif left_skyline[i][start] <= right_skyline[j][start]:
                (i, j) = self.MergeIntersectSkylines(merged, left_skyline[i], i, right_skyline[j], j)
            else:
                (j, i) = self.MergeIntersectSkylines(merged, right_skyline[j], j, left_skyline[i], i)
        merged += left_skyline[i:]
        merged += right_skyline[j:]
        return merged

    def MergeIntersectSkylines(self, merged, a, a_idx, b, b_idx):
        if False:
            print('Hello World!')
        if a[end] <= b[end]:
            if a[height] > b[height]:
                if b[end] != a[end]:
                    b[start] = a[end]
                    merged.append(a)
                    a_idx += 1
                else:
                    b_idx += 1
            elif a[height] == b[height]:
                b[start] = a[start]
                a_idx += 1
            else:
                if a[start] != b[start]:
                    merged.append([a[start], b[start], a[height]])
                a_idx += 1
        elif a[height] >= b[height]:
            b_idx += 1
        else:
            if a[start] != b[start]:
                merged.append([a[start], b[start], a[height]])
            a[start] = b[end]
            merged.append(b)
            b_idx += 1
        return (a_idx, b_idx)