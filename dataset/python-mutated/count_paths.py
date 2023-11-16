def count_paths(m, n):
    if False:
        i = 10
        return i + 15
    if m < 1 or n < 1:
        return -1
    count = [[None for j in range(n)] for i in range(m)]
    for i in range(n):
        count[0][i] = 1
    for j in range(m):
        count[j][0] = 1
    for i in range(1, m):
        for j in range(1, n):
            count[i][j] = count[i - 1][j] + count[i][j - 1]
    print(count[m - 1][n - 1])

def main():
    if False:
        for i in range(10):
            print('nop')
    (m, n) = map(int, input('Enter two positive integers: ').split())
    count_paths(m, n)
if __name__ == '__main__':
    main()