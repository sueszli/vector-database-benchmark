def find_sums(numbers: list, target: int) -> list:
    if False:
        for i in range(10):
            print('nop')

    def backtracking(start: int, target: int, current_combination: list) -> None:
        if False:
            while True:
                i = 10
        if target == 0:
            combinations.append(current_combination[:])
            return
        for i in range(start, len(numbers)):
            if i > start and numbers[i] == numbers[i - 1]:
                continue
            if numbers[i] > target:
                break
            current_combination.append(numbers[i])
            backtracking(i + 1, target - numbers[i], current_combination)
            current_combination.pop()
    combinations = []
    backtracking(0, target, [])
    return combinations

def main():
    if False:
        while True:
            i = 10
    numbers = [1, 1, 2, 1, 1, 1, 2]
    target = 6
    combinations = find_sums(numbers, target)
    if combinations:
        print(f'Combinaciones que suman {target}:')
        for combo in combinations:
            print(combo)
    else:
        print('No se encontraron combinaciones que sumen el objetivo.')
if __name__ == '__main__':
    main()