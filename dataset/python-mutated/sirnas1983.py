import math

def triples_pitagoricos(num_dado):
    if False:
        i = 10
        return i + 15
    resp = []
    for i in range(1, num_dado):
        for j in range(1, i):
            b = math.pow(math.pow(i, 2) + math.pow(j, 2), 0.5)
            if b > num_dado:
                break
            if b // 1 == b:
                parc = (j, i, int(b))
                resp.append(parc)
    return resp
print(triples_pitagoricos(50))