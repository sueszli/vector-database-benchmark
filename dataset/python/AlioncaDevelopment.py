def tennis_game(points: list):
    scores = {0: "Love", 1: "15", 2: "30", 3: "40"}
    p1_score = 0
    p2_score = 0
    for point in points:
        if point == "P1":
            p1_score += 1
        elif point == "P2":
            p2_score += 1
        if p1_score < 4 and p2_score < 4:
            print(scores[p1_score] + " - " + scores[p2_score])
        elif p1_score == p2_score:
            print("Deuce")
        elif p1_score > p2_score:
            if p1_score - p2_score == 1:
                print("Ventaja P1")
            else:
                print("Ha ganado el P1")
                return
        else:
            if p2_score - p1_score == 1:
                print("Ventaja P2")
            else:
                print("Ha ganado el P2")
                return

# Example usage
points = ["P1", "P1", "P2", "P2", "P1", "P2", "P1", "P1"]
tennis_game(points)
