def get_map(rec, hid):
    user_songs = []

    # build the dict to index songs
    with open("kaggle_songs.txt", "r") as f:
        its = dict(map(lambda line: list((line.strip().split(' ')[1],line.strip().split(' ')[0])), f.readlines()))
    f.close()


    # Read rec songs for each user
    rec = "MSD_result.txt"
    with open(rec) as f:
        for line in f:
            song_set = line.strip().split(" ")
            for i in range(len(song_set)):
                song_set[i] = its[song_set[i]]
            user_songs.append(song_set)


    # Read hidden songs for each users from file
    hid = "Result_hidden.txt"
    user_index = -1
    user_hidden = []

    f = open(hid)
    user_list = []

    for line in f:
        user, song,_, = line.split("\t")
        if user not in user_list:
            user_list.append(user)
            user_hidden.append([])
            user_index += 1

        user_hidden[user_index].append(song)

    # Calculate map values
    sum = 0.0

    for i in range(0, len(user_hidden)):
        value = 0.0
        for hidden_song in range(0, len(user_hidden[i])):
            for rec_song in range(0, len(user_songs[i])):
                if user_hidden[i][hidden_song] == user_songs[i][rec_song]:
                    value +=  (hidden_song + 1.0) / (rec_song + 1.0)

        value /= len(user_hidden[i])
        sum += value

    sum /= len(user_hidden)
    return sum

rec = ""
hid = ""

print get_map(rec, hid)

