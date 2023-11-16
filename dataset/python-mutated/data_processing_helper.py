import csv

def preprocess_station_data(input_filename: str, output_filename: str) -> None:
    if False:
        return 10
    out = open(output_filename, 'w')
    with open(input_filename) as f:
        data = f.readlines()
        rows = []
        '\n        Variable\tColumns\tType\tExample\n        ID\t1-11\tCharacter\tEI000003980\n        LATITUDE\t13-20\tReal\t55.3717\n        LONGITUDE\t22-30\tReal\t-7.3400\n        ELEVATION\t32-37\tReal\t21.0\n        STATE\t39-40\tCharacter\n        NAME\t42-71\tCharacter\tMALIN HEAD\n        GSN FLAG\t73-75\tCharacter\tGSN\n        HCN/CRN FLAG\t77-79\tCharacter\n        WMO ID\t81-85\tCharacter\t03980\n        '
        for row in data:
            row = [row[0:12].strip(), row[12:21].strip(), row[21:31].strip(), row[31:38].strip(), row[38:41].strip(), row[41:72].strip()]
            rows.append(row)
    with out:
        write = csv.writer(out)
        write.writerows(rows)
if __name__ == '__main__':
    preprocess_station_data('ghcnd-stations.txt', 'ghcn-stations-processed.csv')