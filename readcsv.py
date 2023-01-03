import csv

pills = []
with open('./working/label.csv', newline='') as csvfile:
    rows = csv.reader(csvfile)
    for row in rows:
        pills.append(row[1])

print(pills)
print(len(pills))