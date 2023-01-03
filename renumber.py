import csv
import os

pills = []
pills_name = []
numbers = []
with open('label.csv', newline='') as csvfile:
    rows = csv.reader(csvfile)
    for row in rows:
        pills.append(row[1])
        pills_name.append(row[2])
        numbers.append(row[3])
i = 0
path = 'runs/detect/exp17/labels/'
# label_txt = 'pill_dataset/dataset/test12/label.txt'
# txt_outfile = open(label_txt, "w")

for file in os.listdir(path):
    fname = file.split('_')[0]
    asdasdas = file.split('_')[-1]
    key = pills.index(fname)
    txt_file = open(path+file,"r")
    lines = txt_file.read().split('\n')
    pill_calculate = lines[0].split(' ',1)[-1]
    # NewTxt_file = open(path+file,"w")
    new_line = str(numbers[key])+ ' ' + pill_calculate
    txt_file = open(path+file,"w")
    txt_file.write(new_line)
    # if(asdasdas == '9.txt'):
    #     i += 1
    #     txt_outfile.write(pills[key] + '\n')
    print(file, pill_calculate)
    # exit()