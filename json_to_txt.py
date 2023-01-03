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

print(pills)
print(pills_name)
print(numbers)

from os import walk, getcwd
from PIL import Image


def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


"""-------------------------------------------------------------------"""

""" Configure Paths"""
mypath = "./dataset/"
outpath = "./result/"
json_backup = "./json_backup/"

wd = getcwd()
# list_file = open('%s_list.txt'%(wd), 'w')

""" Get input json file list """
json_name_list = []
for file in os.listdir(mypath):
    if file.endswith(".json"):
        json_name_list.append(file)

""" Process """
for json_name in json_name_list:
    txt_name = json_name.rstrip(".json") + ".txt"
    """ Open input text files """
    txt_path = mypath + json_name
    print("Input:" + txt_path)
    txt_file = open(txt_path, "r")

    """ Open output text files """
    txt_outpath = outpath + txt_name
    print("Output:" + txt_outpath)
    txt_outfile = open(txt_outpath, "a")

    """ Convert the data to YOLO format """
    lines = txt_file.read().split('\r\n')  # for ubuntu, use "\r\n" instead of "\n"
    for idx, line in enumerate(lines):
        if ("lineColor" in line):
            break  # skip reading after find lineColor
        if ("label" in line):
            x1 = float(lines[idx + 5].rstrip(','))
            y1 = float(lines[idx + 6])
            x2 = float(lines[idx + 9].rstrip(','))
            y2 = float(lines[idx + 10])
            cls = line[16:17]

            # in case when labelling, points are not in the right order
            xmin = min(x1, x2)
            xmax = max(x1, x2)
            ymin = min(y1, y2)
            ymax = max(y1, y2)
            img_path = str('%s/dataset/%s.jpg' % (wd, os.path.splitext(json_name)[0]))

            im = Image.open(img_path)
            w = int(im.size[0])
            h = int(im.size[1])

            print(w, h)
            print(xmin, xmax, ymin, ymax)
            b = (xmin, xmax, ymin, ymax)
            bb = convert((w, h), b)
            print(bb)
            txt_outfile.write(cls + " " + " ".join([str(a) for a in bb]) + '\n')

    os.rename(txt_path, json_backup + json_name)  # move json file to backup folder

# i = 0
# path = 'runs/detect/exp17/labels/'
# # label_txt = 'pill_dataset/dataset/test12/label.txt'
# # txt_outfile = open(label_txt, "w")
#
# for file in os.listdir(path):
#     fname = file.split('_')[0]
#     asdasdas = file.split('_')[-1]
#     key = pills.index(fname)
#     txt_file = open(path+file,"r")
#     lines = txt_file.read().split('\n')
#     pill_calculate = lines[0].split(' ',1)[-1]
#     # NewTxt_file = open(path+file,"w")
#     new_line = str(numbers[key])+ ' ' + pill_calculate
#     txt_file = open(path+file,"w")
#     txt_file.write(new_line)
#     # if(asdasdas == '9.txt'):
#     #     i += 1
#     #     txt_outfile.write(pills[key] + '\n')
#     print(file, pill_calculate)
#     # exit()