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
path = 'runs/detect/exp21/labels/'
# label_txt = 'pill_dataset/dataset/test12/label.txt'
# txt_outfile = open(label_txt, "w")
DetectCount = 0
AccuracyCount = 0
ErrorCount = 0
LossCount = 212
names = ['1000','10070','10223','10228','1022','10238','10244','10245','10246','10249','10255','10258','10260','10261','10277','1071','1076','1096','1133','1233','12439','12448','12558','12590','12591','175','2033','2040','2065','2086','2090','2101','2119','2121','2130','2133','2145','2147','2148','2150','2158','2159','2247','2249','226','2301','2305','231','2321','2323','2326','2331','2366','2368','2370','2378','2387','2389','2397','2398','2399','2400','2408','2412','27','280','281','325','326','328','380','381','394','395','4049','4061','4076','4084','571','6036','6081','6120','6124','626','6307','630','6317','6319','6326','635','6361','6362','6389','6501','712','721','725','748','756','800','818','8320','8406','87','910','930']  # class names

for file in os.listdir(path):
    DetectCount += 1
    fname = file.split('_')[0]
#     key = pills.index(fname)
    txt_file = open(path+file,"r")
    lines = txt_file.read().split('\n')
    DetectPillNumber = int(lines[0].split(' ',1)[0])

    # print(fname,names[DetectPillNumber])
    if(fname == names[DetectPillNumber]):
        AccuracyCount += 1
    else:
        ErrorCount += 1

LossCount -= DetectCount
print(DetectCount,LossCount,AccuracyCount,ErrorCount)
print("precision:{} , LossCount:{}".format(str(round((AccuracyCount/DetectCount)*100, 3))+"%",str(LossCount)) )
#     # print(file,lines[0].split(' ',1)[0])
#     print("pill name：" + fname + "  代號" ,DetectPillNumber , pills[DetectPillNumber])