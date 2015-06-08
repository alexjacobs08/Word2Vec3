import os
import codecs
import csv

import glob
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
    print("Import Error: cElementTree failed")


class MySentences(object):
     def __init__(self, dirname):
         self.dirname = dirname

     def __iter__(self):
         for fname in os.listdir(self.dirname):
             for line in open(os.path.join(self.dirname, fname)):
                 yield line.split()
#for file in `ls *.xml`; do {  sed 's/&nbsp;//g' $file > $file.new; }; done;

newData = []


path = '../../../Documents/blogsNew'
fileList = os.listdir(path)

for infile in fileList:

    #print(infile)
    fileName = str(infile)
    try:
        blogId, gender, age, job, zodiac, xml, new = infile.split('.')
    except ValueError:
        print("fuckoff")
        continue

    if gender == "male":
        gender = 1
    elif gender == "female":
        gender = 0

    with codecs.open(os.path.join(path,infile),'r',encoding='utf-8', errors='ignore') as f:
        lines = (line.rstrip() for line in f)
        lines = list(line for line in lines if line)

        #lines = f.readlines()

        if not lines:
            break
        line_iter = iter(lines)


        for line in line_iter:

            if "<post>" in line:
                lf = next(line_iter)
                newData.append([blogId, gender, lf])


    f.close()


with open("output.tsv", "w") as b:
    writer = csv.writer(b, delimiter='\t')
    writer.writerows(newData)