import csv
import itertools
import codecs


def unfusy_reader(csv_reader):
    while True:
        try:
            yield next(csv_reader)
        except csv.Error:
            print "fuck"
            continue






labledData = []
testData = []
unlabledData = []


with open("data/output.tsv",'rb') as f:
    #f = "data/output.tsv"
    #reader = csv.reader(codecs.open(f, 'rU', 'utf-16'))
    #reader = csv.reader(f, delimiter='\t')
    reader = unfusy_reader(csv.reader(f, delimiter='\t'))

    reader = list(reader)
    #row_count = sum(1 for row in f)
    row_count = 287436
    print("data opened")
    print("row count: ",row_count)

    labledLen = 201205
    print labledLen
    x = row_count - labledLen

    #for row in itertools.islice(reader, 0, labledLen):
    for row in reader[0:labledLen]:
        labledData.append(row)
        #print(row)

    #for row in itertools.islice(reader, labledLen+1, labledLen+x/2):
    for row in reader[labledLen+1:labledLen+x/2]:
        testData.append(row)
    #for row in itertools.islice(reader, labledLen+x/2+1,):
    for row in reader[labledLen+x/2+1:]:
        shit = [row[0] + row[2]]
        unlabledData.append(shit)

f.close()

print('writing files')


with open("labledData.tsv", "w") as b:
    writer = csv.writer(b, delimiter='\t')
    writer.writerows(labledData)
b.close()

with open("unlabledData.tsv", "w") as c:
    writer = csv.writer(c, delimiter='\t')
    writer.writerows(unlabledData)
c.close()

with open("testData.tsv", "w") as d:
    writer = csv.writer(d, delimiter='\t')
    writer.writerows(testData)
d.close()