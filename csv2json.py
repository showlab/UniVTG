import pdb
import csv
import json

file_name = 'oidv6-class-descriptions'

# the path you download the csv.
with open(f'./teacher/{file_name}.csv', 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    header = next(reader)
    data = [row for row in reader]

class_list = [x[1] for x in data]

with open('f./teacher/{file_name}.json', 'w') as jsonfile:
    json.dump(class_list, jsonfile)