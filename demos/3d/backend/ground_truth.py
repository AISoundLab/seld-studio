import csv
# opening the CSV file
rows = []
for i in range(300):
    rows.append([i, 'NOTHING', 0, 0, 0])

with open('raw_gt.csv', mode ='r')as file:
   
  # reading the CSV file
  csvFile = csv.reader(file)
  csvList = list(csvFile)[1:]
 
  # displaying the contents of the CSV file
  for i in range(300):
      for line in csvList:
          if line[0] != 'Start' and (float(line[0]) * 10 <= i <= float(line[1]) *10):
              rows[i] = [i, line[2], line[3], line[4], line[5]]

# name of csv file 
filename = "gt.csv"

fields = ['frame', 'class', 'x', 'y', 'z']
    
# writing to csv file 
with open(filename, 'w', newline='') as csvfile: 
    # creating a csv writer object 
    csvwriter = csv.writer(csvfile) 
        
    # writing the fields 
    csvwriter.writerow(fields) 
        
    # writing the data rows 
    csvwriter.writerows(rows)