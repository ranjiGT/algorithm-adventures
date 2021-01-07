import sys
import csv
import uuid

#************ReadingInput************
unique_filename = str(uuid.uuid4().hex)
args = sys.argv
file = open(args[2], newline='')
reader = csv.reader(file)
data = [row for row in reader]
learning_rate = float(args[4])
threshold_rate = float(args[6])
initial, gradient_list, weight_list, x_values, y_actual = [0], [], [], [], []

grad_list = {}

for i in range(len(data[0])):
    grad_list[i] = float(0)
    
for i in range(0, len(data[0])):
    gradient_list.append(initial)

weight_list = {}

for i in range(len(data[0])):
    weight_list[i] = float(0)
    
for j in range(0, len(data)):
    val = [1]
    val2, c = [], 0
    for i in data[j]:
        c+=1
        if c == len(data[0]):
            val2.append(float(i))
            break
        val.append(float(i))
    y_actual.append(val2)    
    x_values.append(val)
    
l = len(data[0]) - 1
y_original = {}

for j in range(len(data)):
    y_original[j] = float(data[j][l])
    
iteration = 0
stopping = []

with open(unique_filename+'.csv', 'w', newline = '')as csvfile:
    fieldname = ['iteration_number', 'weight0', 'weight1', 'weight2', 'sum_of_squared_errors']
    thewriter = csv.writer(csvfile, delimiter = ',')

    while True:
        print_list = []
        y_predict = []
        
        for i in range(len(x_values)):
            row = 0
            for j in range(len(x_values[0])):
                row = row + weight_list[j] * x_values[i][j]
            y_predict.append(row)
            
#****************Calculting SSE************************
        error = [y_original[i] - y_predict[i] for i in range(len(data))]
        sse = [i*i for i in error]
      
        print_list.append(iteration)
        for i in range(len(data[0])):
            print_list.append(round(weight_list[i], 4))
            
        print_list.append(round(sum(sse), 4))

        thewriter.writerow(print_list)
        stopping.append(sum(sse))
    
#**************Gradient Computation***********************
        for j in range(len(data[0])):
            gg = []
            
            for i in range(len(data)):
                gg.append(x_values[i][j] * error[i])
                
            grad_list[j] = sum(gg)
            
        for i in range(len(data[0])):
            weight_list[i] = weight_list[i] + (learning_rate*grad_list[i])

#******************StopIteration*******************************              
        if((iteration>0) and (stopping[iteration - 1] - stopping[iteration] < threshold_rate)):
            break
        iteration+=1
        
#*****************PrintToConsole*********************        
with open(unique_filename+'.csv') as f:
    curobj = csv.reader(f)
    for _ in curobj:
        print(', '.join(_))