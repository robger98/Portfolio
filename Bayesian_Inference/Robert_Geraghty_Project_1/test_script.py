import os
output = "graphs/test2.csv"
sizes = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,17,20]
samps = [10,50,100,500,1000,5000,10000,50000,100000]
with open(output, 'a+') as out:
    out.write("DAG Tests - size\n")

for size in sizes:
    os.system('python gen-bn dag '+ str(size))
    with open(output, 'a+') as out:
        out.write(str(size)+','+str(1000)+',')
    os.system('python inference.py -n 1000 -o graphs/' + str(1000) +'-'+str(size)+'-DAG.png >> ' + output)

with open(output, 'a+') as out:
    out.write("DAG Tests - samples\n")
os.system('python gen-bn dag '+ str(15))
for samp in samps:
    with open(output, 'a+') as out:
        out.write(str(15)+','+str(samp)+',')
    os.system('python inference.py -n '+ str(samp) + ' -o graphs/' + str(samp) +'-'+str(size)+'-DAG.png  >> ' + output)

with open(output, 'a+') as out:
    out.write("Poly Tests - size\n")
for size in sizes:
    os.system('python gen-bn polytree '+ str(size))
    with open(output, 'a+') as out:
        out.write(str(size)+','+str(1000)+',')
    os.system('python inference.py -n 1000  -o graphs/' + str(1000) +'-'+str(size)+'-Poly.png >> ' + output)
with open(output, 'a+') as out:
    out.write("Poly Tests - samples\n")
os.system('python gen-bn polytree '+ str(15))
for samp in samps:
    with open(output, 'a+') as out:
        out.write(str(15)+','+str(samp)+',')
    os.system('python inference.py -n '+ str(samp) + ' -o graphs/' + str(samp) +'-'+str(size)+'-Poly.png >> ' + output)