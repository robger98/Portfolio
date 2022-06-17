# Generates simple dummy data 

import sys
sys.stdout= open("gen_data_out.txt", 'w')
def mofn(m, n):
    perms = []
    for i in range((2 ** n)):
        # print(i)
        form = "{"+str(n)+":b}"
        ex_str = format(i, "0"+str(n)+"b")
        # print(ex)
        ex = [int(d) for d in ex_str]
        ex_str_list = [d for d in ex_str]
        # print(ex_str_list)
        str_rep = ','.join(ex_str_list)
        if sum(ex) == m:
            str_rep = str(i) + ',1,' + str_rep
        else:
            str_rep = str(i) + ',0,' +  str_rep
        perms.append(str_rep)
    return(perms)

strs = mofn(11, 20)
for s in strs:
    print(s)
