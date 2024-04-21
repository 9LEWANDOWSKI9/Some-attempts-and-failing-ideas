import numpy as np

expert_mat = np.mat([[3, 7, 1],
                  [5, 9, 1],
                  [1, 6, 8],
                  [9, 4, 1]])

print(expert_mat)
rows, cols = expert_mat.shape

####################################### μmk(xi*xj)类 ###########################################################
# 先求分母
total = 0
col1 = int(input("Enter a number that represents the disease : "))
for i in range(rows):
    for j in range(i+1 , rows):
        m = expert_mat[i, col1]+expert_mat[j, col1]
        n = expert_mat[i, col1] * expert_mat[j, col1]
        m0 = n / m
        total = total + m0

# 再求分子
row1 = int(input("Enter a number that represents the reason1: "))
row2 = int(input("Enter a number that represents the reason2: "))
m1 = expert_mat[row1, col1] + expert_mat[row2, col1]
n1 = expert_mat[row1, col1] * expert_mat[row2, col1]
m00 = n1 / m1

final = m00 / total
print(final)



############################################### μmk(xi*xj*xl)类 ##############################################
# 先求分母
col0 = int(input("Enter a number that represents the disease : "))
total1 = 0
for i1 in range(rows):
    for j1 in range(i1 + 1, rows):
        for l1 in range(i1 + 2, rows):
            m_1 = expert_mat[i1, col0] + expert_mat[j1, col0] + expert_mat[l1, col0]
            n_1 = expert_mat[i1, col0] * expert_mat[j1, col0] * expert_mat[l1, col0]
            m_0 = n_1 / m_1
            total1 = total1 + m_0

# 再求分子
row3 = int(input("Enter a number that represents the reason1: "))
row4 = int(input("Enter a number that represents the reason2: "))
row5 = int(input("Enter a number that represents the reason3: "))
m2 = expert_mat[row3, col0] + expert_mat[row4, col0] + expert_mat[row5, col0]
n2 = expert_mat[row3, col0] * expert_mat[row4, col0] * expert_mat[row5, col0]
m_00 = n2 / m2
final1 = m_00 / total1
print(final1)




############################################ μ(mi*mj)xk类  ###############################################
row_input = int(input("Enter a number that represents the reason :"))
total2 = 0
# 先求分母
for a1 in range(cols):
    for b1 in range(a1+1, cols):
        x = expert_mat[row_input, a1] + expert_mat[row_input, b1]
        y = expert_mat[row_input, a1] * expert_mat[row_input, b1]
        z = y /x
        total2 = total2 + z

# 再求分子
col2 = int(input("Enter a number that represents the disease1: "))
col3 = int(input("Enter a number that represents the disease2: "))
x1 = expert_mat[row_input, col2] + expert_mat[row_input, col3]
y1 = expert_mat[row_input, col2] * expert_mat[row_input, col3]
z1 = y1 / x1

final2 = z1 / total2
print(final2)



############################################ μ(mi*mj*ml)xk类  ###############################################
row_input1 = int(input("Enter a number that represents the reason :"))
total3 = 0
# 先求分母
for a2 in range(cols):
    for b2 in range(a2+1, cols):
        for c2 in range(a2+2, cols):
            x2 = expert_mat[row_input1, a2] + expert_mat[row_input1, b2] + expert_mat[row_input1, c2]
            y2 = expert_mat[row_input1, a2] * expert_mat[row_input1, b2] * expert_mat[row_input1, c2]
            z2 = y2 / x2
            total3 += z2

# 再求分子
col4 = int(input("Enter a number that represents the disease1: "))
col5 = int(input("Enter a number that represents the disease2: "))
col6 = int(input("Enter a number that represents the disease3: "))

x3 = expert_mat[row_input1, col4] + expert_mat[row_input1, col5] + expert_mat[row_input1, col6]
y3 = expert_mat[row_input1, col4] * expert_mat[row_input1, col5] * expert_mat[row_input1, col6]
z3 = y3 / x3

final3 = z3 / total3
print(final3)




######################################## μ(mi*mj)(xk*xl)类 ###############################################
total4 = 0
# 先求分母
col_1 = int(input("Enter a number that represents the disease1: "))
col_2 = int(input("Enter a number that represents the disease2: "))
for a3 in range(rows):
    for b3 in range(a3+1, rows):
        x4 = expert_mat[a3, col_1] + expert_mat[a3, col_2] + expert_mat[b3, col_1] + expert_mat[b3, col_2]
        y4 = expert_mat[a3, col_1] * expert_mat[a3, col_2] * expert_mat[b3, col_1] * expert_mat[b3, col_2]
        z4 = y4 / x4
        total4 += z4

# 再求分子
row_1 = int(input("Enter a number that represents the reason1: "))
row_2 = int(input("Enter a number that represents the reason2: "))
x5 = expert_mat[row_1, col_1] + expert_mat[row_2, col_1] + expert_mat[row_2, col_1] + expert_mat[row_2, col_2]
y5 = expert_mat[row_1, col_1] * expert_mat[row_2, col_1] * expert_mat[row_2, col_1] * expert_mat[row_2, col_2]
z5 = y5 / x5

final4 = z5 / total4
print(final4)


######################################### μ(mi*mj*mk)(xo*xp)类 #############################################
total5 = 0
# 先求分母
col_3 = int(input("Enter a number that represents the disease1: "))
col_4 = int(input("Enter a number that represents the disease2: "))
for a4 in range(rows):
    for b4 in range(a4+1, rows):
        for c4 in range(a4+2, rows):
            x6 = expert_mat[a4, col_3] + expert_mat[a4, col_4] + expert_mat[b4, col_3] + expert_mat[b4, col_4] + expert_mat[c4, col_3] + expert_mat[c4, col_4]
            y6 = expert_mat[a4, col_3] * expert_mat[a4, col_4] * expert_mat[b4, col_3] * expert_mat[b4, col_4] * expert_mat[
                c4, col_3] * expert_mat[c4, col_4]
            z6 = y6 / x6
            total5 += z6

# 再求分子
row_3 = int(input("Enter a number that represents the reason1: "))
row_4 = int(input("Enter a number that represents the reason2: "))
row_5 = int(input("Enter a number that represents the reason3: "))
x7 = expert_mat[row_3, col_3] + expert_mat[row_3, col_4] + expert_mat[row_4, col_3] + expert_mat[row_4, col_4] + expert_mat[row_5, col_3] + expert_mat[row_5, col_4]
y7 = expert_mat[row_3, col_3] * expert_mat[row_3, col_4] * expert_mat[row_4, col_3] * expert_mat[row_4, col_4] * expert_mat[row_5, col_3] * expert_mat[row_5, col_4]
z7 = y7 / x7

final5 = z7 / total5
print(final5)



############################################ μ(mi*mj)(xk*xl*xp)类 ############################################
total6 = 0
# 先求分母
row6 = int(input("Enter a number that represents the reason1: "))
row7 = int(input("Enter a number that represents the reason2: "))
for a5 in range(cols):
    for b5 in range(a5+1, cols):
        for c5 in range(a5+2, cols):
            x_1 = expert_mat[row6, a5] + expert_mat[row6, b5] + expert_mat[row6, c5] + expert_mat[row7, a5] + expert_mat[row7, b5] + expert_mat[row7, c5]
            y_1 = expert_mat[row6, a5] * expert_mat[row6, b5] * expert_mat[row6, c5] * expert_mat[row7, a5] * expert_mat[row7, b5] * expert_mat[row7, c5]
            z_1 = y_1 / x_1
            total6 += z_1

# 再求分子
col_5 = int(input("Enter a number that represents the disease1: "))
col_6 = int(input("Enter a number that represents the disease2: "))
col_7 = int(input("Enter a number that represents the disease3: "))
x_2 = expert_mat[row6, col_5] + expert_mat[row6, col_6] + expert_mat[row6, col_7] + expert_mat[row7, col_5] + expert_mat[row7, col_6] + expert_mat[row7, col_7]
y_2 = expert_mat[row6, col_5] * expert_mat[row6, col_6] * expert_mat[row6, col_7] * expert_mat[row7, col_5] * expert_mat[row7, col_6] * expert_mat[row7, col_7]
z_2 = y_2 / x_2

final6 = z_2 / total6
print(final6)



############################################ μ(mi*mj*mk)(xo*xp*xq)类 ##########################################
total7 = 0
# 先求分母
col7 = int(input("Enter a number that represents the disease1: "))
col8 = int(input("Enter a number that represents the disease2: "))
col9 = int(input("Enter a number that represents the disease3: "))

for a6 in range(rows):
    for b6 in range(a6+1, rows):
        for c6 in range(a6+2, rows):
            x_3 = expert_mat[a6, col7] + expert_mat[a6, col8] + expert_mat[a6, col9] + expert_mat[b6, col7] + expert_mat[b6, col8] + expert_mat[b6, col9] + expert_mat[c6, col7] + expert_mat[c6, col8] + expert_mat[c6, col9]
            y_3 = expert_mat[a6, col7] * expert_mat[a6, col8] * expert_mat[a6, col9] * expert_mat[b6, col7] * expert_mat[b6, col8] * expert_mat[b6, col9] * expert_mat[c6, col7] * expert_mat[c6, col8] * expert_mat[c6, col9]
            z_3 = y_3 / x_3
            total7 += z_3

# 再求分子
row8 = int(input("Enter a number that represents the reason1: "))
row9 = int(input("Enter a number that represents the reason2: "))
row10 = int(input("Enter a number that represents the reason3: "))
x_4 = expert_mat[row8, col7] + expert_mat[row8, col8] + expert_mat[row8, col9] + expert_mat[row9, col7] + expert_mat[row9, col8] + expert_mat[row9, col9] + expert_mat[row10, col7] + expert_mat[row10, col8] + expert_mat[row10, col9]
y_4 = expert_mat[row8, col7] * expert_mat[row8, col8] * expert_mat[row8, col9] * expert_mat[row9, col7] * expert_mat[row9, col8] * expert_mat[row9, col9] * expert_mat[row10, col7] * expert_mat[row10, col8] * expert_mat[row10, col9]
z_4 = y_4 / x_4

final7 = z_4 / total7
print(final7)