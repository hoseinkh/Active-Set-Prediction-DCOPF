#
import matplotlib.pyplot as plt
#
#
#
#
dict_LDA_accuracy = {
    # 10: [78.90738060781476, 91.17221418234442, 94.6363965267728, 96.16497829232996, 96.8342981186686, 97.31367583212736, 97.64833574529666, 98.04630969609262, 98.20911722141823, 98.38096960926194],
    10: [96.8342981186686, 98.38096960926194],
    20: [97.42221418234442, 98.82416787264833],
    30: [98.32670043415341, 99.30354558610709],
    40: [96.49059334298119, 98.83321273516643],
    # 50: [92.72793053545585, 97.43125904486251],
    50: [91.51,             96.61],
}
#
dict_Naive_accuracy = {
    10: [87.54522431259045, 95.11577424023154],
    20: [82.71526772793053, 93.37916063675832],
    30: [75.43415340086831, 92.89978292329957],
    40: [85.98950795947902, 92.37518089725036],
    50: [67.10383502170767, 81.85600578871202]
}
#
x_vector = [10, 20, 30, 40, 50]
y_LDA_5_candidate  = [dict_LDA_accuracy[10][0], dict_LDA_accuracy[20][0], dict_LDA_accuracy[30][0], dict_LDA_accuracy[40][0], dict_LDA_accuracy[50][0]]
y_LDA_10_candidate = [dict_LDA_accuracy[10][1], dict_LDA_accuracy[20][1], dict_LDA_accuracy[30][1], dict_LDA_accuracy[40][1], dict_LDA_accuracy[50][1]]
#
y_naive_5_candidate  = [dict_Naive_accuracy[10][0], dict_Naive_accuracy[20][0], dict_Naive_accuracy[30][0], dict_Naive_accuracy[40][0], dict_Naive_accuracy[50][0]]
y_naive_10_candidate = [dict_Naive_accuracy[10][1], dict_Naive_accuracy[20][1], dict_Naive_accuracy[30][1], dict_Naive_accuracy[40][1], dict_Naive_accuracy[50][1]]
#
HHH = [i for i in plt.rcParams["figure.figsize"]]
plt.rcParams["figure.figsize"] = [HHH[0], 0.5*HHH[1]]
plt.plot(x_vector, y_LDA_5_candidate, linestyle='-',     marker='s',     label="LDA:   5-Candidates")
plt.plot(x_vector, y_LDA_10_candidate, linestyle='-',     marker='o',    label="LDA:   10-Candidates")
# plt.plot(x_vector, y_naive_5_candidate, linestyle='--',  marker='s',     label="Naive: 5-Candidates")
# plt.plot(x_vector, y_naive_10_candidate, linestyle='--', marker='o',     label="Naive: 10-Candidates")
plt.xlabel("Renewable penetration level (%)", fontsize=12)
plt.ylabel("Accuracy (%)", fontsize=12)
plt.legend(fontsize=10)
plt.show()
5 + 6
#
#




