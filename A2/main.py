from MC_learning import first_visit, every_visit
from TD_learning import td0_learning
from FormatPrint import format_print

if __name__ == '__main__':
    value1 = first_visit(try_times=10000)
    value2 = every_visit(try_times=10000)
    value3 = td0_learning(try_times=10000)

    print("mc_first_visit:")
    format_print(value1)
    # print("---------------------------------------------")
    print("mc_every_visit:")
    format_print(value2)
    # print("---------------------------------------------")
    print("td0_learning:")
    format_print(value3)
