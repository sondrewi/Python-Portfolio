import re

comp = {
    "0": "0101010",
    "1": "0111111",
    "-1": "0111010",
    "D": "0001100",
    "A": "0110000",
    "!D": "0001101",
    "!A": "0110001",
    "-D": "0001111",
    "-A": "0110011",
    "D+1": "0011111",
    "A+1": "0110111",
    "D-1": "0001110",
    "A-1": "0110010",
    "D+A": "0000010",
    "D-A": "0010011",
    "A-D": "0000111",
    "D&A": "0000000",
    "D|A": "0010101",
    "M": "1110000",
    "!M": "1110001",
    "-M": "1110011",
    "M+1": "1110111",
    "M-1": "1110010",
    "D+M": "1000010",
    "D-M": "1010011",
    "M-D": "1000111",
    "D&M": "1000000",
    "D|M": "1010101"
}


dest = {
    "null": "000",
    "M": "001",
    "D": "010",
    "A": "100",
    "MD": "011",
    "AM": "101",
    "AD": "110",
    "AMD": "111"
}


jump = {
    "null": "000",
    "JGT": "001",
    "JEQ": "010",
    "JGE": "011",
    "JLT": "100",
    "JNE": "101",
    "JLE": "110",
    "JMP": "111"
}

labels = {'SP': '0',
          'LCL': '1',
          'ARG': '2',
          'THIS': '3',
          'THAT': '4',
          'R0': '0',
          'R1': '1',
          'R2': '2',
          'R3': '3',
          'R4': '4',
          'R5': '5',
          'R6': '6',
          'R7': '7',
          'R8': '8',
          'R9': '9',
          'R10': '10',
          'R11': '11',
          'R12': '12',
          'R13': '13',
          'R14': '14',
          'R15': '15',
          'SCREEN': '16384',
          'KBD': '24576'}

txt_file = []
last_label = 15


def assembler(stage, project, file):
    import os
    directory = '/Users/sondrew/Desktop/nand2tetris/projects/07/' + stage + '/' + project
    os.chdir(directory)

    # first pass
    file = open(file, 'r')
    lines = file.readlines()
    lines = [line[:-1] for line in lines if '/' != line[0]]
    lines = [line.split('//', 3)[0] for line in lines]
    lines = [x for x in lines if x]
    lines = [re.sub(r'\s+', '', x) for x in lines]

    linecount = 0

    for line in lines:
        if '(' in line:
            labels[line[1:-1]] = linecount

        else:
            linecount += 1

    lines = [line for line in lines if '(' not in line]

    # second pass
    currentline = 0
    linesleft = len(lines) - currentline

    for line in lines:
        ins_type, content = parser(line)

        if ins_type == 'A':
            txt_file.append('0' + deci_to_bin(content))

        elif ins_type == 'CJ':
            txt_file.append(
                '111' + comp[content[0]] + dest['null'] + jump[content[1]])

        else:
            txt_file.append(
                '111' + comp[content[1]] + dest[content[0]] + jump['null'])

        if linesleft:
            currentline += 1

    new_file = open(project + '.hack', 'w')

    for i in txt_file:
        new_file.write(i + '\n')


def parser(line):
    instruction_type = ''
    content = ''

    if '@' in line:
        instruction_type = 'A'
    elif ';' in line:
        instruction_type = 'CJ'
    else:
        instruction_type = 'CA'

    if instruction_type == 'A':
        if line[1:] in labels:
            content = int(labels[line[1:]])
        elif re.search('[A-Z]|[a-z]', line):
            global last_label
            last_label += 1
            labels[line[1:]] = str(last_label)
            content = int(labels[line[1:]])
        else:
            content = int(line[1:])

    elif instruction_type == 'CA':
        content = line.split('=', 1)

    else:
        content = line.split(';', 1)

    return instruction_type, content


def deci_to_bin(number):
    decimal = number
    binary = ''
    power_list = [2**i for i in reversed(range(15))]
    for i in power_list:
        if decimal >= i:
            binary = binary + '1'
            decimal -= i
        else:
            binary = binary + '0'

    return binary


import re
import os
directory = '/Users/sondrew/Desktop/nand2tetris/projects/08/' + stage + '/' + project
print(os.chdir(directory))
