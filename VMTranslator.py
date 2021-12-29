txt_file = []
import copy

GTcount = 0
LTcount = 0
EQcount = 0
ENDcount = 0
returnaddresscount = 0
static_count = 0
current_func = ''

labels = {'SP': '0',
          'local': 'LCL',
          'argument': 'ARG',
          'pointer': '3',
          'this': 'THIS',
          'that': 'THAT',
          'temp': '5',
          'constant': '0',
          'static': '16',
          }

am_commands = ['add', 'sub', 'neg', 'eq', 'gt', 'lt', 'and', 'or', 'not']


def VMAmTranslator(stage, project, file):
    import re
    import os
    directory = '/Users/sondrew/Desktop/nand2tetris/projects/08/' + stage + '/' + project
    os.chdir(directory)
    file_list = os.listdir(directory)
    stripped_file_list = [x for x in file_list if '.vm' in x]
    stripped_file_list = [x for x in stripped_file_list if 'Sys' not in x]
    file_name = copy.copy(file)

    # first pass
    file = open(file, 'r')
    lines = file.readlines()
    lines = [line[:-1] for line in lines if '/' != line[0]]
    lines = [line.split('//', 3)[0] for line in lines]
    lines = [x for x in lines if x]
    lines = [line.split(' ', 3) + [file_name[:-3]] for line in lines]
    sys_length = copy.copy(len(lines))

    if file_name == 'Sys.vm':

        # bootstrap
        txt_file = ['@256', 'D=A', '@0', 'M=D']

        bootstrap_lines = [['call', 'Sys.init', '0']]

        for class_file in stripped_file_list:
            func_file = open(class_file, 'r')
            class_lines = func_file.readlines()
            class_lines = [line[:-1] for line in class_lines if '/' != line[0]]
            class_lines = [line.split('//', 3)[0] for line in class_lines]
            class_lines = [x for x in class_lines if x]
            class_lines = [line.split(
                ' ', 3) + [class_file[:-3]] for line in class_lines]
            lines.extend(class_lines)

    lines = bootstrap_lines + lines

    for line in lines:
        commandType = ''

        if line[0] in am_commands:
            commandType = 'C_ARITHMETIC'

        elif line[0] == 'push':
            commandType = 'C_PUSH'

        elif line[0] == 'pop':
            commandType = 'C_POP'

        elif line[0] == 'label':
            commandType = 'C_LABEL'

        elif line[0] == 'if-goto':
            commandType = 'C_IF'

        elif line[0] == 'goto':
            commandType = 'C_GOTO'

        elif line[0] == 'return':
            commandType = 'C_RETURN'

        elif line[0] == 'function':
            commandType = 'C_FUNCTION'

        elif line[0] == 'call':
            commandType = 'C_CALL'

        asm = CodeWriter(commandType, line, file, file_name)
        txt_file.append(asm)

    new_file = open(project + '.asm', 'w')

    for i in txt_file:
        new_file.write(str(i) + '\n')

def CodeWriter(commandType, line, file, file_name):

    global labels

    if commandType == 'C_PUSH':
        if line[1] == 'constant':
            line1 = '@' + line[2] + '\n'
            line2 = 'D=A\n'
            line3 = '@SP\n'
            line4 = 'A=M\n'
            line5 = 'M=D\n'
            line6 = '@SP\n'
            line7 = 'M=M+1'

            return line1 + line2 + line3 + line4 + line5 + line6 + line7

        elif line[1] in ['local', 'this', 'that', 'argument']:
            line1 = '@' + (labels[line[1]]) + '\n'
            line2 = 'D=M\n'
            line3 = '@' + line[2] + '\n'
            line4 = 'D=D+A\n'
            line5 = 'A=D\n'
            line6 = 'D=M\n'
            line7 = '@SP\n'
            line8 = 'A=M\n'
            line9 = 'M=D\n'
            line10 = '@SP\n'
            line11 = 'M=M+1'

            return line1 + line2 + line3 + line4 + line5 + line6 + line7 + line8 + line9 + line10 + line11

        elif line[1] in ['pointer', 'temp']:
            line1 = '@' + str(int(labels[line[1]]) + int(line[2])) + '\n'
            line2 = 'D=M\n'
            line3 = '@SP\n'
            line4 = 'A=M\n'
            line5 = 'M=D\n'
            line6 = '@SP\n'
            line7 = 'M=M+1'

            return line1 + line2 + line3 + line4 + line5 + line6 + line7

        elif line[1] == 'static':
            global current_func
            line1 = '@' + current_func + '.' + line[2] + '\n'
            line2 = 'D=M\n'
            line3 = '@SP\n'
            line4 = 'A=M\n'
            line5 = 'M=D\n'
            line6 = '@SP\n'
            line7 = 'M=M+1'

            return line1 + line2 + line3 + line4 + line5 + line6 + line7

    elif commandType == 'C_POP':

        if line[1] in ['local', 'this', 'that', 'argument']:
            line1 = '@' + (labels[line[1]]) + '\n'
            line2 = 'D=M\n'
            line3 = '@' + line[2] + '\n'
            line4 = 'D=D+A\n'
            line5 = '@R13\n'
            line6 = 'M=D\n'
            line7 = '@SP\n'
            line8 = 'M=M-1\n'
            line9 = 'A=M\n'
            line10 = 'D=M\n'
            line11 = '@R13\n'
            line12 = 'A=M\n'
            line13 = 'M=D'

            return line1 + line2 + line3 + line4 + line5 + line6 + line7 + line8 + line9 + line10 + line11 + line12 + line13

        elif line[1] in ['temp', 'pointer']:
            line1 = '@SP\n'
            line2 = 'M=M-1\n'
            line3 = 'A=M\n'
            line4 = 'D=M\n'
            line5 = '@' + str(int(labels[line[1]]) + int(line[2])) + '\n'
            line6 = 'M=D'

            return line1 + line2 + line3 + line4 + line5 + line6

        elif line[1] == 'static':
            line1 = '@SP\n'
            line2 = 'M=M-1\n'
            line3 = 'A=M\n'
            line4 = 'D=M\n'
            line5 = '@' + current_func + '.' + line[2] + '\n'
            line6 = 'M=D'

            return line1 + line2 + line3 + line4 + line5 + line6

    elif commandType == 'C_ARITHMETIC':
        if line[0] == 'add':
            line1 = '@SP\n'
            line2 = 'M=M-1\n'
            line3 = 'A=M\n'
            line4 = 'D=M\n'
            line5 = '@SP\n'
            line6 = 'M=M-1\n'
            line7 = 'A=M\n'
            line8 = 'M=D+M\n'
            line9 = '@SP\n'
            line10 = 'M=M+1'

            return line1 + line2 + line3 + line4 + line5 + line6 + line7 + line8 + line9 + line10

        elif line[0] == 'sub':
            line1 = '@SP\n'
            line2 = 'M=M-1\n'
            line3 = 'A=M\n'
            line4 = 'D=M\n'
            line5 = '@SP\n'
            line6 = 'M=M-1\n'
            line7 = 'A=M\n'
            line8 = 'M=M-D\n'
            line9 = '@SP\n'
            line10 = 'M=M+1'

            return line1 + line2 + line3 + line4 + line5 + line6 + line7 + line8 + line9 + line10

        elif line[0] == 'neg':
            line1 = '@SP\n'
            line2 = 'M=M-1\n'
            line3 = 'A=M\n'
            line4 = 'M=-M\n'
            line5 = '@SP\n'
            line6 = 'M=M+1'

            return line1 + line2 + line3 + line4 + line5 + line6

        elif line[0] == 'eq':
            line1 = '@SP\n'
            line2 = 'M=M-1\n'
            line3 = 'A=M\n'
            line4 = 'D=M\n'
            line5 = '@SP\n'
            line6 = 'M=M-1\n'
            line7 = 'A=M\n'
            line8 = 'D=M-D\n'
            global EQcount
            line9 = f'@EQUAL{EQcount}\n'
            line10 = 'D;JEQ\n'
            line11 = '@SP\n'
            line12 = 'A=M\n'
            line13 = 'M=0\n'
            line14 = '@SP\n'
            line15 = 'M=M+1\n'
            global ENDcount
            line16 = f'@END{ENDcount}\n'
            line17 = '0;JMP\n'
            line18 = f'(EQUAL{EQcount})\n'
            EQcount += 1
            line19 = '@SP\n'
            line20 = 'A=M\n'
            line21 = 'M=-1\n'
            line22 = '@SP\n'
            line23 = 'M=M+1\n'
            line24 = f'(END{ENDcount})'
            ENDcount += 1

            return line1 + line2 + line3 + line4 + line5 + line6 + line7 + line8 + line9 + line10 + line11 + line12 + line13 + line14 + line15 + \
                line16 + line17 + line18 + line19 + line20 + line21 + line22 + line23 + line24

        elif line[0] == 'gt':
            line1 = '@SP\n'
            line2 = 'M=M-1\n'
            line3 = 'A=M\n'
            line4 = 'D=M\n'
            line5 = '@SP\n'
            line6 = 'M=M-1\n'
            line7 = 'A=M\n'
            line8 = 'D=M-D\n'
            global GTcount
            line9 = f'@GT{GTcount}\n'
            line10 = 'D;JGT\n'
            line11 = '@SP\n'
            line12 = 'A=M\n'
            line13 = 'M=0\n'
            line14 = '@SP\n'
            line15 = 'M=M+1\n'
            line16 = f'@END{ENDcount}\n'
            line17 = '0;JMP\n'
            line18 = f'(GT{GTcount})\n'
            GTcount += 1
            line19 = '@SP\n'
            line20 = 'A=M\n'
            line21 = 'M=-1\n'
            line22 = '@SP\n'
            line23 = 'M=M+1\n'
            line24 = f'(END{ENDcount})'
            ENDcount += 1

            return line1 + line2 + line3 + line4 + line5 + line6 + line7 + line8 + line9 + line10 + line11 + line12 + line13 + line14 + line15 + \
                line16 + line17 + line18 + line19 + line20 + line21 + line22 + line23 + line24

        elif line[0] == 'lt':
            line1 = '@SP\n'
            line2 = 'M=M-1\n'
            line3 = 'A=M\n'
            line4 = 'D=M\n'
            line5 = '@SP\n'
            line6 = 'M=M-1\n'
            line7 = 'A=M\n'
            line8 = 'D=M-D\n'
            global LTcount
            line9 = f'@LT{LTcount}\n'
            line10 = 'D;JLT\n'
            line11 = '@SP\n'
            line12 = 'A=M\n'
            line13 = 'M=0\n'
            line14 = '@SP\n'
            line15 = 'M=M+1\n'
            line16 = f'@END{ENDcount}\n'
            line17 = '0;JMP\n'
            line18 = f'(LT{LTcount})\n'
            LTcount += 1
            line19 = '@SP\n'
            line20 = 'A=M\n'
            line21 = 'M=-1\n'
            line22 = '@SP\n'
            line23 = 'M=M+1\n'
            line24 = f'(END{ENDcount})'
            ENDcount += 1

            return line1 + line2 + line3 + line4 + line5 + line6 + line7 + line8 + line9 + line10 + line11 + line12 + line13 + line14 + line15 + \
                line16 + line17 + line18 + line19 + line20 + line21 + line22 + line23 + line24

        elif line[0] == 'and':
            line1 = '@SP\n'
            line2 = 'M=M-1\n'
            line3 = 'A=M\n'
            line4 = 'D=M\n'
            line5 = '@SP\n'
            line6 = 'M=M-1\n'
            line7 = 'A=M\n'
            line8 = 'M=D&M\n'
            line9 = '@SP\n'
            line10 = 'M=M+1'

            return line1 + line2 + line3 + line4 + line5 + line6 + line7 + line8 + line9 + line10

        elif line[0] == 'or':
            line1 = '@SP\n'
            line2 = 'M=M-1\n'
            line3 = 'A=M\n'
            line4 = 'D=M\n'
            line5 = '@SP\n'
            line6 = 'M=M-1\n'
            line7 = 'A=M\n'
            line8 = 'M=D|M\n'
            line9 = '@SP\n'
            line10 = 'M=M+1'

            return line1 + line2 + line3 + line4 + line5 + line6 + line7 + line8 + line9 + line10

        elif line[0] == 'not':
            line1 = '@SP\n'
            line2 = 'M=M-1\n'
            line3 = 'A=M\n'
            line4 = 'M=!M\n'
            line5 = '@SP\n'
            line6 = 'M=M+1'

            return line1 + line2 + line3 + line4 + line5 + line6

    elif commandType == 'C_LABEL':
        line1 = '(' + line[-1] + '$' + line[1] + ')'

        return line1

    elif commandType == 'C_IF':
        line1 = '@SP\n'
        line2 = 'M=M-1\n'
        line3 = 'A=M\n'
        line4 = 'D=M\n'
        line5 = '@' + line[-1] + '$' + line[1] + '\n'
        line6 = 'D;JNE'

        return line1 + line2 + line3 + line4 + line5 + line6

    elif commandType == 'C_GOTO':
        line1 = '@' + line[-1] + '$' + line[1] + '\n'
        line2 = '0;JMP'

        return line1 + line2

    elif commandType == 'C_FUNCTION':
        line1 = '(' + line[1] + ')\n'
        Clines = []

        # set local vars to 0
        for i in range(int(line[2])):
            Clines.append('@SP\n')
            Clines.append('A=M\n')
            Clines.append('M=0\n')
            Clines.append('@SP\n')
            Clines.append('M=M+1\n')

        Clines = ''.join(Clines)
        all_lines = line1 + Clines

        current_func = line[1].split('.', 3)[0]

        return all_lines[:-1]

    elif commandType == 'C_RETURN':
        # set frame to LCL
        line1 = '@LCL\n'
        line2 = 'D=M\n'
        line3 = '@13\n'
        line4 = 'M=D\n'

        # set RET to frame-5
        line5 = '@5\n'
        line6 = 'D=A\n'
        line7 = '@13\n'
        line8 = 'A=M-D\n'
        line9 = 'D=M\n'
        line10 = '@14\n'
        line11 = 'M=D\n'

        # put the value to be returned in the position where ARG starts
        line12 = '@SP\n'
        line13 = 'A=M-1\n'
        line14 = 'D=M\n'
        line15 = '@ARG\n'
        line16 = 'A=M\n'
        line17 = 'M=D\n'

        # restore SP location
        line18 = '@ARG\n'
        line19 = 'D=M\n'
        line20 = 'D=D+1\n'
        line21 = '@SP\n'
        line22 = 'M=D\n'

        # restore THAT of caller
        line23 = '@13\n'
        line24 = 'A=M-1\n'
        line25 = 'D=M\n'
        line26 = '@THAT\n'
        line27 = 'M=D\n'

        # restore THIS of caller
        line28 = '@13\n'
        line29 = 'A=M-1\n'
        line30 = 'A=A-1\n'
        line31 = 'D=M\n'
        line32 = '@THIS\n'
        line33 = 'M=D\n'

        # restore ARG of caller
        line34 = '@13\n'
        line35 = 'A=M-1\n'
        line36 = 'A=A-1\n'
        line37 = 'A=A-1\n'
        line38 = 'D=M\n'
        line39 = '@ARG\n'
        line40 = 'M=D\n'

        # restore LCL of caller
        line41 = '@13\n'
        line42 = 'A=M-1\n'
        line43 = 'A=A-1\n'
        line44 = 'A=A-1\n'
        line45 = 'A=A-1\n'
        line46 = 'D=M\n'
        line47 = '@LCL\n'
        line48 = 'M=D\n'

        # goto return address (a point in the caller's code)
        line49 = '@14\n'
        line50 = 'A=M\n'
        line51 = '0;JMP'

        return line1 + line2 + line3 + line4 + line5 + line6 + line7 + line8 + line9 + line10 + line11 + line12 + line13 + line14 + line15 + \
            line16 + line17 + line18 + line19 + line20 + line21 + \
            line22 + line23 + line24 + line25 + line26 + line27 + line28 + line29 + line30 + line31 + line32 + line33 + line34 + line35 + \
            line36 + line37 + line38 + line39 + line40 + \
            line41 + line42 + line43 + line44 + line45 + \
            line46 + line47 + line48 + line49 + line50 + line51

    elif commandType == 'C_CALL':

        # push return address
        global returnaddresscount
        line1 = f'@return-address{returnaddresscount}\n'
        line2 = 'D=A\n'
        line3 = '@SP\n'
        line4 = 'A=M\n'
        line5 = 'M=D\n'
        line6 = '@SP\n'
        line7 = 'M=M+1\n'

        # push LCL
        line8 = '@LCL\n'
        line9 = 'D=M\n'
        line10 = '@SP\n'
        line11 = 'A=M\n'
        line12 = 'M=D\n'
        line13 = '@SP\n'
        line14 = 'M=M+1\n'

        # push ARG
        line15 = '@ARG\n'
        line16 = 'D=M\n'
        line17 = '@SP\n'
        line18 = 'A=M\n'
        line19 = 'M=D\n'
        line20 = '@SP\n'
        line21 = 'M=M+1\n'

        # push this
        line22 = '@THIS\n'
        line23 = 'D=M\n'
        line24 = '@SP\n'
        line25 = 'A=M\n'
        line26 = 'M=D\n'
        line27 = '@SP\n'
        line28 = 'M=M+1\n'

        # push that
        line29 = '@THAT\n'
        line30 = 'D=M\n'
        line31 = '@SP\n'
        line32 = 'A=M\n'
        line33 = 'M=D\n'
        line34 = '@SP\n'
        line35 = 'M=M+1\n'

        # reposition ARG
        line36 = '@' + line[2] + '\n'
        line37 = 'D=A\n'
        line38 = '@5\n'
        line39 = 'D=D+A\n'
        line40 = '@SP\n'
        line41 = 'D=M-D\n'
        line42 = '@ARG\n'
        line43 = 'M=D\n'

        # reposition LCL
        line44 = '@SP\n'
        line45 = 'D=M\n'
        line46 = '@LCL\n'
        line47 = 'M=D\n'

        # goto f
        line48 = '@' + line[1] + '\n'
        line49 = '0;JMP\n'

        current_func = line[1]

        # declare return address
        line50 = f'(return-address{returnaddresscount})'
        returnaddresscount += 1

        return line1 + line2 + line3 + line4 + line5 + line6 + line7 + line8 + line9 + line10 + line11 + line12 + line13 + line14 + line15 + \
            line16 + line17 + line18 + line19 + line20 + line21 + \
            line22 + line23 + line24 + line25 + line26 + line27 + line28 + line29 + line30 + line31 + line32 + line33 + line34 + line35 + \
            line36 + line37 + line38 + line39 + line40 + line41 + line42 + line43 + line44 + line45 + line46 + line47 + line48 + \
            line49 + line50


VMAmTranslator('FunctionCalls', 'NestedCall', 'Sys.vm')
