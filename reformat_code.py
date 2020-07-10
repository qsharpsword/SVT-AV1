######
## Jack Zhou 2019
######

from __future__ import print_function
import os

def rm_tab_and_trailing_spaces(fpath):

    modified_lines, filebuffer = [], []

    with open(fpath,'r+') as in_f:
        filebuffer = in_f.readlines()

    for i in range(len(filebuffer)):

        cleaned_Str = ""
        # for char in string with no trailing space
        for ch in filebuffer[i].rstrip('\n').rstrip():
            if ch == '\t':
                cleaned_Str += "    " # 4 spaces instead of tabs
            else:
                cleaned_Str += ch

        if filebuffer[i] != cleaned_Str+'\n':
            modified_lines.append(i + 1)

        filebuffer[i] = cleaned_Str

    with open(fpath,'w+') as out_f:
        out_f.write('\n'.join(filebuffer)+'\n')

    return modified_lines

def isWritablefile(fn):

    acceptable_formats = [  '.md','.txt','.c',
                            '.cc','h','.cfg',
                            '.patch','.build',
                            '.clang-format',]

    for format_ in acceptable_formats:
        if fn.endswith(format_):
            return True

    return False

folders = [file_ for file_ in os.listdir('.') if (
                (os.path.isdir(file_)) and 
                (file_ != "Build") and 
                (not file_.startswith('.'))
                )
        ]

report = []

for folder in folders:
    all_files = []
    for root, dirs, files in os.walk(folder, topdown=True):
        for name in files:
            all_files.append(os.path.join(root, name))
        for name in dirs:
            all_files.append(os.path.join(root, name))

    for file_ in [f for f in all_files if not os.path.isdir(f)]:
        if isWritablefile(file_):
            modified_buffer = rm_tab_and_trailing_spaces(file_)
            if modified_buffer != []:
                for modified_line in modified_buffer:
                    report.append("In file: " + file_ + "\tLine " + str(modified_line) +  " has been modified.")

for line in report:
    print(line)

print("DONE")
os.system("PAUSE")