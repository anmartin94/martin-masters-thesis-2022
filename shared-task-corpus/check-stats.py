import os
import pandas as pd
import re

txt_files = os.listdir("annotated-txt-files")
n_null, n_title, n_abstract, n_intro, n_task, n_conclusion = 0, 0, 0, 0, 0, 0
other_headers = []
for txt_file in txt_files:
    with open("annotated-txt-files/"+txt_file, 'r') as f:
        lines = f.readlines()
        current_header = ""
        prev_blank = False
        found = False
        title = ""
        for line in lines:
            if line == "abstract\n" or line == "title\n" or prev_blank:
                current_header = line[:-1]
                prev_blank = False
                continue
            if line == '\n':
                prev_blank = True
                continue
            else:
                prev_blank = False
            if current_header == "title":
                title = line[:-1]
            if "<TASK>" in line:
                if current_header != "title":
                    pattern = re.compile(r"<TASK>(.+)</TASK>")
                    match = pattern.search(line)
                    #if match is not None:
                        #if match.group(1) in title:
                            #print(current_header, txt_file)
                if current_header == "title":
                    n_title += 1
                elif current_header == "abstract":
                    n_abstract += 1
                elif "introduction" in current_header.lower():
                    n_intro += 1
                elif "task" in current_header.lower():
                    n_task += 1
                elif "conclusion" in current_header.lower() or "summary" in current_header.lower():
                    n_conclusion += 1
                else:
                    other_headers.append(current_header)
                found = True
        if not found:
            n_null += 1
            print(txt_file)
    f.close()
print(n_null, n_title, n_abstract, n_intro, n_task, n_conclusion)
print(other_headers)
