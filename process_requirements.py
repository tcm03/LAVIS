import re

new_req = {}

with open('new_requirements.txt') as f:
    for line in f:
        if '==' not in line:
            continue
        libname, version = line.split('==')
        new_req[libname] = version

with open('requirements.txt') as f:
    requirements_txt = f.read()
    # Extract library names using regex
    lib_names = []
    for line in requirements_txt.splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            lib_name = re.split(r"[=<>!~]+", line)[0]
            lib_names.append(lib_name)

final_req = {}
for lib_name in lib_names:
    if lib_name in new_req:
        final_req[lib_name] = new_req[lib_name]

with open('final_requirements.txt', 'w') as f:
    for lib_name, version in final_req.items():
        f.write(f"{lib_name}=={version}\n")