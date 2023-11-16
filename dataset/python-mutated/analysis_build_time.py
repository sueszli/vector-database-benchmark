import os
root_path = os.path.abspath(os.path.join(os.getcwd(), '..'))

def strToSecond(strTime):
    if False:
        print('Hello World!')
    minute = int(strTime.split(':')[0])
    second = int(strTime.split(':')[1].split('.')[0]) + 1
    return minute * 60 + second

def getUsefulBuildTimeFile(filename):
    if False:
        i = 10
        return i + 15
    os.system(f"grep -Po -- '-o .*' {filename} | grep ' elapsed' | grep -P -v '0:00.* elapse' > {root_path}/tools/analysis_build_time")
    os.system(f"grep -v  -- '-o .*' {filename} |grep ' elapse' |  grep -P -v '0:00.* elapse' >> {root_path}/tools/analysis_build_time")

def analysisBuildTime():
    if False:
        for i in range(10):
            print('nop')
    filename = '%s/build/build-time' % root_path
    getUsefulBuildTimeFile(filename)
    os.system('rm -rf %s/tools/tempbuildTime.txt' % root_path)
    with open('%s/tools/analysis_build_time' % root_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            try:
                line = line.strip()
                if '-o ' in line:
                    buildFile = line.split(', ')[0].split(' ')[1]
                    buildTime = line.split(', ')[1].split('elapsed')[0].strip()
                    secondTime = strToSecond(buildTime)
                    os.system(f'echo {buildFile}, {secondTime} >> {root_path}/tools/tempbuildTime.txt')
                else:
                    buildTime = line.split(', ')[1].split('elapsed')[0].strip()
                    secondTime = strToSecond(buildTime)
                    if secondTime > 30:
                        os.system(f'echo {line}, {secondTime} >> {root_path}/tools/tempbuildTime.txt')
            except ValueError:
                print(line)
    os.system(f'sort -n -k 2 -r {root_path}/tools/tempbuildTime.txt > {root_path}/tools/buildTime.txt')
analysisBuildTime()