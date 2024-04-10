# read two cvs files and compare the results
# Column 1: immage name
# Column 2: PSNR
# Column 3: SSIM


import csv
import os
import sys

def read_csv(file):
    with open(file, 'r') as f:
        reader = csv.reader(f)
        result = list(reader)
    return result

def write_csv(file, data):
    if not os.path.exists(os.path.dirname(file)):
        try:
            os.makedirs(os.path.dirname(file))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
        
    with open(file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(data)

def compare(data1, data2):
    better = []
    worse = []
    for i in range(len(data1)):
        if data1[i][0] != data2[i][0]:
            print('Error: image name is not the same')
            sys.exit()
        psnr1 = float(data1[i][1])
        psnr2 = float(data2[i][1])
        diff  = psnr1 - psnr2
        if diff > 1:
            better.append([data1[i][0], diff])
        elif diff < -1:
            worse.append([data1[i][0], diff])
    return better, worse

def main():
    filePath1 = './results/reside6k/30.79_0.9754_PA2_5.csv'
    filePath2 = './results/reside6k/30.32_0.9739_baseline.csv'
    data1 = read_csv(filePath1)
    data2 = read_csv(filePath2)
    better, worse = compare(data1, data2)
    write_csv('./results/compare/better.csv', better)
    write_csv('./results/compare/worse.csv', worse)


if __name__ == '__main__':
    main()