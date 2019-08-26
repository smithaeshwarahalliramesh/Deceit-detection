"""
Script to build dataset
- Binary attributes
- 39 columns
- csv format
"""

import random
import csv
import sys
import getopt

def getVal(valueSet, n):
  val = random.sample(valueSet, k = n)
  return val[0]

def usage():
    print('python create_data.py -o <outputFile in csv format> -n <number of rows>')

def main(argv):
    # Initializations
    outFile = None
    nrows = None
    attrValueSet = [0, 1]
    n = 1
    colHdrs = ["OtherGestures", "Smile", "Laugh", "Scowl", "otherEyebrowMovement", "Frown", "Raise", "OtherEyeMovements", "Close-R", "X-Open",
               "Close-BE", "gazeInterlocutor", "gazeDown", "gazeUp", "otherGaze", "gazeSide", "openMouth", "closeMouth", "lipsDown", "lipsUp",
               "lipsRetracted", "lipsProtruded", "SideTurn", "downR", "sideTilt", "backHead", "otherHeadM", "sideTurnR", "sideTiltR","waggle",
               "forwardHead", "downRHead", "singleHand", "bothHands", "otherHandM", "complexHandM", "sidewaysHand", "downHands", "upHands"] 

    try:
        opts, args = getopt.getopt(argv,"hn:o:")
    except getopt.GetoptError:
        usage()
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            usage()
            sys.exit()
        elif opt == '-o':
            outFile = arg
        elif opt == '-n':
            nrows = int(arg)
        else:
            usage()
            print('Invalid argument %s' % opt)
            sys.exit(2)

    if (None == outFile) or (None == nrows):
        print('Missing arguments')
        usage()
        sys.exit(2)

    rows = [[getVal(attrValueSet, n) for i in range(len(colHdrs))] for j in range(nrows)]
    with open(outFile, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(hdr for hdr in colHdrs)
        writer.writerows(rows)

    print('%s successfully created' % outFile)

main(sys.argv[1:])
