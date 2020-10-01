import numpy as np

def readData():
    data = np.genfromtxt('out.csv', delimiter=',')
    # drop last element due to comma
    data = data[:-1]
    return data

def mergeIter(data, mergeSize, baseCase):
    newData = np.zeros(data.size)

    mergeSize = 16 if baseCase else mergeSize
    mergeCount = data.size / mergeSize
    
    for i in range(mergeCount):
        begin = i * mergeSize
        end = (i + 1) * mergeSize

        mergeBlock = data[begin:end]
        newData[begin:end] = np.sort(mergeBlock)

    return newData

def mergeSort(data, ways, iterCount):
    buffer0 = np.zeros(data.size)
    buffer1 = np.zeros(data.size)

    # base case
    mergeSize = 16
    for i in range(iterCount):
        isEven = (i % 2 == 0)
        currentBuffer = buffer0 if isEven else buffer1
        nextBuffer = buffer1 if isEven else buffer0

        if (i == 0):
            n = mergeIter(data, None, True)
            np.copyto(nextBuffer, n)
        else:
            mergeSize *= ways
            n = mergeIter(currentBuffer, mergeSize, False)
            np.copyto(nextBuffer, n)

    return (buffer0, buffer1)

def main():
    listSize = 1 << 12
    iterations = 4

    testRun = readData()
    assert(testRun.size == (listSize * 2))

    sortedData = np.arange(0, listSize)
    unsortedData = np.flip(sortedData)

    gold0 = testRun[0:listSize]
    gold1 = testRun[listSize:listSize*2]
    
    (buf0, buf1) = mergeSort(unsortedData, 4, 4)

    print(buf0, buf1)
    print(gold0, gold1)

    assert(np.array_equal(buf0, gold0))
    assert(np.array_equal(buf1, gold1))

if __name__ == '__main__':
    main()
