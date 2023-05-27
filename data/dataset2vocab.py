import jieba

cutList = []

mySet = set()

with open('dataset.txt', 'r', encoding='utf-8') as f1:
    for i in f1.readlines():
        text = str(i).strip().split('\t')[1]
        # print(text)
        words = jieba.cut(text)

        cutWords = []
        for word in words:
            cutWords.append(word)

        cutList.append(cutWords)

i = 1

for cut in cutList:
    for word in cut:
        mySet.add(word)

print(mySet)

with open('vocab.txt', 'a', encoding='utf-8') as f:
    for item in mySet:
        print(item + '\t' + str(i))
        f.write(item + '\t' + str(i) + '\n')
        i = i + 1
