import jieba
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB


# 加载数据
def load_data(filename):
    texts = []
    labels = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            label, text = line.strip().split('\t')
            texts.append(text)
            labels.append(label)
    return texts, labels


# 对文本进行分词和编码
def encode_text(texts):
    encoded_texts = []
    for text in texts:
        words = jieba.lcut(text)
        encoded_texts.append(' '.join(words))
    return encoded_texts


# 训练模型
def train_model(x_train, y_train):
    classifier = MultinomialNB()
    param_grid = {'alpha': [0.1, 0.5, 1, 2, 5, 10]}
    grid_search = GridSearchCV(classifier, param_grid, cv=5)
    grid_search.fit(x_train, y_train)
    classifier = grid_search.best_estimator_
    return classifier


# 测试模型
def test_model(classifier, x_test, y_test):
    y_pred = classifier.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy


def predict_sentence(classifier, vectorizer, input_text):
    input_encoded = encode_text([input_text])
    input_vectorized = vectorizer.transform(input_encoded).toarray()
    prediction = classifier.predict(input_vectorized)
    return prediction[0]


if __name__ == '__main__':
    # 加载训练数据和情感词汇表
    texts, labels = load_data('./data/dataset.txt')
    vocab = {}
    with open('./data/vocab.txt', 'r', encoding='utf-8') as f:
        for line in f:
            word, index = line.strip().split('\t')
            vocab[word] = int(index)

    # 对文本进行编码和填充，拆分训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(
        encode_text(texts), labels, test_size=0.2, random_state=123)

    # 使用TF-IDF作为特征选择方法
    vectorizer = TfidfVectorizer(stop_words=None, analyzer='word', token_pattern=r'\w{1,}', max_df=0.8,
                                 min_df=2, use_idf=True, smooth_idf=True, sublinear_tf=True)
    x_train = vectorizer.fit_transform(x_train).toarray()
    x_test = vectorizer.transform(x_test).toarray()

    # 训练模型并输出准确率
    classifier = train_model(x_train, y_train)
    train_accuracy = test_model(classifier, x_train, y_train)
    test_accuracy = test_model(classifier, x_test, y_test)
    print('训练准确率: {:.2f}%'.format(train_accuracy * 100))
    print('测试准确率: {:.2f}%'.format(test_accuracy * 100))
    joblib.dump(classifier, './model/model.pkl')
    joblib.dump(vectorizer, './model/vectorizer.pkl')
    print('模型保存成功!')

    # 预测句子示例
    input_text = '房间环境很差'
    sentiment_dict = {'0': '消极的', '1': '积极的'}
    predicted_label = predict_sentence(classifier, vectorizer, input_text)
    predicted_label = sentiment_dict[predicted_label]
    print('输入句子: "{}"'.format(input_text))
    print('预测结果: "{}"'.format(predicted_label))
