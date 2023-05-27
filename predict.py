import joblib

from train import encode_text

classifier = joblib.load('./model/model.pkl')
vectorizer = joblib.load('./model/vectorizer.pkl')


def predict_sentence(classifier, vectorizer, input_text):
    input_encoded = encode_text([input_text])
    input_vectorized = vectorizer.transform(input_encoded).toarray()
    prediction = classifier.predict(input_vectorized)
    return prediction[0]


input_text = '环境真不好'
prediction = predict_sentence(classifier, vectorizer, input_text)
print('输入句子: "{}"'.format(input_text))
print('预测结果: "{}"'.format(prediction))
