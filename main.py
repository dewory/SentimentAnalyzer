import jieba
import joblib
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QPalette
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QWidget, QVBoxLayout, \
    QHBoxLayout, QTextEdit
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
from wordcloud import WordCloud

classifier = joblib.load('./model/model.pkl')
vectorizer = joblib.load('./model/vectorizer.pkl')


def encode_text(texts):
    encoded_texts = []
    for text in texts:
        words = jieba.lcut(text)
        encoded_texts.append(' '.join(words))
    return encoded_texts


def predict_sentence(classifier, vectorizer, input_text):
    input_encoded = encode_text([input_text])
    input_vectorized = vectorizer.transform(input_encoded).toarray()
    prediction = classifier.predict(input_vectorized)
    return prediction[0]


class InitUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setFixedSize(2000, 1000)
        screen = QApplication.desktop().screenGeometry()
        self.move((screen.width() - self.width()) // 2, (screen.height() - self.height()) // 2)
        self.setWindowTitle('基于机器学习的情感分析应用')
        palette = QPalette()
        self.setPalette(palette)

        # 左侧
        left_widget = QWidget(self)
        left_layout = QVBoxLayout(left_widget)

        input_label = QLabel('待分析评论', left_widget)
        input_label.setFont(QFont("楷体", 20))
        input_layout = QHBoxLayout()
        input_layout.addWidget(input_label, alignment=Qt.AlignCenter)
        left_layout.addLayout(input_layout)

        self.input_edit = QTextEdit(left_widget)
        self.input_edit.setFont(QFont("楷体", 18))
        self.input_edit.textChanged.connect(self._on_text_changed)
        left_layout.addWidget(self.input_edit)

        # 中间
        middle_widget = QWidget(self)
        middle_layout = QVBoxLayout(middle_widget)

        predict_btn = QPushButton('分析', middle_widget)
        predict_btn.setFont(QFont("楷体", 20))
        predict_btn.setStyleSheet("QPushButton{background-color: blue; color:white; border-radius:10px;}")
        predict_btn.setFixedWidth(250)
        predict_btn.setFixedHeight(100)
        predict_btn.clicked.connect(self._on_predict_clicked)
        middle_layout.addWidget(predict_btn, alignment=Qt.AlignCenter)

        # 右侧
        right_widget = QWidget(self)
        right_layout = QVBoxLayout(right_widget)

        result_label = QLabel('分析结果', right_widget)
        result_label.setFont(QFont("楷体", 20))
        result_layout = QHBoxLayout()
        result_layout.addWidget(result_label, alignment=Qt.AlignCenter)
        right_layout.addLayout(result_layout)

        self.result_edit = QTextEdit(right_widget)
        self.result_edit.setFont(QFont("楷体", 18))
        self.result_edit.setReadOnly(True)
        right_layout.addWidget(self.result_edit)

        # 设置左中右的位置和大小
        left_width = 600
        middle_width = 200
        right_width = 600
        height = 900

        left_widget.setGeometry(0, 0, left_width, height)
        middle_widget.setGeometry(left_width, 0, middle_width, height)
        right_widget.setGeometry(left_width + middle_width, 0, right_width, height)

        # 将左中右添加到主窗口
        central_widget = QWidget(self)
        central_layout = QHBoxLayout(central_widget)
        central_layout.addWidget(left_widget)
        central_layout.addWidget(middle_widget)
        central_layout.addWidget(right_widget)

        self.setCentralWidget(central_widget)

    def _on_text_changed(self):
        self.result_edit.setText('')

    def _on_predict_clicked(self):
        lines = self.input_edit.toPlainText()
        if lines == '':
            self.result_edit.setText('输入为空。')
            return
        try:
            negative = 0
            positive = 0
            lineList = lines.split('\n')
            targetList = []
            for line in lineList:
                encoded_text = vectorizer.transform([line])
                predicted_label = int(classifier.predict(encoded_text)[0])
                if predicted_label == 0:
                    negative = negative + 1
                if predicted_label == 1:
                    positive = positive + 1
                sentiment_dict = {0: '消极的', 1: '积极的'}
                predicted_label = sentiment_dict[predicted_label]
                targetList.append(predicted_label)
            result = '\n'.join([item for item in targetList])

            font_file = 'font/SimHei.ttf'

            wc = WordCloud(font_path=font_file, background_color='white')
            wc.generate(' '.join(list(jieba.cut(''.join([item for item in lineList])))))

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))  # 创建两个子图
            fig.suptitle('情感分析可视化', fontsize=24, fontweight='bold',
                         fontproperties=FontProperties(fname=font_file))
            ax1.imshow(wc)
            ax1.axis('off')

            # 显示柱状图
            x_pos = [0, 1]
            y_pos = [negative, positive]
            colors = ['red', 'green']
            ax2.bar(x_pos, y_pos, color=colors)
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(['消极的', '积极的'], fontproperties=FontProperties(fname=font_file, size=20))
            ax2.set_ylabel('数量', fontproperties=FontProperties(fname=font_file, size=20))

            for i, v in enumerate(y_pos):
                ax2.text(i, v + 0.5, str(v), ha='center', fontproperties=FontProperties(fname=font_file, size=20))

            # 显示图表
            plt.show()

        except ValueError as e:
            result = '未知的'
            print('ValueError:\n', e)
        self.result_edit.setText(result)


if __name__ == '__main__':
    app = QApplication([])
    win = InitUI()
    win.show()
    app.exec_()
