from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from flask import Flask,render_template,request
from sklearn import svm
import numpy as np

app=Flask(__name__)

groups={5:'very positive',4:'positive',3:'neutral',2:'negative',1:'very negative'}

classifier=svm.SVC()
sentiment_scores =np.array([
        [0.9, 0.05, 0.05],  # خیلی مثبت
        [0.8, 0.1, 0.1],  # مثبت
        [0.1, 0.7, 0.2],  # خنثی
        [0.1, 0.2, 0.7],  # منفی
        [0.05, 0.1, 0.85],  # خیلی منفی
        [0.95, 0.03, 0.02],  # خیلی مثبت
        [0.2, 0.6, 0.2],  # خنثی
        [0.1, 0.2, 0.7],  # منفی
        [0.7, 0.2, 0.1],  # مثبت
        [0.15, 0.6, 0.25],  # خنثی
        [0.92, 0.04, 0.04],  # خیلی مثبت
        [0.85, 0.1, 0.05],  # مثبت
        [0.2, 0.65, 0.15],  # خنثی
        [0.12, 0.2, 0.68],  # منفی
        [0.08, 0.12, 0.8],  # خیلی منفی
        [0.94, 0.02, 0.04],  # خیلی مثبت
        [0.22, 0.58, 0.2],  # خنثی
        [0.15, 0.25, 0.6],  # منفی
        [0.75, 0.18, 0.07],  # مثبت
        [0.18, 0.62, 0.2],  # خنثی
        [0.9, 0.05, 0.05],  # خیلی مثبت
        [0.83, 0.1, 0.07],  # مثبت
        [0.25, 0.5, 0.25],  # خنثی
        [0.1, 0.3, 0.6],  # منفی
        [0.06, 0.12, 0.82],  # خیلی منفی
        [0.91, 0.03, 0.06],  # خیلی مثبت
        [0.3, 0.4, 0.3],  # خنثی
        [0.1, 0.22, 0.68],  # منفی
        [0.8, 0.15, 0.05],  # مثبت
        [0.2, 0.68, 0.12]  # خنثی
    ])
sentiment_labels =np.array([
        5, 4, 3, 2, 1, 5, 3, 2, 4, 3,
        5, 4, 3, 2, 1, 5, 3, 2, 4, 3,
        5, 4, 3, 2, 1, 5, 3, 2, 4, 3
    ])
classifier.fit(sentiment_scores,sentiment_labels)


def classify(array):
    result=classifier.predict(np.array([array]))
    return groups[result[0]]



def analyze_sentiment(input):

    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(input)  # شامل مثبت، منفی، خنثی و compound
    return scores  # امتیاز کلی

@app.route('/analyze',methods=['POST'])
def analyze():
    phrase=request.form.get('phrase')
    result=analyze_sentiment(phrase)
    neg=result.get('neg')*100
    neu=neg+result.get('neu')*100
    pos=neu+result.get('pos')*100
    return render_template('Analyze.html',**{'neg':neg,'neu':neu,'pos':pos,'cluster':classify([result.get('pos'),result.get('neu'),result.get('neg')])})

@app.route('/')
def index():
    return render_template('Index.html')



if __name__=='__main__':
    nltk.download('vader_lexicon')
    app.run('127.0.0.1','5000')