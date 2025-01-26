from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from flask import Flask,render_template,request
from sklearn import svm
import numpy as np

app=Flask(__name__)

groups={5:'very positive',4:'positive',3:'neutral',2:'negative',1:'very negative'}

classifier=svm.SVC(max_iter=5000)
sentiment_scores = np.array([
    [0.9, 0.05, 0.05, 0.75],  # خیلی مثبت
    [0.8, 0.1, 0.1, 0.70],  # مثبت
    [0.1, 0.7, 0.2, 0.50],  # خنثی
    [0.1, 0.2, 0.7, -0.60],  # منفی
    [0.05, 0.1, 0.85, -0.85],  # خیلی منفی
    [0.95, 0.03, 0.02, 0.90],  # خیلی مثبت
    [0.2, 0.6, 0.2, 0.60],  # خنثی
    [0.1, 0.2, 0.7, -0.60],  # منفی
    [0.7, 0.2, 0.1, 0.50],  # مثبت
    [0.15, 0.6, 0.25, 0.30],  # خنثی
    [0.92, 0.04, 0.04, 0.90],  # خیلی مثبت
    [0.85, 0.1, 0.05, 0.80],  # مثبت
    [0.2, 0.65, 0.15, 0.65],  # خنثی
    [0.12, 0.2, 0.68, -0.70],  # منفی
    [0.08, 0.12, 0.8, -0.90],  # خیلی منفی
    [0.94, 0.02, 0.04, 0.95],  # خیلی مثبت
    [0.22, 0.58, 0.2, 0.50],  # خنثی
    [0.15, 0.25, 0.6, -0.50],  # منفی
    [0.75, 0.18, 0.07, 0.70],  # مثبت
    [0.18, 0.62, 0.2, 0.20],  # خنثی
    [0.9, 0.05, 0.05, 0.85],  # خیلی مثبت
    [0.83, 0.1, 0.07, 0.70],  # مثبت
    [0.25, 0.5, 0.25, 0.45],  # خنثی
    [0.1, 0.3, 0.6, -0.50],  # منفی
    [0.06, 0.12, 0.82, -0.75],  # خیلی منفی
    [0.91, 0.03, 0.06, 0.92],  # خیلی مثبت
    [0.3, 0.4, 0.3, 0.55],  # خنثی
    [0.1, 0.22, 0.68, -0.55],  # منفی
    [0.8, 0.15, 0.05, 0.60],  # مثبت
    [0.2, 0.68, 0.12, 0.10]   # خنثی
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
    print(result)
    return render_template('Analyze.html',**{'neg':neg,'neu':neu,'pos':pos,'cluster':classify([result.get('pos'),result.get('neu'),result.get('neg'),
                                                                                               result.get('compound')])})

@app.route('/')
def index():
    return render_template('Index.html')



if __name__=='__main__':
    nltk.download('vader_lexicon')
    app.run('127.0.0.1','5000')