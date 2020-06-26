# Indraneel

#Library Import

import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# data load
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('Disasters', engine)

# model load
model = joblib.load("../models/classifier.pkl")


@app.route('/')
@app.route('/index')
def index():
    
    # extract data for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    cats = df[df.columns[5:]]
    cats_counts = cats.mean()*cats.shape[0]
    cats_names = list(cats_counts.index)
    nlarge_counts = cats_counts.nlargest(5)
    nlarge_names = list(nlarge_counts.index)
    # visual creation

    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=nlarge_names,
                    y=nlarge_counts
                )
            ],

            'layout': {
                'title': 'Top message categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=cats_names,
                    y=cats_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        }
        
    ]
    
    
    
    # encode plotly graphs 
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page 
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page to handles user query, displays model results
@app.route('/go')
def go():
    # save user input 
    query = request.args.get('query', '') 

    # use model to predict
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This renders the go.html  
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()