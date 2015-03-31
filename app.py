import csv
from bpe_class import Preprocessor, ModelAggregator
from flask import Flask, request

ALLOWED_EXTENSIONS = set(['csv'])

STATIC_URL_PATH = '/static/'

app = Flask(__name__, static_url_path=STATIC_URL_PATH)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        training_file = request.files['training_file']
        prediction_file = request.files['prediction_file']
        if training_file and allowed_file(training_file.filename):
            bpe0 = Preprocessor(training_file)
            m = ModelAggregator(bpe0)
            m.train_dummy()
            print map(lambda m: m.best_score_, m.models)
        if prediction_file and allowed_file(prediction_file.filename):
            bpe1 = Preprocessor(prediction_file)
            m.
            print bpe1.training_data

    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form action="" method=post enctype=multipart/form-data>
      <p><input type=file name="training_file">
      <p><input type=file name="prediction_file">
      <input type=submit value=Upload>
    </form>
    '''

if __name__ == '__main__':
    app.run(debug=True)
