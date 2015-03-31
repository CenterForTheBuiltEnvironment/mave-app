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
        _file = request.files['file']
        if _file and allowed_file(_file.filename):
            bpe0 = Preprocessor(_file)
            m = ModelAggregator(bpe0)
            m.train_all()
            print map(lambda m: m.best_score_, m.models)

    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form action="" method=post enctype=multipart/form-data>
      <p><input type=file name=file>
         <input type=submit value=Upload>
    </form>
    '''

if __name__ == '__main__':
    app.run(debug=True)
