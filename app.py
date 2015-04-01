import csv
import json
from bpe.bpe_class import Preprocessor, ModelAggregator
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
            model = m.train_hour_weekday()
        if prediction_file and allowed_file(prediction_file.filename):
            bpe1 = Preprocessor(prediction_file)
            X_s = m.X_standardizer.transform(bpe1.X)            
            y_out_s = model.predict(X_s)
            y_out = m.y_standardizer.inverse_transform(y_out_s)
            rv = {}
            rv['y_predicted'] = y_out.tolist()
            rv['y_input'] = bpe0.y.flatten().tolist()
            return json.dumps(rv)

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
