import csv
import json
from mave.core import Preprocessor, ModelAggregator
from flask import Flask, request, render_template
from flask.ext.bower import Bower

ALLOWED_EXTENSIONS = set(['csv'])

STATIC_URL_PATH = '/static/'

app = Flask(__name__, static_url_path=STATIC_URL_PATH)

# Enable bower static urls
Bower(app)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        training_file = request.files['file']
        if training_file and allowed_file(training_file.filename):
            p0 = Preprocessor(training_file)
            m = ModelAggregator(p0)
            model = m.train_hour_weekday()

        #if prediction_file and allowed_file(prediction_file.filename):
        #    p1 = Preprocessor(prediction_file)
        #    X_s = m.X_standardizer.transform(p1.X)
        #    y_out_s = model.predict(X_s)
        #    y_out = m.y_standardizer.inverse_transform(y_out_s)
        #    rv = {}
        #    to_dict = lambda t: {'datetime': t[0], 'value': t[1]}
        #    p0.datetimes  = map(lambda d: 1000 * d.strftime('%s'), p0.datetimes)
        #    p1.datetimes  = map(lambda d: 1000 * d.strftime('%s'), p1.datetimes)
        #    rv['y_predicted'] = map(to_dict, zip(p1.datetimes, y_out.tolist()))
        #    rv['y_input'] = map(to_dict, zip(p0.datetimes, p0.y.flatten().tolist()))
        #    return render_template('result.html', data=rv)

    return render_template('index.html', data=None)

if __name__ == '__main__':
    app.run(debug=True)
