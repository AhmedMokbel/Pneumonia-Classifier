import os
from flask import Flask, render_template, session, redirect, url_for, flash ,request
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from wtforms import SubmitField,FileField
from wtforms.validators import DataRequired
from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)
app.config['SECRET_KEY'] = 'hard to guess string'

# Bind to PORT if defined, otherwise default to 5000.
port = int(os.environ.get('PORT', 5000))

bootstrap = Bootstrap(app)

def init():
   global model ,graph
   model = load_model('model.hdf5')
   graph = tf.compat.v1.get_default_graph()
   
class NameForm(FlaskForm):
    name = FileField('upload image', validators=[DataRequired()])
    submit = SubmitField('Classify')


@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404


@app.errorhandler(500)
def internal_server_error(e):
    return render_template('500.html'), 500


@app.route('/', methods=['GET', 'POST'])
def index():
    global score ,im2arr ,img
    form = NameForm()
    name=form.name.data
    if request.method == 'POST':
      if form.validate_on_submit():
        img = Image.open(name.stream).convert("L")
        img = np.resize(img, (150,150,3))        
        im2arr = np.array(img)
        im2arr = im2arr.reshape(1,150,150,3)
        score = model.predict(im2arr)
        with graph.as_default():
         if score[0][0] == 1:
           cond = 'Normal'
           category = 'success'
         else :
           cond = 'pneumonia'
           category = 'danger'
           
        flash(' condition of x-ray image :' + cond + ')', category)
        session['name'] = form.name.data
        return redirect(url_for('index'))
    return render_template('index.html', form=form, name=session.get('name'))


if __name__ == '__main__':
    init()
    app.run(host='0.0.0.0', port=port)
    
