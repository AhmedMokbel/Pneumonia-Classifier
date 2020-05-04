import os
from flask import Flask, render_template, redirect, url_for, flash 
from flask_bootstrap import Bootstrap
from tensorflow.keras.models import load_model 
from tensorflow.keras.preprocessing import  image
import numpy as np
from flask_wtf import FlaskForm
from wtforms import SubmitField
from flask_wtf.file import FileField, FileAllowed, FileRequired


app = Flask(__name__)
app.config['SECRET_KEY'] = 'hard to guess string'
app.config['UPLOAD_FOLDER'] = "UPLOAD_FOLDER"




# Bind to PORT if defined, otherwise default to 5000.
port = int(os.environ.get('PORT', 5000))

bootstrap = Bootstrap(app)

def init():
  global model
  model=load_model("model.hdf5")
   
class FileUploadForm(FlaskForm):
    fileName = FileField('photo', validators=[
        FileRequired(),
        FileAllowed(['png', 'pdf','jpeg' ,'jpg'], "wrong format!")
    ])
    submit = SubmitField('classify')

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404


@app.errorhandler(500)
def internal_server_error(e):
    return render_template('500.html'), 500


@app.route('/', methods=['GET', 'POST'])
def index():
  
    form = FileUploadForm()
    photo=form.fileName.data
    if form.validate_on_submit():
        
        test_image=image.load_img(photo , target_size=(150,150))
        test_image=image.img_to_array(test_image)
        test_image=np.expand_dims(test_image ,axis=0)
        score=model.predict(test_image)  
        """
        img = Image.open(name.stream).convert("L")
        img = np.resize(img, (150,150,3))        
        im2arr = np.array(img)
        im2arr = im2arr.reshape(1,150,150,3)
        score = model.predict(im2arr)
        """
        if score[0][0] == 0:
           cond = 'Normal'
           category = 'success'
        else :
           cond = 'pneumonia'
           category = 'danger'
           
        flash(' condition of x-ray :' +  cond  , category)
        return redirect(url_for('index'))
    return render_template('index.html', form=form)


if __name__ == '__main__':
    init()
    app.run(host='0.0.0.0', port=port)



