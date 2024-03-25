from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
import numpy as np
import pickle
app = Flask('__name__')
app.config['SECRET_KEY'] = 'e3d41a799ff8899137a01b827fa02365'

lr_model = pickle.load(open('./models/models.pkl', 'rb'))


@app.route("/")
def home():
    return render_template(template_name_or_list='index.html')


@app.route("/predict", methods=['POST'])
def predict():
    form = request.form
    if request.method == 'POST' and form.get('sepal-length')!='' and form.get('sepal-width')!='' and form.get('petal-length')!='' and form.get('petal-width')!='':
        print(form)
        test_data = [float(x) for x in form.values() if x != '']
        print(test_data , type(test_data))
        # converting into ndarray
        test_data_array = np.array(test_data)
        print(np.reshape(test_data_array, (-1,4)))
        prediction = lr_model.predict(np.reshape(test_data_array, (-1,4)))
        return render_template(template_name_or_list='predict.html', form=form, prediction=prediction)
    else:
        flash('Please enter values for all the fields in form ....')
        return redirect(url_for('home'))


if __name__ == '__main__':
    app.run(debug=True)
