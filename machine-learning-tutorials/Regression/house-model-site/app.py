from flask import Flask, request, render_template

import pickle
import numpy as np
import locale 

app = Flask(__name__)

@app.route('/')  #if the person write the website names --> on the browser --> lead him/her here
def home():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])

def predict():
#---fetch user parameters
    house_age =  float(request.form['house_age']) #age of the house
    dist_nStation = float(request.form['num_stores']) #distance to the nearest store km
    num_cStores = float(request.form['dist_nStores']) #number of public amenities around it

#--load the model
    pickled_model = pickle.load(open('linear_model.pkl','rb'))
    X_i = np.array([house_age,dist_nStation,num_cStores]).reshape(1,-1)
    price_pred = pickled_model.predict(X_i)
    price_pred = round(price_pred[0][0]*1000,0)#in thousands of R

#---number formatting
    locale.setlocale(locale.LC_ALL, '')
    formatted_number = locale.format_string("%d", price_pred, grouping=True)        
#-------
    return render_template('index.html', prediction=formatted_number)
    
if __name__ == '__main__':
    app.run(debug=True, port=8001)