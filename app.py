# -*- coding: utf-8 -*-
"""
Created on 

@author: 
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, confusion_matrix, classification_report

import os
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request
app = Flask(__name__)

#UPLOAD_FOLDER = 'uploads/'
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
ALLOWED_EXTENSIONS = set(['csv'])
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 3 * 1024 * 1024 * 1024
app.secret_key = 'super secret key'


def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def read_file(path):
    try:
        global df
        df = pd.read_csv(path)
        #print('Success')
        return df
    except:
        print('Could not read file.')
        
def dataset_overview(df):
    shape = df.shape
    rows, cols = shape[0], shape[1]
    missing_values_sum = df.isnull().sum().sum()
    desc_stat = df.describe().to_string().split('\n')
    return rows, cols, missing_values_sum, desc_stat

def linear_regression(X_train, X_test, y_train, y_test):
    lin_reg = LinearRegression()
    desc = lin_reg.fit(X_train, y_train)
    pred = lin_reg.predict(X_test)
    lin_cdf = pd.DataFrame({'Features': X_train.columns,
                        'Coefficients': lin_reg.coef_
                        }).sort_values(by='Coefficients', ascending=False).head(10)
    return desc, pred, lin_cdf

def decision_tree_regression(X_train, X_test, y_train, y_test):
    dt_reg = DecisionTreeRegressor()
    desc = dt_reg.fit(X_train, y_train)
    pred = dt_reg.predict(X_test)
    dt_imp =  pd.DataFrame({'Features': X_train.columns,
                            'Importance': dt_reg.feature_importances_
                            }).sort_values(by='Importance', ascending=False).head(10)
    return desc, pred, dt_imp

def random_forest_regression(X_train, X_test, y_train, y_test):
    rf_reg = RandomForestRegressor()
    desc = rf_reg.fit(X_train, y_train)
    pred = rf_reg.predict(X_test)
    rf_imp =  pd.DataFrame({'Features': X_train.columns,
                            'Importance': rf_reg.feature_importances_
                            }).sort_values(by='Importance', ascending=False).head(10)
    return desc, pred, rf_imp

def logistic_regression(X_train, X_test, y_train, y_test):
    log_reg = LogisticRegression()
    desc = log_reg.fit(X_train, y_train)
    pred = log_reg.predict(X_test)
    log_cdf =  pd.DataFrame({'Features': X_train.columns,
                            'Coefficients': log_reg.coef_[0]
                            }).sort_values(by='Coefficients', ascending=False).head(10)
    return desc, pred, log_cdf

def decision_tree_classifier(X_train, X_test, y_train, y_test):
    dt_clf = DecisionTreeClassifier()
    desc = dt_clf.fit(X_train, y_train)
    pred = dt_clf.predict(X_test)
    dtc_imp =  pd.DataFrame({'Features': X_train.columns,
                            'Importance': dt_clf.feature_importances_
                            }).sort_values(by='Importance', ascending=False).head(10)
    return desc, pred, dtc_imp

def random_forest_classifier(X_train, X_test, y_train, y_test):
    rf_clf = RandomForestClassifier()
    desc = rf_clf.fit(X_train, y_train)
    pred = rf_clf.predict(X_test)
    rfc_imp =  pd.DataFrame({'Features': X_train.columns,
                            'Importance': rf_clf.feature_importances_
                            }).sort_values(by='Importance', ascending=False).head(10)
    return desc, pred, rfc_imp

def regression_metrics(y_test, pred):
    mae = mean_absolute_error(y_test, pred)
    mse = mean_squared_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)
    return mae, mse, rmse, r2

def classification_metrics(y_test, pred):
    accuracy = accuracy_score(y_test, pred)
    conf_mat = confusion_matrix(y_test, pred)
    class_rep = classification_report(y_test, pred, output_dict=True)
    return accuracy, conf_mat, class_rep
        

@app.route('/results', methods=['POST'])
def show_results():
    if request.method == 'POST':
        models = request.form.getlist('models')
        test_split = float(request.form.getlist('split')[0])
    new_df = df[selected_cols]
    X = new_df.drop([target_col], axis=1)
    y = new_df[target_col]
    if y.dtype == 'object' and y.str.isdigit().all():
        y=y.astype('int')
    X = pd.get_dummies(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split, random_state=42)
    model_pkg = []
    
    for model in models:
        if model == 'Linear Regression':
            desc, pred, lin_cdf = linear_regression(X_train, X_test, y_train, y_test)
            lin_cdf = lin_cdf.to_html(classes='data table')
            mae, mse, rmse, r2 = regression_metrics(y_test, pred)
            model_pkg.append([desc, mae, mse, rmse, r2, lin_cdf])
            
        if model == 'Decision Tree Regressor':
            desc, pred, dt_imp = decision_tree_regression(X_train, X_test, y_train, y_test)
            dt_imp = dt_imp.to_html(classes='data table')
            mae, mse, rmse, r2 = regression_metrics(y_test, pred)
            model_pkg.append([desc, mae, mse, rmse, r2, dt_imp])
            
        if model == 'Random Forest Regressor':
            desc, pred, rf_imp = random_forest_regression(X_train, X_test, y_train, y_test)
            rf_imp = rf_imp.to_html(classes='data table')
            mae, mse, rmse, r2 = regression_metrics(y_test, pred)
            model_pkg.append([desc, mae, mse, rmse, r2, rf_imp])
            
        if model == 'Logistic Regression':
            desc, pred, log_cdf = logistic_regression(X_train, X_test, y_train, y_test)
            log_cdf = log_cdf.to_html(classes='data table')
            accuracy, conf_mat, class_rep = classification_metrics(y_test, pred)
            conf_mat = pd.DataFrame(conf_mat)
            conf_mat = conf_mat.to_html(classes='data table')
            class_rep_df = pd.DataFrame(class_rep).transpose().to_html(classes='data table')
            model_pkg.append([desc, accuracy, conf_mat, class_rep_df, log_cdf])
            
        if model == 'Decision Tree Classifier':
            desc, pred, dtc_imp = decision_tree_classifier(X_train, X_test, y_train, y_test)
            dtc_imp = dtc_imp.to_html(classes='data table')
            accuracy, conf_mat, class_rep = classification_metrics(y_test, pred)
            conf_mat = pd.DataFrame(conf_mat)
            conf_mat = conf_mat.to_html(classes='data table')
            class_rep_df = pd.DataFrame(class_rep).transpose().to_html(classes='data table')
            model_pkg.append([desc, accuracy, conf_mat, class_rep_df, dtc_imp])
            
        if model == 'Random Forest Classifier':
            desc, pred, rfc_imp = random_forest_classifier(X_train, X_test, y_train, y_test)
            rfc_imp = rfc_imp.to_html(classes='data table')
            accuracy, conf_mat, class_rep = classification_metrics(y_test, pred)
            conf_mat = pd.DataFrame(conf_mat)
            conf_mat = conf_mat.to_html(classes='data table')
            class_rep_df = pd.DataFrame(class_rep).transpose().to_html(classes='data table')
            model_pkg.append([desc, accuracy, conf_mat, class_rep_df, rfc_imp])
    
    if 'Linear Regression' in models or 'Decision Tree Regressor' in models \
    or 'Random Forest Regressor' in models:
        return render_template('aaml_metrics.html', train_len=len(X_train), test_len=len(X_test),
                               model_pkg=model_pkg, alert=1
                              )
    else:
        return render_template('aaml_metrics.html', train_len=len(X_train), test_len=len(X_test),
                               model_pkg=model_pkg, alert=0
                              )
    
@app.route('/buildML', methods=['POST'])
def model_building():
    if request.method == 'POST':
        mean_cols = request.form.getlist('mean')
        median_cols = request.form.getlist('median')
        mode_cols = request.form.getlist('mode')
        
        if len(mean_cols) > 0:
            for col in mean_cols:
                col = col.split('|')[0]
                df[col].fillna(df[col].mean(), inplace=True)
                
        if len(median_cols) > 0:
            for col in median_cols:
                col = col.split('|')[0]
                df[col].fillna(df[col].median(), inplace=True)
                
        if len(mode_cols) > 0:
            for col in mode_cols:
                col = col.split('|')[0]
                df[col].fillna(df[col].mode()[0], inplace=True)
          
        if df[target_col].nunique() > 20:
            suggested_ml_models = ['Linear Regression', 'Decision Tree Regressor',
                                   'Random Forest Regressor' 
                                  ]
        else:
            suggested_ml_models = ['Logistic Regression', 'Decision Tree Classifier',
                                   'Random Forest Classifier' 
                                  ]
        
        return render_template('aaml_model_build.html', models=suggested_ml_models)
        
@app.route('/preprocess', methods=['POST'])
def column_selection():
    if request.method == 'POST':
        col_names = df.columns.tolist()
        global selected_cols
        selected_cols = []
        mv_cols = []
        changed_cols = []
        dtypes_list = []
        for col in col_names:
            if request.form.get(col):
                selected_cols.append(col)
        
        for col in selected_cols:
            if df[col].isnull().sum() > 0:
                col_with_type = col + '| Type: ' + str(df[col].dtypes)
                mv_cols.append(col_with_type)
                
            if df[col].nunique() < 20 and df.dtypes[col] != 'object':
                dtypes_list.append(df.dtypes[col])
                df[col] = df[col].astype('object')
                changed_cols.append(col)
        global target_col
        target_col = request.form.getlist('target')[0]
        
        data = {'Columns': df[changed_cols].isnull().sum().index.values.tolist(),
                'Previous Data Type': dtypes_list,
                'New Data Type': df[changed_cols].dtypes.values.tolist()
               }
        ch_table = pd.DataFrame(data)
        
        return render_template('aaml_col_sel.html', selected_cols=mv_cols, changed_cols=ch_table.to_html(classes='data table'))


@app.route('/', methods=['POST'])
def upload_file():
    if request.method == 'POST':

        if 'file' not in request.files:
            return render_template('aaml_homepage.html', alert=0)
        file = request.files['file']
        
        if file.filename == '':
            return render_template('aaml_homepage.html', alert=0)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            #path_to_file = 'uploads/' + filename
            path_to_file =  os.path.join(os.getcwd(), 'uploads') + filename
            print(path_to_file)
            df = read_file(path_to_file)
            rows, cols, missing_values_sum, desc_stat = dataset_overview(df)
            col_names = df.columns.tolist()
            data = {'Columns': df.isnull().sum().index.values.tolist(),
                    'Missing Values': df.isnull().sum().values.tolist(),
                    'Data Type': df.dtypes.tolist()
                   }
            mv_table = pd.DataFrame(data)
            return render_template('aaml_homepage.html', alert=1, filename=filename, 
                                   rows=rows, cols=cols, col_names=col_names,
                                   mv_table=mv_table.to_html(classes='data table'),
                                   missing_values_sum=missing_values_sum,
                                   desc_table=df.describe().to_html(classes='data table', header='true')
                                  )
            
        else:
            return render_template('aaml_homepage.html', alert=0)


@app.route('/')
def homePage():
    return render_template('aaml_homepage.html')        

if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded=True)
