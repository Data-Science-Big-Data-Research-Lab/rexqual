import pandas as pd
import DL.LSTM
import DL.MLP
import DL.baseline_models
import XAI.SHAP
import XAI.evaluate
import DATA.Electricidad, DATA.SAGRA
import ETL.ETL
import XAI.RULEx, XAI.SHAP, XAI.LIME, XAI.rules, XAI.RANDOM
from tensorflow.keras.models import load_model, save_model

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    #1. Read data.

    #Multivariate example
    #d = pd.read_excel('data.xlsx')
    #d = DATA.SAGRA.prepare_data(d)
    #univariate = False

    #Univariate example
    d = pd.read_csv('data.csv')
    d = DATA.Electricidad.w_y_hr(d) #Specific preprocess for electric demand data.
    # Use your own preprocessing
    univariate = True

    dataset_tr, dataset_te, dataset_v = ETL.ETL.split_data(d, train_size=0.65, test_size=.25, val_size=.1)
    value_cols = dataset_tr.columns

    #Number of features in the time windoe W and to predict H
    #Configurate
    H = 24
    W = 168
    num_vars = 1 #this is univariate
    target_position = 1 ##starting at 1

    if univariate:
        xtrain, ytrain, scaler_tr = DATA.Electricidad.X_e_y(dataset_tr, W, H)
        xtest, ytest, scaler_te = DATA.Electricidad.X_e_y(dataset_te, W, H)
        xval, yval, scalerv = DATA.Electricidad.X_e_y(dataset_v, W, H)
        c = value_cols

    else:
        #Multivariate
        xtrain, ytrain, xtest, ytest, xval, yval, scaler_tr, scaler_te, c = ETL.ETL.all_prepared(dataset_tr, dataset_v, dataset_te, H, W, num_vars, target_position)

    # Add shape to use certain networks, as MLP or LSTM network
    xtrainL, ytrainL, xtestL, ytestL, xvalL, yvalL = DL.MLP.adaptShapesToLSTM(xtrain, xtest, ytrain, ytest, xval, yval, W, num_vars)

    #2. DL
    saved_model = True
    #Training and saving the DP model. If you already have one, you can load it

    if saved_model:
        modelo1 = load_model('modelo.h5')

    else:
        modelo1 = DL.LSTM.fit_lstm_model(xtrainL, ytrainL, xtestL, ytrainL, xvalL, yvalL, scaler_te, W, H, num_vars)
        p_df = DL.LSTM.evalua_LSTM(modelo1, xtestL, ytestL, scaler_te, H)
        modelo1, p_df = DL.baseline_models.SVM(xtrain, ytrain, xtest, ytest, xval, yval, scaler_te, W, H, num_vars)
        modelo1, p_df = DL.baseline_models.decision_tree(xtrain, ytrain, xtest, ytest, xval, yval, scaler_te, W, H, num_vars)
        modelo1, p_df = DL.baseline_models.random_forest_regresor(xtrain, ytrain, xtest, ytest, xval, yval, scaler_te, W, H, num_vars)
        modelo1 = DL.MLP.fit_MLP(xtrainL, ytrainL, xtestL, ytestL, xvalL, yvalL, scaler_te, W, H, num_vars)
        save_model(modelo1,'modelo.h5')

    # Saving predictions made using train set
    p_df = DL.MLP.evalua_MLP(modelo1, xtrainL, ytrainL, scaler_tr, H)
    p_df.columns = c
    archivo_predicciones = 'predictions.xlsx'
    p_df.to_excel(archivo_predicciones)
    archivo_predicciones = pd.read_excel(archivo_predicciones)


    ## 3. XAI
    opcion = 1 #Configuration of which XAI technique is executing
    ## 1 for RULEx
    ## 2 for SHAP
    ## 3 for Random
    ## 4 for all

    ##### XAI ######
    for i in range(H):
        print('H: ' + str(i))

        if (opcion == 1 or opcion == 4):
            ## RULEx
            path = "javaMOQAR"
            p = 'Elect' #folder
            r = '/home/lab08/Escritorio/'  #absolute path where the project is

            ##Assuming MOQAR has already been executed
            ## the rules are in the folder denoted in p
            df = XAI.RULEx.feature_importance_ranking(r, archivo_predicciones, p, H, num_vars*W, i)
            s = XAI.RULEx.importance_ranking(df) #PARA TODOS
            rankingMOQ = XAI.RULEx.ranking(s, c, num_vars*W)
            d_kMOQ, f = XAI.evaluate.features_shap(p_df, rankingMOQ, H, i)
            print('top RULEx features: ')
            print(rankingMOQ[0: len(f)])
            print('Number of RULEx top features: ' + str(len(f)))
            #print(d_kMOQ)
            d_kMOQ.to_csv('filtered_dataset' + str(i) +'.csv')
            d_k = d_kMOQ

            XAI.evaluate.evaluate(d_k, i, 'rulex', len(f))

        elif opcion == 2 or opcion == 4:
            ### SHAP
            rankingSHAP = XAI.SHAP.DeepSHAP(modelo1, xtrainL, c, num_vars*W, i) # DL models
            d_kSHAP, f = XAI.evaluate.features_shap(p_df, rankingSHAP, H, i)
            print('top SHAP features: ')
            print(rankingSHAP[0: len(f)])
            print('Number of SHAP top features: ' + str(len(f)))
            #print(d_kSHAP)
            d_kSHAP.to_csv('filtered_dataset' + str(i) +'.csv')
            d_k = d_kSHAP

            XAI.evaluate.evaluate(d_k, i, 'shap', len(f))


        elif opcion == 3 or opcion == 4:
            #Random
            a = W * num_vars
            ranking_r = XAI.RANDOM.XAI_random(c[0:a])
            ranking_r.to_excel('ranking_' + str(i) + '.xlsx')
            d_kR, f = XAI.evaluate.features_better_than(p_df, ranking_r, H, i) ##better than 0.75
            print('top random features: ')
            print(ranking_r[0: len(f)])
            print('Number of random top features: ' + str(len(f)))
            # print(d_kSHAP)
            d_kR.to_csv('filtered_dataset' + str(i) + '.csv')
            d_k = d_kR

            XAI.evaluate.evaluate(d_k, i, 'random', len(f))

