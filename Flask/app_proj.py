from flask import Flask, jsonify, request, render_template, session, redirect
import pandas as pd
import numpy as np
import os,gc
import xgboost as xgb
import time
import joblib
import flask
from io import StringIO

def data_preprocess(test_df):
    train_cols = ["TransactionID_x", "TransactionDT", "TransactionAmt", "ProductCD", "card1", "card2", "card3", "card4", "card5", "card6", "addr1", "addr2", "dist1", "dist2", "P_emaildomain", "R_emaildomain", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10", "C11", "C12", "C13", "C14", "D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9", "D10", "D11", "D12", "D13", "D14", "D15", "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10", "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28", "V29", "V30", "V31", "V32", "V33", "V34", "V35", "V36", "V37", "V38", "V39", "V40", "V41", "V42", "V43", "V44", "V45", "V46", "V47", "V48", "V49", "V50", "V51", "V52", "V53", "V54", "V55", "V56", "V57", "V58", "V59", "V60", "V61", "V62", "V63", "V64", "V65", "V66", "V67", "V68", "V69", "V70", "V71", "V72", "V73", "V74", "V75", "V76", "V77", "V78", "V79", "V80", "V81", "V82", "V83", "V84", "V85", "V86", "V87", "V88", "V89", "V90", "V91", "V92", "V93", "V94", "V95", "V96", "V97", "V98", "V99", "V100", "V101", "V102", "V103", "V104", "V105", "V106", "V107", "V108", "V109", "V110", "V111", "V112", "V113", "V114", "V115", "V116", "V117", "V118", "V119", "V120", "V121", "V122", "V123", "V124", "V125", "V126", "V127", "V128", "V129", "V130", "V131", "V132", "V133", "V134", "V135", "V136", "V137", "V138", "V139", "V140", "V141", "V142", "V143", "V144", "V145", "V146", "V147", "V148", "V149", "V150", "V151", "V152", "V153", "V154", "V155", "V156", "V157", "V158", "V159", "V160", "V161", "V162", "V163", "V164", "V165", "V166", "V167", "V168", "V169", "V170", "V171", "V172", "V173", "V174", "V175", "V176", "V177", "V178", "V179", "V180", "V181", "V182", "V183", "V184", "V185", "V186", "V187", "V188", "V189", "V190", "V191", "V192", "V193", "V194", "V195", "V196", "V197", "V198", "V199", "V200", "V201", "V202", "V203", "V204", "V205", "V206", "V207", "V208", "V209", "V210", "V211", "V212", "V213", "V214", "V215", "V216", "V217", "V218", "V219", "V220", "V221", "V222", "V223", "V224", "V225", "V226", "V227", "V228", "V229", "V230", "V231", "V232", "V233", "V234", "V235", "V236", "V237", "V238", "V239", "V240", "V241", "V242", "V243", "V244", "V245", "V246", "V247", "V248", "V249", "V250", "V251", "V252", "V253", "V254", "V255", "V256", "V257", "V258", "V259", "V260", "V261", "V262", "V263", "V264", "V265", "V266", "V267", "V268", "V269", "V270", "V271", "V272", "V273", "V274", "V275", "V276", "V277", "V278", "V279", "V280", "V281", "V282", "V283", "V284", "V285", "V286", "V287", "V288", "V289", "V290", "V291", "V292", "V293", "V294", "V295", "V296", "V297", "V298", "V299", "V300", "V301", "V302", "V303", "V304", "V305", "V306", "V307", "V308", "V309", "V310", "V311", "V312", "V313", "V314", "V315", "V316", "V317", "V318", "V319", "V320", "V321", "V322", "V323", "V324", "V325", "V326", "V327", "V328", "V329", "V330", "V331", "V332", "V333", "V334", "V335", "V336", "V337", "V338", "V339", "TransactionID_y", "id_01", "id_02", "id_03", "id_04", "id_05", "id_06", "id_07", "id_08", "id_09", "id_10", "id_11", "id_12", "id_13", "id_14", "id_15", "id_16", "id_17", "id_18", "id_19", "id_20", "id_21", "id_22", "id_23", "id_24", "id_25", "id_26", "id_27", "id_28", "id_29", "id_30", "id_31", "id_32", "id_33", "id_34", "id_35", "id_36", "id_37", "id_38", "DeviceType", "DeviceInfo" ]
    print(len(train_cols))
    test_df.columns = train_cols
    red_col = ['D1', 'D10', 'D11', 'D13', 'D14', 'D15', 'D5', 'D6', 'V1', 'V104', 'V107', 'V108', 'V11', 'V111', 'V115', 'V117', 'V120', 'V121', 'V123', 'V124', 'V127', 'V129', 'V13', 'V130', 'V136', 'V138', 'V139', 'V14', 'V142', 'V147', 'V156', 'V160', 'V162', 'V165', 'V166', 'V169', 'V17', 'V171', 'V173', 'V175', 'V176', 'V178', 'V180', 'V182', 'V185', 'V187', 'V188', 'V198', 'V20', 'V203', 'V205', 'V207', 'V209', 'V210', 'V215', 'V218', 'V220', 'V221', 'V223', 'V224', 'V226', 'V228', 'V229', 'V23', 'V234', 'V235', 'V238', 'V240', 'V250', 'V252', 'V253', 'V257', 'V258', 'V26', 'V260', 'V261', 'V264', 'V266', 'V267', 'V27', 'V271', 'V274', 'V277', 'V281', 'V283', 'V284', 'V285', 'V286', 'V289', 'V291', 'V294', 'V296', 'V297', 'V3', 'V30', 'V301', 'V303', 'V305', 'V307', 'V309', 'V310', 'V314', 'V320', 'V325', 'V332', 'V335', 'V338', 'V36', 'V37', 'V4', 'V40', 'V41', 'V44', 'V47', 'V48', 'V54', 'V55', 'V56', 'V59', 'V6', 'V62', 'V65', 'V67', 'V68', 'V70', 'V76', 'V78', 'V8', 'V80', 'V82', 'V86', 'V88', 'V89', 'V91', 'V96', 'V98', 'V99']
    # droping v cols 
    drop_v_cols = [col for col in test_df.columns if col[0] == 'V' and col not in red_col]
    drop_d_cols = [col for col in test_df.columns if col[0] == 'D' and len(col) <= 3 and col not in red_col]

    drop_cols = []
    drop_cols.extend(drop_v_cols)
    #drop_cols.extend(drop_d_cols)
    print(f'dropping {len(drop_cols)} columns')
    test_df = test_df.drop(columns=drop_cols)

    eps = 0.001 # 0 => 0.1¢
    test_df['Log Ammount'] = np.log(test_df.pop('TransactionAmt')+eps)

    useful_cols = [col for col in test_df.columns if col not in ("isFraud","TransactionID_x", "TransactionID_y", "TransactionDT")]
    print(len(useful_cols))
    test_df = test_df[useful_cols].reset_index(drop=True)

    category_cols = test_df.select_dtypes(include=['object']).columns
    type_map = {c: str for c in category_cols}
    test_df[category_cols] = test_df[category_cols].astype(type_map, copy=False)

    for col in category_cols:
        # label encode all cat columns
        dff = pd.concat([test_df[col]])
        dff,_ = pd.factorize(dff,sort=True)
        if dff.max()>32000: 
            print(col,'needs int32 datatype')
        test_df[col] = dff[:len(test_df)].astype('int16')

    # Scaling numeric features
    for col in useful_cols:
        if col not in category_cols:
            # min max scalar
            dff = pd.concat([test_df[col]])
            dff = (dff - dff.min())/(dff.max() - dff.min())
            dff.fillna(-1,inplace=True)
            test_df[col] = dff[:len(test_df)]
    
    del dff

    print(f'fitting model on {len(useful_cols)} columns')
    test_df.fillna(-1,inplace=True)

    return test_df



app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/index')
def index():
    return flask.render_template('index_proj.html')


@app.route('/predict', methods=['POST'])
def predict():
    xg_mdl = joblib.load('xgb_fe.pkl')
    Tran = request.files['Tran']
    s_Tran = str(Tran.read(),'utf-8')
    Tran_data = StringIO(s_Tran)
    Tran_data = pd.read_csv(Tran_data)
    ID = request.files['ID']
    s_ID = str(ID.read(),'utf-8')
    ID_data = StringIO(s_ID)
    ID_data = pd.read_csv(ID_data)

    train_data = Tran_data.merge(ID_data, how='left', left_index=True, right_index=True)
    print("Training Merged Data Shape:", train_data.shape)    
    x_test = data_preprocess(train_data)    
    x_TrxID = train_data.pop("TransactionID_x")
    print(x_test.head())
    y_pred_test = xg_mdl.predict_proba(x_test)
    submission = {}
    submission.update(dict(zip(x_TrxID.values,y_pred_test)))
    submission = pd.DataFrame.from_dict(submission, orient="index").reset_index()
    submission.columns = ["TransactionID", "isFraud-No", "isFraud-yes"]
    print(submission.head())
    return submission.to_html(header="true", table_id="table")
    #return render_template('output.html', tables = [submission.to_html(classes='data')], header = "true") 
    #return jsonify({'prediction probabilities': submission.to_json(orient="index")})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
