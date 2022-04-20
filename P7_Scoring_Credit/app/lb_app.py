import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import shap
import plotly.express as px
from sklearn.neural_network import MLPClassifier
from zipfile import ZipFile
from sklearn.cluster import KMeans
plt.style.use('fivethirtyeight')
#sns.set_style('darkgrid')



def main() :

    @st.cache
    def load_data():
        z = ZipFile("data/default_risk.zip")
        data = pd.read_csv(z.open('default_risk.csv'), index_col='SK_ID_CURR', encoding ='utf-8')

        z = ZipFile("data/X_sample.zip")
        sample = pd.read_csv(z.open('X_sample.csv'), index_col='SK_ID_CURR', encoding ='utf-8')
        
        description = pd.read_csv("data/features_description.csv", 
                                  usecols=['Row', 'Description'], index_col=0, encoding= 'unicode_escape')

        target = data.iloc[:, -1:]

        return data, sample, target, description


    def load_model():
        '''loading the trained model'''
        pickle_in = open('model/LGBMClassifier.pkl', 'rb') # MLPClassifier
        clf = pickle.load(pickle_in)
        return clf


    @st.cache(allow_output_mutation=True)
    def load_knn(sample):
        knn = knn_training(sample)
        return knn


    @st.cache
    def load_infos_gen(data):
        lst_infos = [data.shape[0],
                     round(data["AMT_INCOME_TOTAL"].mean(), 2),
                     round(data["AMT_CREDIT"].mean(), 2)]

        nb_credits = lst_infos[0]
        rev_moy = lst_infos[1]
        credits_moy = lst_infos[2]

        targets = data.TARGET.value_counts()

        return nb_credits, rev_moy, credits_moy, targets


    def identite_client(data, id):
        data_client = data[data.index == int(id)]
        return data_client

    @st.cache
    def load_age_population(data):
        data_age = round((data["DAYS_BIRTH"]/365), 2)
        return data_age

    @st.cache
    def load_income_population(sample):
        df_income = pd.DataFrame(sample["AMT_INCOME_TOTAL"])
        df_income = df_income.loc[df_income['AMT_INCOME_TOTAL'] < 200000, :]
        return df_income

    @st.cache
    def load_prediction(sample, id, clf):
        X=sample.iloc[:, :-1]
        score = clf.predict_proba(X[X.index == int(id)])[:,1]
        return score

    @st.cache
    def load_kmeans(sample, id, mdl):
        index = sample[sample.index == int(id)].index.values
        index = index[0]
        data_client = pd.DataFrame(sample.loc[sample.index, :])
        df_neighbors = pd.DataFrame(knn.fit_predict(data_client), index=data_client.index)
        df_neighbors = pd.concat([df_neighbors, data], axis=1)
        return df_neighbors.iloc[:,1:].sample(10)

    @st.cache
    def knn_training(sample):
        knn = KMeans(n_clusters=2).fit(sample)
        return knn 



    #Loading data……
    data, sample, target, description = load_data()
    id_client = sample.index.values
    clf = load_model()

# Charte et CSS



    st.markdown("<head><style>a { text-decoration: none;} .header {color: maroon} </style></a>", unsafe_allow_html=True)
  
    #Title display
    st.markdown("<table border=1 align=center><tr><td bgcolor=#f7d6a6><font size=+7 color=Maroon><b>Dashboard Scoring Credit</b></td></font></tr></table>", unsafe_allow_html=True)
    

    
       
      
    

    #######################################
    # SIDEBAR
    #######################################

    #Selection du client
    st.sidebar.header("Information générale")

    # Menu navigation
    chk_id = st.sidebar.selectbox("Client ID", id_client)
    st.sidebar.markdown("&nbsp; Ce client est <font size=+4 color=red><b>Non Solvable</b><font>" , unsafe_allow_html=True) 
    st.sidebar.markdown("<u><b>Détais de la décisions</u></b> :", unsafe_allow_html=True)
    st.sidebar.markdown("<a href='#S1'><font color=maroon>1 - Score et Solvabilité du client</font></a>" , unsafe_allow_html=True)
    st.sidebar.markdown("<a href='#S2'><font color=maroon>2 - Informations descriptives du client</font></a>", unsafe_allow_html=True)
    st.sidebar.markdown("<a href='#S2.1'><font color=maroon>&nbsp;&nbsp;&nbsp;2.1 - Détails client</font></a>", unsafe_allow_html=True)
    st.sidebar.markdown("<a href='#S2.2'><font color=maroon>&nbsp;&nbsp;&nbsp;2.2 - Revenus client</font></a>", unsafe_allow_html=True)
    st.sidebar.markdown("<a href='#S2.3'><font color=maroon>&nbsp;&nbsp;&nbsp;2.3 - Données du client</font></a>", unsafe_allow_html=True)
    st.sidebar.markdown("<a href='#S3'><font color=maroon>3 - les caractérisques importantes </font></a>", unsafe_allow_html=True)    
    st.sidebar.markdown("<a href='#S4'><font color=maroon>4 - Clients simailaire</font></a>", unsafe_allow_html=True) 
    st.sidebar.markdown("<a href='#S5'><font color=maroon>5 - Interprétabilité des résultats</font></a>", unsafe_allow_html=True)
        

    st.markdown("<br><br>", unsafe_allow_html=True) 
    st.write("Client séléctionné :", chk_id)

                    
    
                      
                             
                             

    # Solvabilité du client
    st.markdown("<a name='S1'>", unsafe_allow_html=True) 
    st.markdown("<font color=maroon size=+5><b>1 - Score et Solvabilité du client</b></font>", unsafe_allow_html=True) 
    st.markdown("** Le score reflète la pertinence de la décision.., de 1 à 10 **")
    st.markdown("<font color=maroon size=+3><b> <u>Solvabilité du client</u></b></font>", unsafe_allow_html=True) 
    prediction = load_prediction(sample, chk_id, clf)
    st.write("** Score : **{:.0f} ".format(round(float(prediction)*10, 1)))

    # Détails client  : Sexe , âge, situation de famille, enfants, …
    st.markdown("<a name='S2'>", unsafe_allow_html=True) 
    st.markdown("<font color=maroon size=+5><b>2 - Informations descriptives du client</b></font>", unsafe_allow_html=True) 

    # Info client :
    st.markdown("<a name='S2.1'>", unsafe_allow_html=True) 
    st.markdown("<font color=maroon size=+3><b>2.1 <u>Détails client</u></b></font>", unsafe_allow_html=True) 
    infos_client = identite_client(data, chk_id)
    st.markdown("<b>Nom : </b>XXXXX   - <b>Prénom :</b> XXXXXXX", unsafe_allow_html=True)
    st.write('**Sexe : **', infos_client["CODE_GENDER"].values[0])
    st.write("**Age : **{:.0f} ans".format(int(infos_client["DAYS_BIRTH"]/365)))
    st.write("** Status : **", infos_client["NAME_FAMILY_STATUS"].values[0])
    st.write("** Nombre d'enfants : **{:.0f}".format(infos_client["CNT_CHILDREN"].values[0]))
    
    

    #Diagramme de distribution d'âge
    data_age = load_age_population(data)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(data_age, edgecolor = 'k', color="goldenrod", bins=20)
    ax.axvline(int(infos_client["DAYS_BIRTH"].values / 365), color="green", linestyle='--')
    ax.set(title='Age client', xlabel='Age(Year)', ylabel='')
    st.pyplot(fig)
 
    st.markdown("<a name='S2.2'>", unsafe_allow_html=True)  
    st.markdown("<font color=maroon size=+3><b>2.2 <u>Revenus client(USD)</u></b></font>", unsafe_allow_html=True) 

    st.write("*Revenu total : *{:.0f}".format(infos_client["AMT_INCOME_TOTAL"].values[0]))
    st.write("**Montant du crédit : **{:.0f}".format(infos_client["AMT_CREDIT"].values[0]))
    st.write("**Rentes de crédit : **{:.0f}".format(infos_client["AMT_ANNUITY"].values[0]))
    st.write("**Montant du bien à créditer : **{:.0f}".format(infos_client["AMT_GOODS_PRICE"].values[0]))
    
    #Diagramme de répartition des revenus
    st.markdown ("<b><u>Diagramme de répartition des revenus</u> :</b>", unsafe_allow_html=True)     
    data_income = load_income_population(data)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(data_income["AMT_INCOME_TOTAL"], edgecolor = 'k', color="goldenrod", bins=10)
    ax.axvline(int(infos_client["AMT_INCOME_TOTAL"].values[0]), color="green", linestyle='--')
    ax.set(title='Revenus Client', xlabel='Revenus (USD)', ylabel='')
    st.pyplot(fig)
    
    #Relation Âge / Revenu Graphique interactif total
    st.markdown ("<b><u>Relation Âge / Revenus total  </u> :</b>", unsafe_allow_html=True)    
    data_sk = data.reset_index(drop=False)
    data_sk.DAYS_BIRTH = (data_sk['DAYS_BIRTH']/365).round(1)
    fig, ax = plt.subplots(figsize=(10, 10))
    fig = px.scatter(data_sk, x='DAYS_BIRTH', y="AMT_INCOME_TOTAL", 
                     size="AMT_INCOME_TOTAL", color='CODE_GENDER',
                     hover_data=['NAME_FAMILY_STATUS', 'CNT_CHILDREN', 'NAME_CONTRACT_TYPE', 'SK_ID_CURR'])

    fig.update_layout({'plot_bgcolor':'#f0f0f0'}, 
                      title={'text':"Relation Âge / Revenus total ", 'x':0.5, 'xanchor': 'center'}, 
                      title_font=dict(size=20, family='Verdana'), legend=dict(y=1.1, orientation='h'))


    fig.update_traces(marker=dict(line=dict(width=0.5, color='#3a352a')), selector=dict(mode='markers'))
    fig.update_xaxes(showline=True, linewidth=2, linecolor='#f0f0f0', gridcolor='#cbcbcb',
                     title="Age", title_font=dict(size=18, family='Verdana'))
    fig.update_yaxes(showline=True, linewidth=2, linecolor='#f0f0f0', gridcolor='#cbcbcb',
                     title="Income Total", title_font=dict(size=18, family='Verdana'))

    st.plotly_chart(fig)
    
 



    #Compute decision according to the best threshold
    #if prediction <= xx :
    #    decision = "<font color='green'>**LOAN GRANTED**</font>" 
    #else:
    #    decision = "<font color='red'>**LOAN REJECTED**</font>"

    #st.write("**Decision** *(with threshold xx%)* **: **", decision, unsafe_allow_html=True)

    st.markdown("<a name='S2.3'>", unsafe_allow_html=True)  
    st.markdown("<font color=maroon size=+3><b>2.3 <u>Données du client</u></b></font>", unsafe_allow_html=True) 
    
    st.write(identite_client(data, chk_id))

    
    #Feature importance / description
    
    st.markdown("<a name='S3'>", unsafe_allow_html=True) 
    st.markdown("<font color=maroon size=+5><b>3 - les caractérisques importantes </b></font>", unsafe_allow_html=True) 


    shap.initjs()
    X = sample.iloc[:, :-1]
    X = X[X.index == chk_id]
    number = st.slider("Pick a number of features…", 0, 20, 5)
    
    fig, ax = plt.subplots(figsize=(10, 10))
    explainer = shap.TreeExplainer(load_model())
    shap_values = explainer.shap_values(X)
    shap.summary_plot(shap_values[0], X, plot_type ="bar", max_display=number, color_bar=False, plot_size=(5, 5))
    st.pyplot(fig)
    
    if st.checkbox("Need help about feature description ?") :
        list_features = description.index.to_list()
        feature = st.selectbox('Feature checklist…', list_features)
        st.table(description.loc[description.index == feature][:1])
        
    else:
        st.markdown("<i>…</i>", unsafe_allow_html=True)
            
    

    #Similar customer files display
    st.markdown("<a name='S4'>", unsafe_allow_html=True) 
    st.markdown("<font color=maroon size=+5><b>4 - Clients simailaires</b></font>", unsafe_allow_html=True) 

    knn = load_knn(sample)
    st.markdown("<u>Liste des 10 fichiers les plus proches de ce Client :</u>", unsafe_allow_html=True)
    st.dataframe(load_kmeans(sample, chk_id, knn))
    #st.markdown("<i>Target 1 = Customer with default</i>", unsafe_allow_html=True)
 
 
    st.markdown("<a name='S5'>", unsafe_allow_html=True) 
    st.markdown("<font color=maroon size=+5><b>5 -  Interprétabilité des résultats</b></font>", unsafe_allow_html=True) 
        
    from lime.lime_tabular import LimeTabularExplainer

    X1=sample.iloc[:, :-1]
    X1[X1.index == int(chk_id)] 

    #transformation échantillon
    from sklearn.preprocessing import StandardScaler
    stds = StandardScaler()
    Z = stds.fit_transform(sample.drop('TARGET', axis=1))
    ZC = stds.transform(X1[X1.index == int(chk_id)] )
    
    from lime.lime_tabular import LimeTabularExplainer

    lime1 = LimeTabularExplainer(Z,
                                 feature_names=sample.drop('TARGET', axis=1).columns,
                                 class_names=["Solvable", "Non Solvable"],
                                 discretize_continuous=False)
                                
    
  
   
    
    exp = lime1.explain_instance(ZC[0,:], clf.predict_proba,     num_samples=100) 
    
    st.write(exp.show_in_notebook())
    
    st.markdown("<b> Affichage sous forme graphique</b>", unsafe_allow_html=True) 
    

    plt.tight_layout()
    st.pyplot(exp.as_pyplot_figure())

if __name__ == '__main__':
    main()