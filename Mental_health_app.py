import numpy as np
import pandas as pd
import streamlit as st
from mlxtend.frequent_patterns import fpgrowth
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(page_title="Horizon Health",page_icon="data:image/png;base64,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", layout='centered', initial_sidebar_state='auto', menu_items=None)

header = st.container()
dataset = st.container()
modelTraining = st.container()
Doctor = st.container()

with header:

    st.image("https://en.horizonnb.ca/media/953975/horizon_logo_black.jpg" , use_column_width=True)
   


with dataset:

    df = pd.read_csv("Data/dataset3.csv")

    x=df.loc[:, 'feeling nervous':'blamming yourself'].replace("yes", pd.Series(True, df.columns))
    x=x.loc[:, 'feeling nervous':'blamming yourself'].replace("no", pd.Series(False, df.columns))

    doyou = ['feel hopeless' ,'feel angry' , 'over react' , 'feel negative' , 'have changes in eating' , 'have suicidal thought' , 'have a close friend',
         'have a social media addiction' ,'a popping up stressful memory','avoid people or activities', 'have material possessions', 'panic',
         'have trouble concentrating' ]
    
    areyou = ['sweating' , 'having trouble in concentration' , 'having trouble in sleeping' , 'feeling nervous' , 'having nightmares' , 'an intovert',
            'having trouble with work'  , 'feeling tired' , 'blamming yourself' , 'breathing rapidly']
    
    haveYou = ['gained weight']

    final = fpgrowth(x, min_support=0.6 , use_colnames=True )

    fin = final['itemsets']
    y = list()
    for i in fin:
            y.append(list(i))


    def getInList(word , test1): 

        t = [i for i in test1 if word in i]

        for d in t:
            d.remove(word)
        return t

    def getNotInList(word , temp):
        t = [i for i in temp if word not in i]

        return t

    def getMax(freqList):
        fin = final['itemsets']
        k = list()
        for i in fin:
            k.append(list(i))
        u = dict()
        for i in k[0:17]:
            count = 0
            for j in freqList:
                if i[0] in j:
                    count+=1
            u.update({i[0]:count})

        MaxDictVal = max(u, key=u.get)
            
        return MaxDictVal


with modelTraining:

    tel_col1 = st.table()
    tel5 = st.table()
    col1 , col2 = tel5.columns(2)


    col_1 , col_2 =tel_col1.columns(2)
    fName =  col_1.text_input("Enter Your Fist Name : ")
    lName =  col_2.text_input("Enter Your Last Name : ")
    Age =  col_1.text_input("Enter Your Age     : ")
    ticket =  col_2.text_input("Enter Your Booking Number     : ")

    symptoms = list()
    bel = st.header("Answer the following questions")
    
    rflag = 1
    tempwordlist = y
    maxword = getMax(tempwordlist)  
    sel = st.table()
    answers = dict()
    
    word = sel.selectbox("Question 1 : Do you "+maxword+" all the time" ,("-" , "yes" , "no") , 0, help = "test")

    prog=1  
    progressBar = st.progress(prog)
    
    for i in range(0,3):
        if word == 'yes':
            
            answers.update({maxword : word})
            prog+=25
            progressBar.progress(prog)

            symptoms.append(maxword)
            sel.empty()           
            tempwordlist = getInList(maxword ,tempwordlist)
            maxword = getMax(tempwordlist)
            if maxword in areyou:
                question = " Are you "
            elif maxword in doyou:
                question = " Do you "
            else:
                question = " Have you "
            word = sel.selectbox("Question "+str(i+2) + question +maxword+" all the time" ,("-" , "yes" , "no") , 0 , help = "test")
            
            rflag+=1

        elif word == 'no':
            answers.update({maxword : word})
            prog+=25
            progressBar.progress(prog)
            tempwordlist = getNotInList(maxword ,tempwordlist)
            maxword = getMax(tempwordlist)
            if maxword in areyou:
                question = " Are you "
            elif maxword in doyou:
                question = " Do you "
            else:
                question = " Have you "
            word = sel.selectbox("Question "+str(i+2) +question +maxword+" all the time" ,("-" , "yes" , "no") , 0 , help = "test")
            rflag+=1
    if word == 'yes':
        answers.update({maxword : word})

        prog+=24
        progressBar.progress(prog)
        symptoms.append(maxword)
        tempwordlist = getInList(maxword ,tempwordlist)
    elif word == 'no':
        answers.update({maxword : word})
        prog+=24
        progressBar.progress(prog)
        tempwordlist = getNotInList(maxword ,tempwordlist)


    if rflag == 4 and word == 'yes' or word == "no":            
        progressBar.empty()
        bel.empty()

        for i in answers.items():
                output = i[0] +" "+ i[1]
                col1.text(".   ⚙️"+ output)
        
        temp = sel.button("Submit")
        if temp:
            sel.empty()
            tel_col1.empty()
            tel5.empty()
            st.success("Your answers has been saved successfully! Thank You")
            st.balloons()




    frequentSymptoms = list()
    for i in tempwordlist:
        for j in i:
            if j not in frequentSymptoms:
                frequentSymptoms.append(j)
    
    Demo = st.button("Doctor Demo")
    if Demo:
        sel.empty()
        progressBar.empty()
        tel_col1.empty()
        tel5.empty()

        with Doctor:


            st.subheader("Patient Name: " + fName + " "+ lName)
            st.subheader("Patient Age: " + Age)
            st.subheader("Patient Booking number: " + ticket)

            c1 , c2 = st.columns(2)


            c1.subheader("Patients answers")
            if len(symptoms) != 0:
                for x in symptoms:
                    c1.text(".  "+"⚙️" + x)
            else:
                c1.error("No symptoms to be shown : The patient has selected 0 symptoms")
            
            c2.subheader("Other possible Symptoms")

            for i in frequentSymptoms:
                c2.text(".   ⚙️" + i)
            
            le=LabelEncoder()
            for i in df[:]:
                df[i]=le.fit(df[i]).transform(df[i])

            col = list()
            for i in df:
                col.append(i)
            
            X,Y=df[col[:24]], df['Disorder']

            clf_gini = DecisionTreeClassifier(criterion='gini', max_depth=3 ,  random_state=1)
            clf_gini.fit(X, Y)

            final_symptoms = frequentSymptoms + symptoms
            rampage = []
            for i in col[0:24]:
                if i in final_symptoms:
                    rampage.append(1)
                else:
                    rampage.append(0)
            rampage=[rampage]
            ramp = pd.DataFrame(rampage)

            y_pred_gini = clf_gini.predict(ramp)

            if len(symptoms) != 0:
                if y_pred_gini[0] == 0:
                    disorder = "Anxiety"
                elif y_pred_gini[0] == 1:
                    disorder = "Depression"
                elif y_pred_gini[0] == 2:
                    disorder = "Loneliness"
                elif y_pred_gini[0] == 3:
                    disorder = "None"
                elif y_pred_gini[0] == 4:
                    disorder = "Stress"
            else:
                disorder = "Disorder cannot be predicted since no symptoms shown"

            st.error("Predicted disorder : " + disorder )
    

    

