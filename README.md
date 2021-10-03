# FDM_Mental_Health_app

This repository contains a Data Mining mini project on Mental health disorder prediction using Association rule mining and decision tree classifier as an assignment for a data science undergraduate module at SLIIT

## Problem background:

This project investigates a real-world problem and proposes a solution for it using data mining and machine learning techniques. Therefore, the selected problem is associated with a psychiatric clinical center.   

‘Horizon’ is a psychiatric clinical center which provides facilities to the people who are struggling with mental disorders to visit, get checkups and to get treated under the trained, licensed psychiatrists. So, the patients can reserve a time from the counter to meet a psychiatrist and on that reserved time, they will be given the chances to explain their illnesses and to get treated from the psychiatrist.

Usually, a psychiatrist takes more than 30 minutes to examine a patient. Due to the pandemic situation and the resulted environment of the country, the number of the people who are suffering from the mental disorders have got rapidly increased and therefore, the number of bookings per a psychiatrist has become increased. So, most of the time, there is a queue in front of every room which has reserved for a psychiatrist. Also, considering the time which a psychiatrist spends for a patient, the time which the other patients must wait in the queue also could be more than 30 minutes.

With this number increment, the clinical center has identified that every psychiatrist in the center is having a tight schedule because of the long queue of patients. Also, recently the clinical center had to go through some problems related to some critical patients’ behaviors who got angry because of the hours of waiting time in the queue. So, from clinical center’s point of view, they felt the need of something which could help the psychiatrists to make this diagnosing people process easier so that they can save time for both doctor and patient parties. Also, they felt the need of identifying the critical level patients so that those patients do not have to wait for a long time in the queue and as a staff they could help those patients by giving them a priority and providing the relevant facilities.

So as a solution for the above problems, we are suggesting a small questionnaire which contains 4 questions and has to be answered by the patients in the queue according to the first come first served basis. This will not be compulsory to every patient, but it will be recommended for all. Questions will not be just questions, but which have a connection between each other considering the frequent symptoms patterns identified from the past data of the patients who got diagnosed with those symptoms. The answer summary of each patient will be provided to the psychiatrist with the patient’s entrance and at the end of the summary, the predictive symptoms that this patient could have also will be suggested to the psychiatrist for his easiness. 

We hope our solution will help the clinical center to solve above mentioned problems. The summary being an extra help to the psychiatrists to save time, therefore the decrement of the time reserved for each patient leads to decrease the waiting time in the queue. Also, can be identified critical level patients from the questionnaire considering their answers and the summary.

So, they can take necessary precautions regarding that problem also before facing many more problems which could affect other patients also. 

APP link :- https://mental-health-arm-app.herokuapp.com/

## Files in this repository:
  1) FP_Growth.ipynb :- The built model for data training  using FP growth algorithm
  2) apriori.ipynb :- The built model for data training  using Apriori algorithm
  3) prediction.ipynb :- This file contains the model training used for the prediction of the disorder
  4) Mental_health_app.py :- The user web app build using streamlit library 
  5) Health_app_demo.mp4 :- Demonstration of the app and project background
