
# coding: utf-8

# In[1]:


# # Importing necessary Libraries
import pandas as pd
import numpy as np
from sklearn import preprocessing
import re
import scipy
from scipy import spatial
from scipy.stats import ttest_ind
import json
import time, os, fnmatch, shutil
import os


# In[2]:


def UI(Path):
    raw_data=pd.read_csv(Path)
    #Lower case
    raw_data.columns=[x.lower() for x in raw_data.columns]
    # Conversion of TC_FLAG into int variables
    #dtype_list
    dtype_list=[]
    for i in range(0,len(raw_data.columns)):
        dtype_list.append(str(raw_data[list(raw_data.columns)[i]].dtype))


    #Defining weights
    Semi_weights= list(np.ones(len(raw_data.columns)))
    weights=[]
    for i in range(0,len(Semi_weights)):
        weights.append(int(Semi_weights[i]))
        
        #Defining Standardisation_flag
        Standardisation_Flag=1
        #Selected Values
        Semi_Selected= list(np.ones(len(raw_data.columns)))
        Selected=[]
    for i in range(0,len(Semi_Selected)):
        Selected.append(int(Semi_Selected[i]))
    ##Java
    import json
    data = {}
    data['Algorithm'] = {"Default_Value": "greedy","Selected_Value": "greedy","Values": [{"Display_Name": "GREEDY ALGORITHM","Value": "greedy"}                                                                                         
       ,{"Display_Name": "GROUPING ALGORITHM","Value": "grouping"}]
    }
    data['Column_List']=[]
    for i in range(0,len(raw_data.columns)):
        data2={'Value':raw_data.columns[i],'Display_Name':raw_data.columns[i].upper()}
        data['Column_List'].append(data2)
        
    data['Column_Details']=[]
    for i in range (0,len(raw_data.columns)):
        data3={"Column_Name":raw_data.columns[i],'Display_Name':raw_data.columns[i].upper(),"Column_Dtype":dtype_list[i],"Weights":weights[i],"Selected":Selected[i]}
        data['Column_Details'].append(data3)

    data['Standardisation_Flag']={"Default_Value": 0,"Selected_Value": 1,"Values": [0,1]}
    data['Primary_Column']={"Default_Value":"","Selected_Value":"id"}
    data['Secondary_Column']={"Default_Value":"","Selected_Value":"testcont_flag"}
    import os
    wd = os.getcwd()
    wd=wd.replace('\\','/')
    data['Input_File_Path']=Path
    json_data = json.dumps(data)
    #Returning Json
    import json
    json_data_final=json.loads(json_data)
    return json_data_final


# In[3]:


json_data_final=UI("C:/Users/rishabh.malhotra/Desktop/Conversion/Categorical Final Data.csv")


# In[6]:


def runalgorithm_variables(json_data_final):    

    #Importing File
    Path=json_data_final['Input_File_Path']
    raw_data=pd.read_csv(Path)
    #Categorical Columns in lower case
    raw_data.columns=[x.lower() for x in raw_data.columns]
    #Reading rawdata from keycol and tc_flag
    keycol=json_data_final['Primary_Column'].get('Selected_Value')
    tc_flag=json_data_final['Secondary_Column'].get('Selected_Value')
    #All_Integer
    ### Treating null values
    for x in raw_data.isnull().sum():
        if x==1:
            print('There are missing values')
    pd.isnull(raw_data).any(1).nonzero()
    raw_data.dropna(inplace=True)
    #Defining dtype_list
    raw_data.set_index(keycol,inplace=True)
    dtype_list=[]
    for i in range(0,len(raw_data.columns)):
        dtype_list.append(str(raw_data[list(raw_data.columns)[i]].dtype))
    #Defining All_Integer
    if dtype_list.count('object')>0:
        All_Integer='No'
    else:
        All_Integer='Yes'
    
    #
    #Defining char_col
    if dtype_list.count('object')>0:
        char_col=[keycol,tc_flag,'categorical']
    else:
        char_col=[keycol,tc_flag] 
    #Defining categorical_column
    categorical_column=[]
    for i in range(0,len(dtype_list)):
        if dtype_list[i]=='object':
            categorical_column.append(raw_data.columns[i])
        else:
            pass
    raw_data.reset_index(inplace=True)
    #drop_var
    drop_var=[]
    
    for i in range(0,len(raw_data.columns)):
        x=json_data_final['Column_Details'][i].get('Selected')
        if x==0.0:
            drop_var.append(raw_data.columns[i])
        else:
            pass
    #Standardisation_Flag
    Standardisation_Flag=json_data_final['Standardisation_Flag'].get('Selected_Value')
    #Final_Weights
    Final_Weights=[]
    for i in range(0,len(raw_data.columns)):
        X=json_data_final['Column_Details'][i].get('Weights')
        Final_Weights.append(X)
    json_input={'raw_data':raw_data,'Standardisation_Flag':Standardisation_Flag,'drop_var':drop_var,'Final_Weights':Final_Weights,'categorical_column':categorical_column,'char_col':char_col,'All_Integer':All_Integer,'dtype_list':dtype_list}
    return json_input


# In[7]:


def Creating_raw_data(json_input,json_data_final):
    raw_data=json_input['raw_data']
    keycol=json_data_final['Primary_Column'].get('Selected_Value')
    tc_flag=json_data_final['Secondary_Column'].get('Selected_Value')
    
    #Defining variables
    categorical_column=json_input.get('categorical_column')
    char_col=json_input.get('char_col')
    All_Integer=json_input.get('All_Integer')
    drop_var=json_input.get('drop_var')
                                                
    raw_data.drop(drop_var, axis=1,inplace=True)
    raw_data.shape

    # ### Concatnating User on Keycol column and converting it into string so that it can be called later on while bucketing and binning
    raw_data[keycol]='User' + raw_data[keycol].astype(str)
    raw_data[keycol]=raw_data[keycol].astype(str)
    #Defining All Integer
    

    if All_Integer=='No':
        raw_data["categorical"] = raw_data[categorical_column].apply(''.join, axis=1)

        for i in range(0,len(categorical_column)):
            raw_data=raw_data.drop(categorical_column[i],axis=1)
    else:
        pass


    # # Removing row level and userid duplicates

    # In[41]:


    raw_data.drop_duplicates(subset=[keycol],keep='first', inplace= True)
    raw_data.drop_duplicates(subset=keycol,keep=False,inplace=True)
    raw_data.reset_index(drop=True,inplace=True)



    # # Sorting values based on TC_Flag

    # In[43]:


    raw_data= raw_data.sort_values(tc_flag,ascending=True)
    raw_data=raw_data.reset_index(drop=True)


    # ### Defining key columns

    # Converting keycol into keycol_df

    # In[44]:


    ##### define key column which identifies unique rows, in this case "userid"
    key_column= raw_data[keycol]
    keycol_df= pd.DataFrame(key_column)
    keycol_df


    # ### Flag for test and control

    # Converting tc_flag_column into tc_flag_df

    # In[45]:


    tc_flag_column= raw_data[tc_flag]
    tc_flag_df= pd.DataFrame(tc_flag_column)


    # ## TO BE DONE IF TC_FLAG Categorical

    # ## Creating a categorical list of categorical columns unique values forsorting and filtering of data

    # In[46]:


    if All_Integer=='No':
        Categorical_col=raw_data['categorical']
        Categorical_list=raw_data['categorical'].unique()
    else:
        Categorical_list=['No_categorical_var']


    # ## Final Raw Data

    # ### Creating raw_data_categorical which is used in binning

    # In[47]:


    if All_Integer=='No':
        raw_data_categorical=raw_data['categorical']
    else:
        raw_data_categorical=pd.DataFrame()

    raw_data_binning=raw_data
    Variable_Dict={'keycol':keycol,'tc_flag':tc_flag,'raw_data':raw_data,'raw_data_binning':raw_data_binning,'raw_data_categorical':raw_data_categorical,'Categorical_list':Categorical_list,'tc_flag_df':tc_flag_df,'keycol_df':keycol_df}
    return Variable_Dict


# # Our before_algo contains
# 1. raw_data
# 2. raw_data_binning
# 3. raw_data_categorical
# 4. Categorical_list
# 5. tc_flag_df
# 6. keycol_df
# 

# ## Mapped Cell

# In[8]:


def Goku(Cat_list,Variable_Dict,json_data_final):
    tc_flag_df=Variable_Dict['tc_flag_df']
    keycol_df=Variable_Dict['keycol_df']
    raw_data=Variable_Dict['raw_data']
    raw_data_binning=Variable_Dict['raw_data_binning']
    Categorical_list=Variable_Dict['Categorical_list']
    raw_data_categorical=Variable_Dict['raw_data_categorical']
    keycol=json_data_final['Primary_Column'].get('Selected_Value')
    tc_flag=json_data_final['Secondary_Column'].get('Selected_Value')
    
    Categorical_list=raw_data['categorical'].unique()
    raw_data_1=raw_data[raw_data['categorical'] == Categorical_list[Cat_list]] 
    a=len(raw_data_1)
    tc_flag_df_1=raw_data_1[tc_flag][0:a]
    keycol_df_1=raw_data_1[keycol][0:a]
    tc_flag_df_1=tc_flag_df_1.reset_index(drop=True)
    keycol_df_1=keycol_df_1.reset_index(drop=True)
    raw_data_1=raw_data_1.reset_index(drop=True)
    print(keycol_df_1.shape)
    print(tc_flag_df_1.shape)
    print(raw_data_1.shape)
    Goku1={'raw_data_1':raw_data_1,'keycol_df_1':keycol_df_1,'tc_flag_df_1':tc_flag_df_1}
    return Goku1


# In[9]:


def Goku_Int(Variable_Dict,json_data_final):
    tc_flag_df=Variable_Dict['tc_flag_df']
    keycol_df=Variable_Dict['keycol_df']
    raw_data=Variable_Dict['raw_data']
    raw_data_binning=Variable_Dict['raw_data_binning']
    Categorical_list=Variable_Dict['Categorical_list']
    raw_data_categorical=Variable_Dict['raw_data_categorical']
    keycol=json_data_final['Primary_Column'].get('Selected_Value')
    tc_flag=json_data_final['Secondary_Column'].get('Selected_Value')
    raw_data_1=raw_data
    a=len(raw_data_1)
    tc_flag_df_1=raw_data_1[tc_flag][0:a]
    keycol_df_1=raw_data_1[keycol][0:a]
    tc_flag_df_1=tc_flag_df_1.reset_index(drop=True)
    keycol_df_1=keycol_df_1.reset_index(drop=True)
    raw_data_1=raw_data_1.reset_index(drop=True)
    print(keycol_df_1.shape)
    print(tc_flag_df_1.shape)
    print(raw_data_1.shape)
    Goku_Int1={'raw_data_1':raw_data_1,'keycol_df_1':keycol_df_1,'tc_flag_df_1':tc_flag_df_1}
    return Goku_Int1


# In[10]:


def Standardisation(Goku1,json_data_final,json_input):
    char_col=json_input['char_col']
    raw_data_1=Goku1['raw_data_1']
    Standardisation_Flag=json_data_final['Standardisation_Flag'].get('Selected_Value')
    if Standardisation_Flag==1:
        
        from sklearn.preprocessing import StandardScaler
        raw_data_1=raw_data_1.drop(char_col,axis=1)
        Scaler = StandardScaler()  
        Scaler.fit(raw_data_1)
        scaled_features= Scaler.transform(raw_data_1)
        Standardised_df= pd.DataFrame(scaled_features,columns=raw_data_1.columns)
        Standardised_Data= Standardised_df
        return Standardised_Data
    else:
        
        raw_data_1=raw_data_1.drop(char_col,axis=1)
        Standardised_df=raw_data_1
        Standardised_Data= Standardised_df
        return Standardised_Data


# In[11]:


def Standardisation_Int(Goku_Int1,json_data_final,json_input):
    char_col=json_input['char_col']
    raw_data_1=Goku_Int1['raw_data_1']
    Standardisation_Flag=json_data_final['Standardisation_Flag'].get('Selected_Value')
    if Standardisation_Flag==1:
        
        from sklearn.preprocessing import StandardScaler
        raw_data_1=raw_data_1.drop(char_col,axis=1)
        Scaler = StandardScaler()  
        Scaler.fit(raw_data_1)
        scaled_features= Scaler.transform(raw_data_1)
        Standardised_df= pd.DataFrame(scaled_features,columns=raw_data_1.columns)
        Standardised_Data= Standardised_df
        return Standardised_Data
    else:
        
        raw_data_1=raw_data_1.drop(char_col,axis=1)
        Standardised_df=raw_data_1
        Standardised_Data= Standardised_df
        return Standardised_Data


# In[35]:


def Weight_Multiplication(Goku1,Standardised_Data,json_input,json_data_final):
    raw_data_1=Goku1['raw_data_1']
    keycol_df_1=Goku1['keycol_df_1']
    tc_flag_df_1=Goku1['tc_flag_df_1']
    Final_Weights=json_input['Final_Weights']
    Final_Standardised_Data1=pd.DataFrame()
    Standardised_columns=list(Standardised_Data.columns)
    Weight_Dict={json_data_final['Column_Details'][0].get('Column_Name'):json_data_final['Column_Details'][0].get('Weights')}
    for i in range(1,len(raw_data_1.columns)):
        Weight_Dict.update({json_data_final['Column_Details'][i].get('Column_Name'):json_data_final['Column_Details'][i].get('Weights')})

    Standardised_columns=list(Standardised_Data.columns)
    for i in Standardised_columns:
        Final_Std_Data= pd.DataFrame(Standardised_Data.loc[:,i]*Weight_Dict[i])
        Final_Standardised_Data1= pd.concat([Final_Standardised_Data1,Final_Std_Data],axis=1)
        print(i)
    Final_Standardised_Data=pd.concat([keycol_df_1,tc_flag_df_1,Final_Standardised_Data1],axis=1)
    return Final_Standardised_Data   

def Weight_Multiplication_Int(Goku_Int1,Standardised_Data,json_input,json_data_final):
    raw_data_1=Goku_Int1['raw_data_1']
    keycol_df_1=Goku_Int1['keycol_df_1']
    tc_flag_df_1=Goku_Int1['tc_flag_df_1']
    Final_Standardised_Data1=pd.DataFrame()
    Standardised_columns=list(Standardised_Data.columns)
    Weight_Dict={json_data_final['Column_Details'][0].get('Column_Name'):json_data_final['Column_Details'][0].get('Weights')}
    for i in range(1,len(raw_data_1.columns)):
        Weight_Dict.update({json_data_final['Column_Details'][i].get('Column_Name'):json_data_final['Column_Details'][i].get('Weights')})

    Standardised_columns=list(Standardised_Data.columns)
    for i in Standardised_columns:
        Final_Std_Data= pd.DataFrame(Standardised_Data.loc[:,i]*Weight_Dict[i])
        Final_Standardised_Data1= pd.concat([Final_Standardised_Data1,Final_Std_Data],axis=1)
        print(i)
    Final_Standardised_Data=pd.concat([keycol_df_1,tc_flag_df_1,Final_Standardised_Data1],axis=1)
    return Final_Standardised_Data 


# In[13]:


# GREEDY ALGORITHM
def Array_Making(Final_Standardised_Data,json_data_final):
    tc_flag=json_data_final['Secondary_Column'].get('Selected_Value')
    control=Final_Standardised_Data[Final_Standardised_Data[tc_flag]==0]
    test=Final_Standardised_Data[Final_Standardised_Data[tc_flag]==1]

    Final_Standardised_Data=Final_Standardised_Data.reset_index(drop=True)
    Array_Make={'test':test,'control':control}
    return Array_Make



# In[14]:


def Euc_df(x,y,Array_Make):
    test=Array_Make['test']
    control=Array_Make['control']
    ARRAY= scipy.spatial.distance.cdist(test.iloc[:,2:],control.iloc[x:y,2:], metric='euclidean')
    return ARRAY


# In[15]:


### Applying Greedy Algorithm and finding Greedy matches by finding minimum euclidean distances
def Greedy(ARRAY,Array_Make,json_data_final):
    keycol=json_data_final['Primary_Column'].get('Selected_Value')
    control=Array_Make['control']
    test=Array_Make['test']
    Test=[]
    Control=[]
    x=0
    while x<len(test):
        Pairs=pd.DataFrame(columns=['Test','Control'])
        a=(control.iloc[np.argmin(ARRAY[x])][keycol])

        Control.append(a)
        b=(test.iloc[x][keycol]) 

        Test.append(b)  
        ARRAY[:, np.argmin(ARRAY[x]) ] = 666
        print(x)
        x=x+1
    Greedy={'Test':Test,'Control':Control}
    return Greedy


# In[16]:


def Create_Pairs_Final(Greedy,Final_Standardised_Data,Goku1,json_data_final):
    global raw_data_1
    raw_data_1=Goku1['raw_data_1']
    keycol_df_1=Goku1['keycol_df_1']
    tc_flag_df_1=Goku1['tc_flag_df_1']
    keycol=json_data_final['Primary_Column'].get('Selected_Value')
    tc_flag=json_data_final['Secondary_Column'].get('Selected_Value')
    Test=Greedy['Test']
    Control=Greedy['Control']
    Test_PRED_Final_1=pd.DataFrame(Test)
    Control_PRED_Final_1=pd.DataFrame(Control)
    test_control_pair_rd=pd.concat([Test_PRED_Final_1,Control_PRED_Final_1],axis=1)
    test_control_pair_rd.columns=['Test','Control']
    tc_flag_df_1=Final_Standardised_Data[[tc_flag]]
    raw_data_1=raw_data_1.set_index([keycol],drop=True)
    raw_test_1=raw_data_1[raw_data_1[tc_flag]==1]
    raw_control_1=raw_data_1[raw_data_1[tc_flag]==0]
    Control_list=list(test_control_pair_rd['Control'])
    test_pair_final=raw_test_1
    control_pair_final=raw_control_1.loc[Control_list]
    test_control_pair={'test_pair_final':test_pair_final,'control_pair_final':control_pair_final}
    test_pair_final=test_pair_final.drop('categorical',axis=1)
    control_pair_final=control_pair_final.drop('categorical',axis=1)
    Pairs_Final={'test_pair_final':test_pair_final,'control_pair_final':control_pair_final,'test_control_pair_rd':test_control_pair_rd}
    return Pairs_Final

def Create_Pairs_Final_Int(Greedy,Final_Standardised_Data,Goku_Int1,json_data_final):
    raw_data_1=Goku_Int1['raw_data_1']
    keycol_df_1=Goku_Int1['keycol_df_1']
    tc_flag_df_1=Goku_Int1['tc_flag_df_1']
    keycol=json_data_final['Primary_Column'].get('Selected_Value')
    tc_flag=json_data_final['Secondary_Column'].get('Selected_Value')
    Test=Greedy['Test']
    Control=Greedy['Control']
    Test_PRED_Final_1=pd.DataFrame(Test)
    Control_PRED_Final_1=pd.DataFrame(Control)
    test_control_pair_rd=pd.concat([Test_PRED_Final_1,Control_PRED_Final_1],axis=1)
    test_control_pair_rd.columns=['Test','Control']
    tc_flag_df_1=Final_Standardised_Data[[tc_flag]]
    raw_data_1=raw_data_1.set_index([keycol],drop=True)
    raw_test_1=raw_data_1[raw_data_1[tc_flag]==1]
    raw_control_1=raw_data_1[raw_data_1[tc_flag]==0]
    Control_list=list(test_control_pair_rd['Control'])
    test_pair_final=raw_test_1
    control_pair_final=raw_control_1.loc[Control_list]
    test_control_pair={'test_pair_final':test_pair_final,'control_pair_final':control_pair_final}
    Pairs_Final_Int={'test_pair_final':test_pair_final,'control_pair_final':control_pair_final,'test_control_pair_rd':test_control_pair_rd}
    return Pairs_Final_Int


# In[17]:


## T_test function with 2 inputs of arrays whose t-test is to be done
def T_test_Func(Pairs_Final,json_data_final):
    test_pair_final=Pairs_Final['test_pair_final']
    control_pair_final=Pairs_Final['control_pair_final']
    ttest=[]
    ttest_array=ttest_ind(test_pair_final,control_pair_final,axis=0)
    x=0
    while x<len(ttest_array.pvalue):

        if ttest_array.pvalue[x]>0.05:
            ttest.append('pass')
            print('pass')   
        else:
            print('fail') 
            ttest.append('fail')
        x=x+1
    print(ttest_array.pvalue)
    Test_mean=[]
    for i in range(0,len(test_pair_final.columns)):
        Test_mean.append(test_pair_final[test_pair_final.columns[i]].mean())

    Control_mean=[]
    for i in range(0,len(control_pair_final.columns)):
        Control_mean.append(control_pair_final[control_pair_final.columns[i]].mean())

    Test_Total_rows=[]
    for i in range(0,len(test_pair_final.columns)):
        Test_Total_rows.append(len(test_pair_final))

    Control_Total_rows=[]
    for i in range(0,len(control_pair_final.columns)):
        Control_Total_rows.append(len(control_pair_final))
        
    Greedy_T_Test={'ttest_array':ttest_array,'ttest':ttest,'Test_mean':Test_mean,'Control_mean':Control_mean,'Test_Total_Rows':Test_Total_rows,'Control_Total_Rows':Control_Total_rows}

    return Greedy_T_Test

## T_test function with 2 inputs of arrays whose t-test is to be done
def T_test_Func_Int(Pairs_Final_Int,json_data_final):
    test_pair_final=Pairs_Final_Int['test_pair_final']
    control_pair_final=Pairs_Final_Int['control_pair_final']
    ttest=[]
    ttest_array=ttest_ind(test_pair_final,control_pair_final,axis=0)
    x=0
    while x<len(ttest_array.pvalue):

        if ttest_array.pvalue[x]>0.05:
            ttest.append('pass')
            print('pass')   
        else:
            print('fail') 
            ttest.append('fail')
        x=x+1
    print(ttest_array.pvalue)
    Test_mean=[]
    for i in range(0,len(test_pair_final.columns)):
        Test_mean.append(test_pair_final[test_pair_final.columns[i]].mean())

    Control_mean=[]
    for i in range(0,len(control_pair_final.columns)):
        Control_mean.append(control_pair_final[control_pair_final.columns[i]].mean())

    Test_Total_rows=[]
    for i in range(0,len(test_pair_final.columns)):
        Test_Total_rows.append(len(test_pair_final))

    Control_Total_rows=[]
    for i in range(0,len(control_pair_final.columns)):
        Control_Total_rows.append(len(control_pair_final))
        
    Greedy_T_Test_Int={'ttest_array':ttest_array,'ttest':ttest,'Test_mean':Test_mean,'Control_mean':Control_mean,'Test_Total_Rows':Test_Total_rows,'Control_Total_Rows':Control_Total_rows}

    return Greedy_T_Test_Int


# In[18]:


def Greedy_Calling(json_data_final):
    json_input=runalgorithm_variables(json_data_final)
    Variable_Dict=Creating_raw_data(json_input,json_data_final)
    All_Integer=json_input['All_Integer']
    Pairs_Final=pd.DataFrame()
    T_test_Summary=pd.DataFrame()
    Categorical_list=list(Variable_Dict['Categorical_list'])
    if All_Integer=='No':
        for i in range(0,len(Categorical_list)):
            Goku1=Goku(i,Variable_Dict,json_data_final)
            Standardised_Data=Standardisation(Goku1,json_data_final,json_input)
            Final_Standardised_Data=Weight_Multiplication(Goku1,Standardised_Data,json_input,json_data_final)
            Array_Make=Array_Making(Final_Standardised_Data,json_data_final)
            ARRAY=Euc_df(0,len(Goku1['tc_flag_df_1']==0),Array_Make)
            Output_Greedy=Greedy(ARRAY,Array_Make,json_data_final)
            Create_Output_Pairs_Final=Create_Pairs_Final(Output_Greedy,Final_Standardised_Data,Goku1,json_data_final)
            Greedy_T_Test=T_test_Func(Create_Output_Pairs_Final,json_data_final)
            ttest_array=Greedy_T_Test['ttest_array']
            ttest=Greedy_T_Test['ttest']
            Test_mean=Greedy_T_Test['Test_mean']
            Control_mean=Greedy_T_Test['Control_mean']
            Test_Total_Rows=Greedy_T_Test['Test_Total_Rows']
            Control_Total_Rows=Greedy_T_Test['Control_Total_Rows']
            test_control_pair_rd=Create_Output_Pairs_Final['test_control_pair_rd']
            control_pair_final=Create_Output_Pairs_Final['control_pair_final']
            test_pair_final=Create_Output_Pairs_Final['test_pair_final']
            Categorical_list=Variable_Dict['Categorical_list']
            TC_pair_1=pd.DataFrame(test_control_pair_rd)
            Pairs_Final=pd.concat([Pairs_Final,TC_pair_1],axis=1)
            Final_categorical_list=[]
            for j in range(0,len(control_pair_final.columns)):
                Final_categorical_list.append(list(Categorical_list)[i])
                                                  
            T_test_Final=pd.DataFrame(np.column_stack([list(Final_categorical_list),list(control_pair_final.columns),Test_mean,Control_mean,Test_Total_Rows,Control_Total_Rows,list(ttest_array.pvalue),ttest]),columns=['Categorical Column','Columns','Test Mean','Control Mean','Test Count','Control Count','P Value','Pass/Fail'])
            T_test_Summary=pd.concat([T_test_Summary,T_test_Final])
            t = time.localtime()
            timestamp = time.strftime('%b-%d-%Y_%H%M', t)
            wd = os.getcwd()
            wd=wd.replace('\\','/')
            Pairs_Final.to_csv('Pairs_Final'+timestamp+'.csv')
            T_test_Summary.to_csv('T_test_Summary'+timestamp+".csv")

            Output_Module_1={}
            Output_Module_1['T-test'] = {"Display_Name":"T_test_Summary.csv",
                   "url": wd+"/T_test_Summary"+timestamp+".csv"},
            Output_Module_1["Pairs_Data"]={"Display_Name": "Pairs_Final.csv",
                    "url": wd + "/Pairs_Final"+timestamp+".csv"}
            Output_Module_1['T_Test_Table']=[]
            for i in range(0,len(T_test_Summary)):
                data4={"Categorical_Column":T_test_Summary.iloc[i][0],"Columns":T_test_Summary.iloc[i][1],"Test_Mean":T_test_Summary.iloc[i][2],"Control_Mean":T_test_Summary.iloc[i][3],"Test_Count":T_test_Summary.iloc[i][4],"Control_Count":T_test_Summary.iloc[i][5],"P_Value":T_test_Summary.iloc[i][6],"Pass/Fail":T_test_Summary.iloc[i][7]}
                Output_Module_1['T_Test_Table'].append(data4)
    else:
        Goku_Int1=Goku_Int(Variable_Dict,json_data_final)
        Standardised_Data=Standardisation_Int(Goku_Int1,json_data_final,json_input)
        Final_Standardised_Data=Weight_Multiplication_Int(Goku_Int1,Standardised_Data,json_input,json_data_final)
        Array_Make=Array_Making(Final_Standardised_Data,json_data_final)
        ARRAY=Euc_df(0,len(Goku_Int1['tc_flag_df_1']==0),Array_Make)
        Output_Greedy=Greedy(ARRAY,Array_Make,json_data_final)
        Create_Output_Pairs_Final_Int=Create_Pairs_Final_Int(Output_Greedy,Final_Standardised_Data,Goku_Int1,json_data_final)
        Greedy_T_Test_Int=T_test_Func_Int(Create_Output_Pairs_Final_Int,json_data_final)
        ttest_array=Greedy_T_Test_Int['ttest_array']
        ttest=Greedy_T_Test_Int['ttest']
        Test_mean=Greedy_T_Test_Int['Test_mean']
        Control_mean=Greedy_T_Test_Int['Control_mean']
        Test_Total_Rows=Greedy_T_Test_Int['Test_Total_Rows']
        Control_Total_Rows=Greedy_T_Test_Int['Control_Total_Rows']
        test_control_pair_rd=Create_Output_Pairs_Final_Int['test_control_pair_rd']
        control_pair_final=Create_Output_Pairs_Final_Int['control_pair_final']
        test_pair_final=Create_Output_Pairs_Final_Int['test_pair_final']

        TC_pair_1=pd.DataFrame(test_control_pair_rd)
        Pairs_Final=pd.concat([Pairs_Final,TC_pair_1],axis=1)
        T_test_Final=pd.DataFrame(np.column_stack([list(control_pair_final.columns),Test_mean,Control_mean,Test_Total_Rows,Control_Total_Rows,list(ttest_array.pvalue),ttest]),columns=['Columns','Test_Mean','Control_Mean','Test_Count','Control_Count','P_Value','Pass/Fail'])
        T_test_Summary=pd.concat([T_test_Summary,T_test_Final])
        t = time.localtime()
        timestamp = time.strftime('%b-%d-%Y_%H%M', t)
        wd = os.getcwd()
        wd=wd.replace('\\','/')
        Pairs_Final.to_csv('Pairs_Final'+timestamp+'.csv')
        T_test_Summary.to_csv('T_test_Summary'+timestamp+".csv")

        Output_Module_1={}
        Output_Module_1['T-test'] = {"Display_Name":"T_test_Summary.csv",
               "url": wd+"/T_test_Summary"+timestamp+".csv"},
        Output_Module_1["Pairs_Data"]={"Display_Name": "Pairs_Final.csv",
                "url": wd + "/Pairs_Final"+timestamp+".csv"}
        Output_Module_1['T_Test_Table']=[]
        for i in range(0,len(T_test_Summary)):
            data4={"Columns":T_test_Summary.iloc[i][0],"Test_Mean":T_test_Summary.iloc[i][1],"Control_Mean":T_test_Summary.iloc[i][2],"Test_Count":T_test_Summary.iloc[i][3],"Control_Count":T_test_Summary.iloc[i][4],"P_Value":T_test_Summary.iloc[i][5],"Pass/Fail":T_test_Summary.iloc[i][6]}
            Output_Module_1['T_Test_Table'].append(data4)            

    return Output_Module_1


# # GROUPING ALGORITHM

# In[19]:


def Bucketing(Variable_Dict,json_input,json_data_final):
    keycol=json_data_final['Primary_Column'].get('Selected_Value')
    tc_flag=json_data_final['Secondary_Column'].get('Selected_Value')
    All_Integer=json_input['All_Integer']
    raw_data=Variable_Dict['raw_data']
    raw_data=raw_data.set_index([keycol])
    raw_data=raw_data.drop([tc_flag],axis=1)
    if All_Integer == 'No':
        raw_data=raw_data.drop('categorical',axis=1)
    else:
        pass
    raw1_data=pd.DataFrame(index=raw_data.index)
    for i in range(0,len(raw_data.columns)):
        col=raw_data.columns[i]
        raw_buckets=pd.cut(raw_data[col], 10)
        raw1_data=pd.concat([raw1_data,raw_buckets],axis=1)
    return raw1_data


# In[20]:


def Col_removal(Bucket_Raw_Data,Variable_Dict,json_input,json_data_final):
    keycol=json_data_final['Primary_Column'].get('Selected_Value')
    tc_flag=json_data_final['Secondary_Column'].get('Selected_Value')
    All_Integer=json_input['All_Integer']
    raw_data_categorical=Variable_Dict['raw_data_categorical']
    tc_flag_df=Variable_Dict['tc_flag_df']
    keycol_df=Variable_Dict['keycol_df']
    
    raw1_data=Bucket_Raw_Data
    A=[]
    for i in range(0,len(raw1_data.columns)):


        col=raw1_data.columns[i]
        Value_count=raw1_data[raw1_data.columns[i]].value_counts([1]).reset_index()
        Value_count.drop('index',axis=1,inplace=True)
        Value_count.is_copy= False

        for y in range(0,10):
            if Value_count.iloc[y][0]>0.995:
                A.append(raw1_data.columns[i])
                print(A)
            else:
                pass         
    raw1_data.drop(A,inplace=True,axis=1)
    if All_Integer=='No':
        raw1_data=pd.concat([raw1_data,pd.DataFrame(raw_data_categorical).set_index(raw1_data.index)],axis=1)
    else:
        raw1_data=raw1_data
    tc_flag_df1=pd.concat([keycol_df,tc_flag_df],axis=1).set_index([keycol],drop=True)
    raw1_data=pd.concat([tc_flag_df1,raw1_data],axis=1)
    Col_Output={'raw1_data':raw1_data,'A':A}
    return Col_Output


# In[21]:


def Binning_makeover(Variable_Dict,json_data_final,Final_Col_Output):
    keycol=json_data_final['Primary_Column'].get('Selected_Value')
    tc_flag=json_data_final['Secondary_Column'].get('Selected_Value')
    raw_data_binning=Variable_Dict['raw_data_binning']
    tc_flag_df=Variable_Dict['tc_flag_df']
    A=Final_Col_Output['A']
    raw1_data=Final_Col_Output['raw1_data']
    i=0
    for x in range(0,len(raw1_data)):

        if raw1_data.iloc[x][0] ==0:
            i=i+1
        else:
            pass
    raw_data_binning.drop(A,axis=1,inplace=True)
    raw_data_binning=raw_data_binning.set_index([keycol])
    raw1_control=raw1_data[0:i]
    raw1_test=raw1_data[i:]
    tc_flag_control=tc_flag_df[0:len(raw1_control)]
    tc_flag_test=tc_flag_df[len(raw1_control):]
    print(tc_flag_control.shape)
    print(tc_flag_test.shape)
    raw1_control=raw1_control.drop(tc_flag,axis=1)
    raw1_test=raw1_test.drop(tc_flag,axis=1)
    Binning_TC_Out={'raw1_test':raw1_test,'raw1_control':raw1_control,'tc_flag_test':tc_flag_test,'tc_flag_control':tc_flag_control,'raw_data_binning':raw_data_binning}
    return Binning_TC_Out


# In[22]:


def Groups(Bin_Variable_Dict):
    raw1_control=Bin_Variable_Dict['raw1_control']
    raw1_test=Bin_Variable_Dict['raw1_test']
    Group_columns=list(raw1_control.columns)
    Groups_control=raw1_control.groupby(Group_columns).groups
    Groups_control=pd.DataFrame(list(Groups_control.items()), columns=['Groups', 'Userid'])
    Groups_control_list=Groups_control['Userid'].astype(list)
    Groups_control_grps_list=Groups_control['Groups'].astype(list)

    Groups_test=raw1_test.groupby(Group_columns).groups
    Groups_test=pd.DataFrame(list(Groups_test.items()), columns=['Groups', 'Userid'])
    Groups_test_list=Groups_test['Userid'].astype(list)
    Groups_test_grps_list=Groups_test['Groups'].astype(list)
    
    Groups_Output={'Groups_control_list':Groups_control_list,'Groups_test_list':Groups_test_list,'Groups_control_grps_list':Groups_control_grps_list,'Groups_test_grps_list':Groups_test_grps_list,'Groups_control':Groups_control,'Groups_test':Groups_test}
    return Groups_Output


# In[23]:


def Pairing(Groups_Output):
    Groups_control_list=Groups_Output['Groups_control_list']
    Groups_control_grps_list=Groups_Output['Groups_control_grps_list']
    Groups_test_list=Groups_Output['Groups_test_list']
    Groups_test_grps_list=Groups_Output['Groups_test_grps_list']
    Index_test=[]
    Index_control=[]
    for x in range(0,len(Groups_test_list)):

        for y in range(0,len(Groups_control_list)):
            if Groups_test_grps_list[x] == Groups_control_grps_list[y]:
                Index_test.append(Groups_test_list[x])
                Index_control.append(Groups_control_list[y])
            else:
                pass    
    Pairs=pd.DataFrame(np.column_stack([Index_test,Index_control]),columns=['Index_test','Index_control'])
    Test_index_f=[]
    Control_index_f=[]
    for i in range(0,len(Pairs)):
            Test_var = [x for x in Pairs.iloc[i][0] if re.search("User" , x)]
            Test_index_f.append(Test_var)
    for i in range(0,len(Pairs)):
            Control_var = [x for x in Pairs.iloc[i][1] if re.search("User" , x)]
            Control_index_f.append(Control_var)
    print(len(Test_index_f))
    print(len(Control_index_f))
    Pairs_Final=pd.DataFrame(np.column_stack([Test_index_f,Control_index_f]),columns=['Index_test','Index_control'])
    Pairing_Output={'Pairs_Final':Pairs_Final,'Index_test':Index_test,'Index_control':Index_control}
    return Pairing_Output


# In[24]:


## Matching
def Matching(Pairing_Output,Bin_Variable_Dict,json_data_final):
    keycol=json_data_final['Primary_Column'].get('Selected_Value')
    tc_flag=json_data_final['Secondary_Column'].get('Selected_Value')
    Pairs_Final=Pairing_Output['Pairs_Final']
    Index_test=Pairing_Output['Index_test']
    Index_control=Pairing_Output['Index_control']
    raw_data_binning=Bin_Variable_Dict['raw_data_binning']
    raw1_test=Bin_Variable_Dict['raw1_test']
    raw1_control=Bin_Variable_Dict['raw1_control']
    for i in range(0,len(Pairs_Final)):
        if len(Index_test[i])<=len(Index_control[i]):
            pass
        else:
            Pairs_Final=Pairs_Final.drop(Pairs_Final.index[i])

    Index_list_test=list(Pairs_Final['Index_test'])
    Test_pair_final_grp=pd.DataFrame()
    for i in range(0,len(Pairs_Final)):
        Test_pair_final_grp=Test_pair_final_grp.append(raw_data_binning.loc[Index_list_test[i]])
    Semi_Final= pd.DataFrame()
    for i in range(0,len(Pairs_Final)):
        for y in range(0,len(Index_test[i])):
            a=raw_data_binning.loc[Index_test[i][y]]
            Semi_Final=Semi_Final.append(a)
            b=raw_data_binning.loc[Index_control[i][y]]
            Semi_Final=Semi_Final.append(b)
    Semi_Final_control= Semi_Final[Semi_Final[tc_flag]==0]
    Semi_Final_test= Semi_Final[Semi_Final[tc_flag]==1]

    Semi_Final_control= Semi_Final_control[raw1_test.columns]
    Semi_Final_test= Semi_Final_test[raw1_control.columns]   
    Matching_Out={'Semi_Final_control':Semi_Final_control,'Semi_Final_test':Semi_Final_test,'Test_pair_final_grp':Test_pair_final_grp}
    return Matching_Out


# In[25]:


def Goku_Bin(Cat_list,Matching_Output,Variable_Dict):
    Categorical_list=Variable_Dict['Categorical_list']
    Semi_Final_control=Matching_Output['Semi_Final_control']
    Semi_Final_test=Matching_Output['Semi_Final_test']
    control_pair_final=Semi_Final_control[Semi_Final_control['categorical']==Categorical_list[Cat_list]]
    test_pair_final=Semi_Final_test[Semi_Final_test['categorical']==Categorical_list[Cat_list]]
    control_pair_final=control_pair_final.drop('categorical',axis=1)
    test_pair_final=test_pair_final.drop('categorical',axis=1)
    Test_Control_Pair_Final={'test_pair_final':test_pair_final,'control_pair_final':control_pair_final}
    return Test_Control_Pair_Final


# In[26]:


def Goku_Bin_int(Matching_Output,Variable_Dict):
    Semi_Final_control=Matching_Output['Semi_Final_control']
    Semi_Final_test=Matching_Output['Semi_Final_test']
    control_pair_final=Semi_Final_control
    test_pair_final=Semi_Final_test
    Test_Control_Pair_Final={'test_pair_final':test_pair_final,'control_pair_final':control_pair_final}
    return Test_Control_Pair_Final


# In[27]:


def T_test_Func(Goku_Bin1,json_data_final):
    test_pair_final=Goku_Bin1['test_pair_final']
    control_pair_final=Goku_Bin1['control_pair_final']
    ttest=[]
    ttest_array=ttest_ind(test_pair_final,control_pair_final,axis=0)
    x=0
    while x<len(ttest_array.pvalue):

        if ttest_array.pvalue[x]>0.05:
            ttest.append('pass')
            print('pass')   
        else:
            print('fail') 
            ttest.append('fail')
        x=x+1
    print(ttest_array.pvalue)
    Test_mean=[]
    for i in range(0,len(test_pair_final.columns)):
        Test_mean.append(test_pair_final[test_pair_final.columns[i]].mean())

    Control_mean=[]
    for i in range(0,len(control_pair_final.columns)):
        Control_mean.append(control_pair_final[control_pair_final.columns[i]].mean())

    Test_Total_rows=[]
    for i in range(0,len(test_pair_final.columns)):
        Test_Total_rows.append(len(test_pair_final))

    Control_Total_rows=[]
    for i in range(0,len(control_pair_final.columns)):
        Control_Total_rows.append(len(control_pair_final))
        
    Group_T_Test={'ttest_array':ttest_array,'ttest':ttest,'Test_mean':Test_mean,'Control_mean':Control_mean,'Test_Total_Rows':Test_Total_rows,'Control_Total_Rows':Control_Total_rows}

    return Group_T_Test


def T_test_Func_Int(Goku_Bin_Int1,json_data_final):
    test_pair_final=Goku_Bin_Int1['test_pair_final']
    control_pair_final=Goku_Bin_Int1['control_pair_final']
    ttest=[]
    ttest_array=ttest_ind(test_pair_final,control_pair_final,axis=0)
    x=0
    while x<len(ttest_array.pvalue):

        if ttest_array.pvalue[x]>0.05:
            ttest.append('pass')
            print('pass')   
        else:
            print('fail') 
            ttest.append('fail')
        x=x+1
    print(ttest_array.pvalue)
    Test_mean=[]
    for i in range(0,len(test_pair_final.columns)):
        Test_mean.append(test_pair_final[test_pair_final.columns[i]].mean())

    Control_mean=[]
    for i in range(0,len(control_pair_final.columns)):
        Control_mean.append(control_pair_final[control_pair_final.columns[i]].mean())

    Test_Total_rows=[]
    for i in range(0,len(test_pair_final.columns)):
        Test_Total_rows.append(len(test_pair_final))

    Control_Total_rows=[]
    for i in range(0,len(control_pair_final.columns)):
        Control_Total_rows.append(len(control_pair_final))
        
    Group_T_Test_Int={'ttest_array':ttest_array,'ttest':ttest,'Test_mean':Test_mean,'Control_mean':Control_mean,'Test_Total_Rows':Test_Total_rows,'Control_Total_Rows':Control_Total_rows}

    return Group_T_Test_Int


# # Group Wrapper Fuction

# In[28]:


def Group_Calling(json_data_final):
    json_input=runalgorithm_variables(json_data_final)
    Variable_Dict=Creating_raw_data(json_input,json_data_final)
    All_Integer=json_input['All_Integer']
    Pairs_Final=pd.DataFrame()
    T_test_Summary=pd.DataFrame()
    Categorical_list=list(Variable_Dict['Categorical_list'])
    
    Bucket_Raw_Data=Bucketing(Variable_Dict,json_input,json_data_final)
    Final_Col_Output=Col_removal(Bucket_Raw_Data,Variable_Dict,json_input,json_data_final)
    Bin_Variable_Dict=Binning_makeover(Variable_Dict,json_data_final,Final_Col_Output)
    Groups_Output=Groups(Bin_Variable_Dict)
    Pairing_Output=Pairing(Groups_Output)
    Matching_Output=Matching(Pairing_Output,Bin_Variable_Dict,json_data_final)
    for i in range(0,len(Categorical_list)):
        if All_Integer=='No':
            Goku_Bin1=Goku_Bin(i,Matching_Output,Variable_Dict)
            T_test_Calling=T_test_Func(Goku_Bin1,json_data_final)
            ttest_array=T_test_Calling['ttest_array']
            ttest=T_test_Calling['ttest']
            Test_mean=T_test_Calling['Test_mean']
            Control_mean=T_test_Calling['Control_mean']
            Test_Total_Rows=T_test_Calling['Test_Total_Rows']
            Control_Total_Rows=T_test_Calling['Control_Total_Rows']
            test_control_pair_rd=Pairing_Output['Pairs_Final']
            control_pair_final=Goku_Bin1['control_pair_final']
            test_pair_final=Goku_Bin1['test_pair_final']

            TC_pair_1=pd.DataFrame(test_control_pair_rd)
            Pairs_Final=pd.concat([Pairs_Final,TC_pair_1],axis=1)
            Final_categorical_list=[]
            for j in range(0,len(control_pair_final.columns)):
                Final_categorical_list.append(list(Categorical_list)[i])
                                                  
            T_test_Final=pd.DataFrame(np.column_stack([list(Final_categorical_list),list(control_pair_final.columns),Test_mean,Control_mean,Test_Total_Rows,Control_Total_Rows,list(ttest_array.pvalue),ttest]),columns=['Categorical Column','Columns','Test Mean','Control Mean','Test Count','Control Count','P Value','Pass/Fail'])
            T_test_Summary=pd.concat([T_test_Summary,T_test_Final])
            t = time.localtime()
            timestamp = time.strftime('%b-%d-%Y_%H%M', t)
            wd = os.getcwd()
            wd=wd.replace('\\','/')
            Pairs_Final.to_csv('Pairs_Final'+timestamp+'.csv')
            T_test_Summary.to_csv('T_test_Summary'+timestamp+".csv")

            Output_Module_1={}
            Output_Module_1['T-test'] = {"Display_Name":"T_test_Summary.csv",
                   "url": wd+"/T_test_Summary"+timestamp+".csv"},
            Output_Module_1["Pairs_Data"]={"Display_Name": "Pairs_Final.csv",
                    "url": wd + "/Pairs_Final"+timestamp+".csv"}
            Output_Module_1['T_Test_Table']=[]
            for i in range(0,len(T_test_Summary)):
                data4={"Categorical_Column":T_test_Summary.iloc[i][0],"Columns":T_test_Summary.iloc[i][1],"Test_Mean":T_test_Summary.iloc[i][2],"Control_Mean":T_test_Summary.iloc[i][3],"Test_Count":T_test_Summary.iloc[i][4],"Control_Count":T_test_Summary.iloc[i][5],"P_Value":T_test_Summary.iloc[i][6],"Pass/Fail":T_test_Summary.iloc[i][7]}
                Output_Module_1['T_Test_Table'].append(data4)
           
        else:
            Goku_Bin_Int1=Goku_Bin_int(Matching_Output,Variable_Dict)
            T_test_Calling_Int=T_test_Func_Int(Goku_Bin_Int1,json_data_final)
            ttest_array=T_test_Calling_Int['ttest_array']
            ttest=T_test_Calling_Int['ttest']
            Test_mean=T_test_Calling_Int['Test_mean']
            Control_mean=T_test_Calling_Int['Control_mean']
            Test_Total_Rows=T_test_Calling_Int['Test_Total_Rows']
            Control_Total_Rows=T_test_Calling_Int['Control_Total_Rows']
            test_control_pair_rd=Pairing_Output['Pairs_Final']
            control_pair_final=Goku_Bin_Int1['control_pair_final']
            test_pair_final=Goku_Bin_Int1['test_pair_final']

            TC_pair_1=pd.DataFrame(test_control_pair_rd)
            Pairs_Final=pd.concat([Pairs_Final,TC_pair_1],axis=1)
            T_test_Final=pd.DataFrame(np.column_stack([list(control_pair_final.columns),Test_mean,Control_mean,Test_Total_Rows,Control_Total_Rows,list(ttest_array.pvalue),ttest]),columns=['Columns','Test_Mean','Control_Mean','Test_Count','Control_Count','P_Value','Pass/Fail'])
            T_test_Summary=pd.concat([T_test_Summary,T_test_Final])
            t = time.localtime()
            timestamp = time.strftime('%b-%d-%Y_%H%M', t)
            wd = os.getcwd()
            wd=wd.replace('\\','/')
            Pairs_Final.to_csv('Pairs_Final'+timestamp+'.csv')
            T_test_Summary.to_csv('T_test_Summary'+timestamp+".csv")

            Output_Module_1={}
            Output_Module_1['T-test'] = {"Display_Name":"T_test_Summary.csv",
                   "url": wd+"/T_test_Summary"+timestamp+".csv"},
            Output_Module_1["Pairs_Data"]={"Display_Name": "Pairs_Final.csv",
                    "url": wd + "/Pairs_Final"+timestamp+".csv"}
            Output_Module_1['T_Test_Table']=[]
            for i in range(0,len(T_test_Summary)):
                data4={"Columns":T_test_Summary.iloc[i][0],"Test_Mean":T_test_Summary.iloc[i][1],"Control_Mean":T_test_Summary.iloc[i][2],"Test_Count":T_test_Summary.iloc[i][3],"Control_Count":T_test_Summary.iloc[i][4],"P_Value":T_test_Summary.iloc[i][5],"Pass/Fail":T_test_Summary.iloc[i][6]}
                Output_Module_1['T_Test_Table'].append(data4)

    return Output_Module_1  
        


# # MAIN_WRAPPER

# In[37]:


def Main_Caller(json_data_final):
    Algorithm=json_data_final['Algorithm'].get('Selected_Value')
    if Algorithm=='greedy':
        Final=Greedy_Calling(json_data_final)
    else:
        Final=Group_Calling(json_data_final)
    return Final


# In[38]:


Answer=Main_Caller(json_data_final)


# In[39]:


Answer

