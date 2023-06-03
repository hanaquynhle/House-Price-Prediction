import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt
import seaborn as sns


# =============================================================================
def main():
    # Read the original data files
    trainDF = pd.read_csv("data/train.csv")
    testDF = pd.read_csv("data/test.csv")

    demonstrateHelpers(trainDF)

    trainInput, testInput, trainOutput, testIDs, predictors = transformData(trainDF, testDF)
    
    doExperiment(trainInput, trainOutput, predictors)
    
    doExperiment2 (trainInput, trainOutput, predictors, 0.15)
    
    doExperiment3(trainInput, trainOutput, predictors, 0.15)
    
    doExperiment4(trainInput, trainOutput, predictors, 0.15)
    
    doKaggleTest(trainInput, testInput, trainOutput, testIDs, predictors)
    
    tuneGBR(trainInput, trainOutput, predictors)
    
    tuneLinearRidge(trainInput, trainOutput, predictors)
    
    tuneLasso(trainInput, trainOutput, predictors)
    
# ===============================================================================
'''
Does k-fold CV on the Kaggle training set using LinearRegression.
(You might review the discussion in hw06 about the so-called "Kaggle training set"
versus other sets.)
'''
def doExperiment(trainInput, trainOutput, predictors):
    alg = LinearRegression()
    cvMeanScore = model_selection.cross_val_score(alg, trainInput.loc[:, predictors], trainOutput, cv=10, scoring='r2', n_jobs=-1).mean()
    print("CV Average Score:", cvMeanScore)

#Ridge regression is a linear regression extension in which the loss function is adjusted to 
#minimize model's complexity

def doExperiment2(trainInput, trainOutput, predictors, x):
    alg = linear_model.Ridge(alpha=x, copy_X=True, fit_intercept=True, max_iter=None, normalize=False, random_state=None, solver='auto', tol=0.001)
    cvMeanScore = model_selection.cross_val_score(alg, trainInput.loc[:, predictors], trainOutput, cv=10, scoring='r2', n_jobs=-1).mean()
    print("ridge CV Average Score:", cvMeanScore)
    return cvMeanScore

# X is the alpha which is the learning rate that is how big of a step we want to take 
# Gradient boosting is a machine learning technique used in regression and classification tasks,
# among others. It gives a prediction model in the form of an ensemble of weak prediction models, 
# which are typically decision trees.

def doExperiment3(trainInput, trainOutput, predictors, x):
    gbrt=GradientBoostingRegressor(n_estimators=100,learning_rate=x) 
    cvMeanScore = model_selection.cross_val_score(gbrt, trainInput.loc[:, predictors], trainOutput, cv=10, scoring='r2', n_jobs=-1).mean() 
    print("GBR CV Average Score:", cvMeanScore)
    return cvMeanScore

#The lasso regression allows you to shrink or regularize these coefficients to avoid overfitting 
#and make them work better on different datasets. 

def doExperiment4(trainInput, trainOutput, predictors , x):
    alg = Lasso(alpha=x)
    cvMeanScore = model_selection.cross_val_score(alg, trainInput.loc[:, predictors], trainOutput, cv=10, scoring='r2', n_jobs=-1).mean()
    print("CV Average Score for Lasso:", cvMeanScore)
    return cvMeanScore


# ===============================================================================

def tuneLinearRidge(trainInput,trainOutput,predictors):
    #alphaList = pd.Series([.0001,.001,.01,.03,.1,.3,1,1.3,2,100,1000])
    tuneSeqence = np.arange(.01,1,.01)
    alphaLs = pd.Series(tuneSeqence,index=tuneSeqence)
    acc = alphaLs.map(lambda x: doExperiment2(trainInput, trainOutput, predictors,x) )
    print("Highest Accuracy N Value:",acc.idxmax())
    print("\nResult from Ride Regression:",acc.max())
    plt.figure(figsize=(11,7))
    plt.plot(alphaLs, acc)
    plt.xlabel('Alpha')
    plt.ylabel('Accuracy')
    plt.title('Tuning of Linear Ridge')
    plt.grid(True)
    plt.savefig("TuningRidge.png", dpi=500, bbox_inches='tight')
    plt.show()
    

def tuneGBR(trainInput,trainOutput,predictors):
    
    #alphaList = pd.Series([.0001,.001,.01,.03,.1,.3,1,1.3,2,100,1000])
    tuneSequence = np.arange(.01,1,.01)
    alphaLs = pd.Series(tuneSequence,index=tuneSequence)
    acc = alphaLs.map(lambda x: doExperiment3(trainInput, trainOutput, predictors,x) ) #tuning the models we have
    print("Highest Accuracy N Value:",acc.idxmax())
    print("\nResult from GBR:",acc.max())
    plt.figure(figsize=(11,7))
    plt.plot(alphaLs, acc)
    plt.xlabel('Learning Rate')
    plt.ylabel('Accuracy')
    plt.title('Tuning of GBR')
    plt.grid(True)
    plt.savefig("TuningGBR.png", dpi=500, bbox_inches='tight')
    plt.show() 
    
def tuneLasso(trainInput,trainOutput,predictors):
    #alphaList = pd.Series([.0001,.001,.01,.03,.1,.3,1,1.3,2,100,1000])
    tuneSeq = np.arange(.01,1,.01)
    alphaList = pd.Series(tuneSeq,index=tuneSeq)
    acc = alphaList.map(lambda x: doExperiment4(trainInput, trainOutput, predictors, x) )
    print("Highest Accuracy N Value:",acc.idxmax())
    print("\nResult:",acc.max())
    plt.figure(figsize=(11,7))
    plt.plot(alphaList, ["{:.2f}".format(i) for i in acc])
    plt.xlabel('Alpha')
    plt.ylabel('Accuracy')
    plt.title('Tuning of Lasso')
    plt.grid(True)
    plt.savefig("TuningLasso.png", dpi=500, bbox_inches='tight')
    plt.show()

# =================================================================================
'''
Runs the algorithm on the testing set and writes the results to a csv file.
'''
def doKaggleTest(trainInput, testInput, trainOutput, testIDs, predictors):
    alg = LinearRegression()

    # Train the algorithm using all the training data
    alg.fit(trainInput.loc[:, predictors], trainOutput)

    # Make predictions on the test set.
    predictions = alg.predict(testInput.loc[:, predictors])

    # Create a new dataframe with only the columns Kaggle wants from the dataset.
    submission = pd.DataFrame({
        "Id": testIDs,
        "SalePrice": predictions
    })

    # Prepare CSV
    submission.to_csv('data/testResults.csv', index=False)
    # Now, this .csv file can be uploaded to Kaggle

# ============================================================================

# Data cleaning - conversion, normalization
'''
Pre-processing code will goes in this function
'''
def transformData(trainDF, testDF):
    
    numericAttributes = getNumericAttrs(trainDF).tolist()
    numericAttributes.remove("Id")
    numericAttributes.remove("SalePrice")
    
    StringAttributes = getNonNumericAttrs(trainDF).tolist()
    
    #dealing with Numeric and String missing values
    
    for col in numericAttributes:   
        trainDF.loc[:,col] = trainDF.loc[:,col].fillna(0)
        testDF.loc[:,col] = testDF.loc[:,col].fillna(0)
    
    for col in StringAttributes:
        if (testDF[col].isnull().sum()<10):
            fillNaWithMode(trainDF,testDF,col)
        else:
            trainDF.loc[:,col] = trainDF.loc[:,col].fillna('None')
            testDF.loc[:,col] = testDF.loc[:,col].fillna('None')
            
    #filled with none to make it string and not NA 
                
    #creating a YearOld attribute to calculate how old the house is
    trainDF['YearOld'] = 2022-trainDF['YearBuilt']
    testDF['YearOld'] = 2022-trainDF['YearBuilt']

    #converting the yearRemodAdd attribute to Yes = 1 or No = 0 based on whether it was remodeled or not
    yrRemodToBool(trainDF,testDF)
    
    #ATTRIBUTE ENGINEERING
    #creating total area to encompass the size of the house (how large it is)
    trainDF['TotalArea']= +trainDF['GarageArea']+ trainDF['TotalBsmtSF'] + trainDF['1stFlrSF'] + trainDF['2ndFlrSF'] + trainDF['GrLivArea'] 
    testDF['TotalArea']= +testDF['GarageArea']+ testDF['TotalBsmtSF'] + testDF['1stFlrSF'] + testDF['2ndFlrSF'] + testDF['GrLivArea'] 
   
    #creating a NumberofBathrooms attribute to encompass all bathrooms in the house
    trainDF['NumberOfBathrooms'] = trainDF.apply(lambda row: row['FullBath']+row['HalfBath']*(.5)+row['BsmtHalfBath']*(.5)+row['BsmtFullBath'],axis=1)
    testDF['NumberOfBathrooms'] = testDF.apply(lambda row: row['FullBath']+row['HalfBath']*(.5)+row['BsmtHalfBath']*(.5)+row['BsmtFullBath'],axis=1)
    
    #synthesizing and categorizing areas and characteristics of the house to be used as predictors     
    location = ['Neighborhood','Condition1','Condition2']
    squareFoot = ['TotalArea','BsmtFinSF1', 'GrLivArea','LowQualFinSF','TotalBsmtSF','LotArea','1stFlrSF','2ndFlrSF']
    ageTraits =['YearOld','YearBuilt'] 
    quality = ['ExterQual','ExterCond','OverallQual','OverallCond','FireplaceQu', 'GarageQual','KitchenQual','HeatingQC']
    rooms = ['TotRmsAbvGrd','NumberOfBathrooms','KitchenAbvGr','BedroomAbvGr']
      
    #normalization of numeric attributes
    cols = squareFoot + ['MSSubClass','LotFrontage','MasVnrArea', 'MoSold', 'OpenPorchSF', 'EnclosedPorch', 'WoodDeckSF' , 'BsmtUnfSF']  
    trainDF.loc [:, cols], testDF.loc [:,cols] = normalize (trainDF,testDF,cols)
        
    #changing nominal values to ascending ordinal values by calling the ordinalConversion funcion
    rankedAttributes = ['BsmtCond', 'BsmtExposure','ExterQual','ExterCond', 'BsmtQual','HeatingQC','KitchenQual', 'FireplaceQu', 'GarageQual', 'PoolQC'] 
    for i in rankedAttributes:
        ordinalConversion(trainDF,testDF,i)
  
    #converting nominal values to ascending ordinal values by calling roofStyleConversion function
    roofStyleConversion(trainDF,testDF,'RoofStyle')
    
    #roofMaterialConversion (trainDF,testDF,'RoofMatl')
    
    #converting nominal values to ascending ordinal values by calling roofStyleConversion function
    houseStyleConversion (trainDF, testDF, 'HouseStyle')
    
    predictors = squareFoot + ageTraits + rooms + ['ExterQual','ExterCond', 'OverallQual','KitchenQual', 'HeatingQC',
                                                   'isItRemodeled', 'OverallCond','Fireplaces', 'GarageYrBlt',
                                                   'YrSold', 'GarageCars','RoofStyle', 'HouseStyle']  
    
    #oneHotEncoding 
    
    #used stackOverflow for reference to learn about get_dummies()
    #link: https://stackoverflow.com/questions/37077254/is-it-possible-to-get-feature-names-from-pandas-get-dummies
    colsToOneHotEncode = ['Neighborhood', 'MSZoning', 'Street','SaleCondition', 'RoofMatl', 'BldgType']
    trainDF = pd.get_dummies(trainDF, columns = colsToOneHotEncode, prefix='Neighborhood')
    testDF = pd.get_dummies(testDF, columns = colsToOneHotEncode, prefix='Neighborhood')
    added_dummy_cols = [item for item in list(trainDF.columns.values) if item.startswith('Neighborhood')]
    
    predictors += added_dummy_cols

    trainInput = trainDF.loc[:, predictors]
    testInput = testDF.loc[:, predictors]
    
    '''
    Any transformations we do on the trainInput is done on the
    testInput the same way. (For example, using the exact same min and max, if
    we're doing normalization.)
    '''
   
    print('-------')
    print (trainDF.columns)
    trainOutput = trainDF.loc[:, 'SalePrice']
    testIDs = testDF.loc[:, 'Id']
    
    return trainInput, testInput, trainOutput, testIDs, predictors
    
# ===============================================================================
#helper functions 

def fillNaWithMean(trainDF,testDF,col):
    trainDF.loc[:, col] = trainDF.loc[:, col].fillna(round(trainDF.loc[:, col].mean()))
    testDF.loc[:, col] = trainDF.loc[:, col].fillna(round(trainDF.loc[:, col].mean()))
    
def fillNaWithMode(trainDF,testDF,col):
    trainDF.loc[:, col] = trainDF.loc[:, col].fillna(trainDF.loc[:, col].mode().loc[0])
    testDF.loc[:, col] = trainDF.loc[:, col].fillna(trainDF.loc[:, col].mode().loc[0])
        
def yrRemodToBool(trainDF,testDF):
    trainDF['YearRemodAdd'] = trainDF.apply(lambda row: 0 if row['YearRemodAdd']==row['YearBuilt'] else 1,axis=1)
    testDF['YearRemodAdd'] = testDF.apply(lambda row: 0 if row['YearRemodAdd']==row['YearBuilt'] else 1,axis=1)
    trainDF.rename(columns = {'YearRemodAdd':'isItRemodeled'}, inplace = True)
    testDF.rename(columns = {'YearRemodAdd':'isItRemodeled'}, inplace = True)

def normalize(trainDF, testDF, cols):
    newDF1= trainDF.loc[:,cols] = (trainDF.loc[:,cols]- trainDF.loc[:,cols].min()) / (trainDF.loc[:,cols].max() - trainDF.loc[:,cols].min())
    newDF2 = testDF.loc[:,cols] = (trainDF.loc[:,cols]- trainDF.loc[:,cols].min()) / (trainDF.loc[:,cols].max() - trainDF.loc[:,cols].min())
    return newDF1, newDF2

def standardize(trainDF,testDF, cols):
    newDF1 = trainDF.loc[:,cols] = (trainDF.loc[:,cols] - trainDF.loc[:,cols].mean()) / trainDF.loc[:,cols].std()
    newDF2 = testDF.loc[:,cols] = (trainDF.loc[:,cols] - trainDF.loc[:,cols].mean()) / trainDF.loc[:,cols].std()
    return newDF1, newDF2

def ordinalConversion(trainDF,testDF,col): #function that transforms the nominal values into ascending ordinal values based on ranking		

    trainDF.loc[:, col] = trainDF.loc[:, col].map(lambda v:  4 if v=="Ex" else v)
    trainDF.loc[:, col] = trainDF.loc[:, col].map(lambda v:  3 if v=="Gd" else v)
    trainDF.loc[:, col] = trainDF.loc[:, col].map(lambda v:  2 if v=="TA" else v)
    trainDF.loc[:, col] = trainDF.loc[:, col].map(lambda v:  1 if v=="Fa" else v)
    trainDF.loc[:, col] = trainDF.loc[:, col].map(lambda v:  0 if v=="Po" else v)
    trainDF.loc[:, col] = trainDF.loc[:, col].map(lambda v:  0 if v=="NA" else v)
    trainDF.loc[:, col] = trainDF.loc[:, col].fillna(trainDF.loc[:, col].mode().loc[0])
    
    testDF.loc[:,col] = testDF.loc[:,col].map(lambda v: 4 if v=="Ex" else v)
    testDF.loc[:,col] = testDF.loc[:,col].map(lambda v: 3 if v=="Gd" else v)
    testDF.loc[:,col] = testDF.loc[:,col].map(lambda v: 2 if v=="TA" else v)
    testDF.loc[:,col] = testDF.loc[:,col].map(lambda v: 1 if v=="Fa" else v)
    testDF.loc[:,col] = testDF.loc[:,col].map(lambda v: 0 if v=="Po" else v)
    testDF.loc[:, col] = testDF.loc[:, col].map(lambda v:  0 if v=="NA" else v)
    testDF.loc[:, col] = testDF.loc[:, col].fillna(trainDF.loc[:, col].mode().loc[0])

def roofStyleConversion(trainDF,testDF,col): #function that transforms the nominal values into ascending ordinal values based on ranking where we have Gable price as the base value since it is the cheapest	

    trainDF.loc[:, col] = trainDF.loc[:, col].map(lambda v:  9.33 if v=="Mansard" else v)
    trainDF.loc[:, col] = trainDF.loc[:, col].map(lambda v:  7.6 if v=="Gambrel" else v)
    trainDF.loc[:, col] = trainDF.loc[:, col].map(lambda v:  6.66 if v=="Hip" else v)
    trainDF.loc[:, col] = trainDF.loc[:, col].map(lambda v:  3.83 if v=="Flat" else v)
    trainDF.loc[:, col] = trainDF.loc[:, col].map(lambda v:  2.5 if v=="Shed" else v)
    trainDF.loc[:, col] = trainDF.loc[:, col].map(lambda v:  1.0 if v=="Gable" else v)


    testDF.loc[:, col] = testDF.loc[:, col].map(lambda v:  9.33 if v=="Mansard" else v)
    testDF.loc[:, col] = testDF.loc[:, col].map(lambda v:  7.6 if v=="Gambrel" else v)
    testDF.loc[:, col] = testDF.loc[:, col].map(lambda v:  6.66 if v=="Hip" else v)
    testDF.loc[:, col] = testDF.loc[:, col].map(lambda v:  3.83 if v=="Flat" else v)
    testDF.loc[:, col] = testDF.loc[:, col].map(lambda v:  2.5 if v=="Shed" else v)
    testDF.loc[:, col] = testDF.loc[:, col].map(lambda v:  1.0 if v=="Gable" else v)

def houseStyleConversion(trainDF,testDF,col): #function that trasforms nominal values into ascending ordinal values based on ranking where we have 1Story as the base	 value

    trainDF.loc[:, col] = trainDF.loc[:, col].map(lambda v:  1.25 if v=="SLvl" else v)
    trainDF.loc[:, col] = trainDF.loc[:, col].map(lambda v:  1.25 if v=="SFoyer" else v)
    trainDF.loc[:, col] = trainDF.loc[:, col].map(lambda v:  2.10 if v=="2.5Unf" else v)
    trainDF.loc[:, col] = trainDF.loc[:, col].map(lambda v:  2.25 if v=="2.5Fin" else v)
    trainDF.loc[:, col] = trainDF.loc[:, col].map(lambda v:  2.0 if v=="2Story" else v)
    trainDF.loc[:, col] = trainDF.loc[:, col].map(lambda v:  1.25 if v=="1.5Unf" else v)
    trainDF.loc[:, col] = trainDF.loc[:, col].map(lambda v:  1.5 if v=="1.5Fin" else v)
    trainDF.loc[:, col] = trainDF.loc[:, col].map(lambda v:  1.0 if v=="1Story" else v)

    
    testDF.loc[:, col] = testDF.loc[:, col].map(lambda v:  1.25 if v=="SLvl" else v)
    testDF.loc[:, col] = testDF.loc[:, col].map(lambda v:  1.25 if v=="SFoyer" else v)
    testDF.loc[:, col] = testDF.loc[:, col].map(lambda v:  2.10 if v=="2.5Unf" else v)
    testDF.loc[:, col] = testDF.loc[:, col].map(lambda v:  2.25 if v=="2.5Fin" else v)
    testDF.loc[:, col] = testDF.loc[:, col].map(lambda v:  2.0 if v=="2Story" else v)
    testDF.loc[:, col] = testDF.loc[:, col].map(lambda v:  1.25 if v=="1.5Unf" else v)
    testDF.loc[:, col] = testDF.loc[:, col].map(lambda v:  1.5 if v=="1.5Fin" else v)
    testDF.loc[:, col] = testDF.loc[:, col].map(lambda v:  1.0 if v=="1Story" else v)
    
'''
def roofMaterialConversion(trainDF,testDF,col): #function that transforms nominal values into ascending ordinal values based on ranking where we khave Tar&Grv as the base value

    trainDF.loc[:, col] = trainDF.loc[:, col].map(lambda v:  18.57 if v=="Roll" else v)
    trainDF.loc[:, col] = trainDF.loc[:, col].map(lambda v:  4.67 if v=="Metal" else v)
    trainDF.loc[:, col] = trainDF.loc[:, col].map(lambda v:  2.21 if v=="WdShake" else v)
    trainDF.loc[:, col] = trainDF.loc[:, col].map(lambda v:  2.07 if v=="Membran" else v)
    trainDF.loc[:, col] = trainDF.loc[:, col].map(lambda v:  1.65 if v=="WdShngl" else v)
    trainDF.loc[:, col] = trainDF.loc[:, col].map(lambda v:  1.57 if v=="ClyTile" else v)
    trainDF.loc[:, col] = trainDF.loc[:, col].map(lambda v:  1.42 if v=="CompShg" else v)
    trainDF.loc[:, col] = trainDF.loc[:, col].map(lambda v:  1.0 if v=="Tar&Grv" else v)

    
    testDF.loc[:, col] = testDF.loc[:, col].map(lambda v:  18.57 if v=="Roll" else v)
    testDF.loc[:, col] = testDF.loc[:, col].map(lambda v:  4.67 if v=="Metal" else v)
    testDF.loc[:, col] = testDF.loc[:, col].map(lambda v:  2.21 if v=="WdShake" else v)
    testDF.loc[:, col] = testDF.loc[:, col].map(lambda v:  2.07 if v=="Membran" else v)
    testDF.loc[:, col] = testDF.loc[:, col].map(lambda v:  1.65 if v=="WdShngl" else v)
    testDF.loc[:, col] = testDF.loc[:, col].map(lambda v:  1.57 if v=="ClyTile" else v)
    testDF.loc[:, col] = testDF.loc[:, col].map(lambda v:  1.42 if v=="CompShg" else v)
    testDF.loc[:, col] = testDF.loc[:, col].map(lambda v:  1.0 if v=="Tar&Grv" else v)
''' 
 # ===============================================================================
   
'''
Demonstrates some provided helper functions that you might find useful.
'''
def demonstrateHelpers(trainDF):
    print("Attributes with missing values:", getAttrsWithMissingValues(trainDF), sep='\n')
    
    numericAttrs = getNumericAttrs(trainDF)
    print("Numeric attributes:", numericAttrs, sep='\n')
    
    nonnumericAttrs = getNonNumericAttrs(trainDF)
    print("Non-numeric attributes:", nonnumericAttrs, sep='\n')

    print("Values, for each non-numeric attribute:", getAttrToValuesDictionary(trainDF.loc[:, nonnumericAttrs]), sep='\n')

# ===============================================================================
'''
Returns a dictionary mapping an attribute to the array of values for that attribute.
'''
def getAttrToValuesDictionary(df):
    attrToValues = {}
    for attr in df.columns.values:
        attrToValues[attr] = df.loc[:, attr].unique()

    return attrToValues

# ===============================================================================
'''
Returns the attributes with missing values.
'''
def getAttrsWithMissingValues(df):
    valueCountSeries = df.count(axis=0)  # 0 to count down the rows
    numCases = df.shape[0]  # Number of examples - number of rows in the data frame
    missingSeries = (numCases - valueCountSeries)  # A Series showing the number of missing values, for each attribute
    attrsWithMissingValues = missingSeries[missingSeries != 0].index
    return attrsWithMissingValues

# =============================================================================

'''
Returns the numeric attributes.
'''
def getNumericAttrs(df):
    return __getNumericHelper(df, True)

'''
Returns the non-numeric attributes.
'''
def getNonNumericAttrs(df):
    return __getNumericHelper(df, False)

def __getNumericHelper(df, findNumeric):
    isNumeric = df.applymap(np.isreal) # np.isreal is a function that takes a value and returns True (the value is real) or False
                                       # applymap applies the given function to the whole data frame
                                       # So this returns a DataFrame of True/False values indicating for each value in the original DataFrame whether it is real (numeric) or not

    isNumeric = isNumeric.all() # all: For each column, returns whether all elements are True
    attrs = isNumeric.loc[isNumeric==findNumeric].index # selects the values in isNumeric that are <findNumeric> (True or False)
    return attrs

# =============================================================================

if __name__ == "__main__":
    main()

