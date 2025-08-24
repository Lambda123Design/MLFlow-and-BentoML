# MLFlow-and-BentoML

### In above code, we didn't directly go ahead and registered; We are logging the parameters; And then we can register it

### Here it isn't showing like it is already registered; It is asking us to Register

### How to know if need to register it; Compare the model run and see with metrics

### We need to validate the models and then only to register it with help of mlflow UI

## log the model using TensorFlow - mlflow.tensorflow.log_model(model,"model",signature=signature)

### Mlflow also Supports Tensorflow

### We used a Tensor Model and Keras is a wrap on Top of it

#### Signature defines the Schema of Input and Output

mlflow version 2.5.0

**MLFlow doesn't have any Tracking URI; Whatever we have in Local that will only come; But we can add, Remote Server Tracking URI**

**After we ran it, we got a folder called MLRuns; We got artifacts** 

### **Once we ran "mlflow ui", it gave the server link, which can copy paste which tracks the complete project; We will get files such as "MLModel", "conda.yaml", "model.pkl", "python_env.yaml", "requirements.txt" - With which we can completely reproduce the work in any environment**

### **We got the each metrics we logged (MAE, R2, RMSE); Parameters; Datasets, Description; Unique ID for the experiments**

**Each and every run gets logged**

### **How to work Collaboratively - Suppose, there is another developer in the same project, and he also wants to give his experiments and try to give it to me; We can select the runs and give the "Compare" option in MLFLow"; We get a graph showing three columns like a line image - alpha, l1_ratio, mse; We can conclude based from the line (We can also set a remote server)**

**Multiple Developers can collaboratively develop and give their experiments to us**

**Remote Server can be AWS, EC2 Instance**

### **Later, all the experimentations will happen in Remote Server and we can send our experiments there**

### **We can sign in into DagsHub using "GitHUb"**

**Git Working - git init, git add., git status; Get the Links from GitHub Repo - git commit -m "first commit"; git branch -M main; git remote add origin https://github.com/Lambda123-design/mlflow_project.git; git push origin main**

####### **MLFlow and DagsHub Integration:**

**1. Login to DagsHub with Email and connect it to GitHub Repository**

**2. Once LoggedIn, click on "Remote" and choose "Experiments"; Copy that code and put in "README" file in VS Code**

**3. If I send the copied, ML_Flow Tracking, Username, Password, you can execute the code and send all experiments to "DagsHub" Remote Server**

**Open GitBash**

**Export these:**

MLFLOW_TRACKING_URI= https://dagshub.com/Lambda123-design/mlflow_project.mlflow

MLFLOW_TRACKING_USERNAME=Lambda123-design     

MLFLOW_TRACKING_PASSWORD=fe8f410c56c46fee139df438504356a01c8644eb

### **Write code for "Remote Server URI (Give DagsHub URI) and below write code for "mlflow.set_tracking_uri(remote_server_uri)"**

**For getting Password (Token) - In Dagshub, top right --> Settings --> Tokens --> Create a New Token**

**Run python script.py**

## **DagsHub will get created with experiments; Same way we can see in MLFlow also, for multiple experiments too**

### **We can work collaboratively as a team too; We can ask everyone to perform different experiments, and ask them to push to same URI**

### **In MLFlow, we had staging and Production; We can also say which model is in staging and which is in Production (Since we have multiple versions)**

### **We can click into the model and transition it to "Production" (In MLFlow)**

**Code Notes:**

**(i) conda.yaml (Basic Information about ML Project)**

**(ii) ML Model - Entire details of ML Project (It is the package which we saw in MLFlow website as MLFlow Projects to package the data science solution (If we use all of those specific files, we can load it in any environment and build the solution)**

**(iii) python_env.yaml - Basic Python Dependencies**

**Notes:**

**1. argv - Positional Arguments for ElasticNet** -  alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5; l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

**2. To start MLFlow -  with mlflow.start_run():**

**3. mlflow.log_param() - Used to Track Parameters**

**4. We can give parameters while running the app.py file; "python app.py 0.3 0.7"; alpha, l1 ratio**

### **Create a "gitignore" file, before commiting to GitHUb****

#### Learning from Error:

**If using CMD, it is taking to "models" folder in venv**

**We connected GitBash to Dagshub Server and ran "python app.py" in that to take it to DagsHub**

Git Bash was using Anaconda's Python (C:\Users\ashwa\anaconda3\python.exe)

But mlflow is not installed in Anaconda's Python 

**Installed in GitBash using "python -m pip install mlflow"**


##### A) MLFlow First Project:

1. from mlflow.models import infer_signature - For inference

2. ### MLFLOW tracking
mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")

##create a new MLFLOW experiment
mlflow.set_experiment("MLFLOW Quickstart")

## Sstart an MLFLOW run
with mlflow.start_run():
    ## log the hyperparameters
    mlflow.log_params(params)
    ## Log the accuracy metrics
    mlflow.log_metric("accuracy",accuracy)
    # Set a tag that we can use to remind ourselves what this run was for
    mlflow.set_tag("Training Info", "Basic LR model for iris data")
    ## Infer the model signature
    signature=infer_signature(X_train,lr.predict(X_train))
    ## log the model
    model_info=mlflow.sklearn.log_model(
        sk_model=lr,
        artifact_path="iris_model",
        signature=signature,
        input_example=X_train,
        registered_model_name="tracking-quickstart",
    )

3. Run "mlflow ui" in command prompt and run the code

4. We could see all the logs in VS Code; Go to mlruns in VS Code and see everything in it; Metrics,params,etc.. We can also see requirements.txt,model,etc..

5. We can run another code with some parameters, train models, and compare in MLFlow

6. **It was like entire model was packed and shown in Artifcats in MLFlow with pkl files, requirements.txt,etc.**

7. We will get artifact link in MLFlow, which will be same as artifcats we see in VS Code

8. That Pickle file can be used for Deployment

9. We can do inferencing using "model_info.model_uri"

10. **After that we give list of Serving Playground as Inputs list (It is like a Test Data)**

from mlflow.models import validate_serving_input

model_uri = model_info.model_uri

# The model is logged with an input example. MLflow converts
# it into the serving payload format for the deployed model endpoint,
# and saves it to 'serving_input_payload.json'
serving_payload = """{
  "inputs": [
    [
      5.7,
      3.8,
      1.7,
      0.3
    ],
    [
      4.8,
      3.4,
      1.6,
      0.2
    ],
    [
      5.6,
      2.9,
      3.6,
      1.3
    ],
    .......
}"""

# Validate the serving payload works on the model
validate_serving_input(model_uri, serving_payload)

12. **Other way is we will be loading the model in the form of a Function**

Load the model back for prediction as a generic python function model

loaded_model=mlflow.pyfunc.load_model(model_info.model_uri)
predictions=loaded_model.predict(X_test)

iris_features_name=datasets.load_iris().feature_names

result=pd.DataFrame(X_test,columns=iris_features_name)
result["actual_class"]=y_test
result["predcited_class"]=predictions

## B) MLFlow Model Registry:

## MLFlow --> Experiments --> Artifacts --> Model Registered

### Below that we can see it's version and registered on 

### We can also go to another run and see version 1 

### Model Registry is like a store which will be making sure that all the models are stored with all functionalities of creating different functionalities, tags and many more things

## If we click on version, we can see different versions of models too

### In below code we directly went ahead and register both the models; It shouldn't be the case; We should save the best model

model_info=mlflow.sklearn.log_model(
    sk_model=lr,
    artifact_path="iris_model",
    signature=signature,
    input_example=X_train,
    registered_model_name="tracking-quickstart",

)

**1. Logging the Parameters**

##create a new MLFLOW experiment
mlflow.set_experiment("MLFLOW Quickstart")

## Sstart an MLFLOW run
with mlflow.start_run():
    ## log the hyperparameters
    mlflow.log_params(params)
    ## Log the accuracy metrics
    mlflow.log_metric("accuracy",1.0)
    # Set a tag that we can use to remind ourselves what this run was for
    mlflow.set_tag("Training Info2", "Basic LR model for iris data")
    ## Infer the model signature
    signature=infer_signature(X_train,lr.predict(X_train))
    ## log the model
    ## log the model
    model_info=mlflow.sklearn.log_model(
        sk_model=lr,
        artifact_path="iris_model",
        signature=signature,
        input_example=X_train,
    )

### In above code, we didn't directly go ahead and registered; We are logging the parameters; And then we can register it

### Here it isn't showing like it is already registered; It is asking us to Register

### How to know if need to register it; Compare the model run and see with metrics

### Later Krish Registered the best model and it automatically showed the version of it (v3)

### Once we go inside Models now, we can see models and versions and we can add Tag too (production:success); Alias (@success)

### When working as a team, this tag and alias will help in better understanding

#### 2) How to do Predictions

## Inferencing from model from model registry

import mlflow.sklearn
model_name="tracking-quickstart"
model_version="latest"

model_uri=f"models:/{model_name}/{model_version}"

model=mlflow.sklearn.load_model(model_uri)
model

model_uri

## Predicting Test Data 

y_pred_new=model.predict(X_test)
y_pred_new

### We can also add a key like "Still Working", so that team can know what is happening


### A) House Price Prediction: MLflow Project

## Prepraing the dataset
data=pd.DataFrame(housing.data,columns=housing.feature_names)
data['Price']=housing.target

1. Defining HyperParameter tuning code:

def hyperparameter_tuning(X_train,y_train,param_grid):
    rf=RandomForestRegressor()
    grid_search=GridSearchCV(estimator=rf,param_grid=param_grid,cv=3,n_jobs=-1,verbose=2,
                             scoring="neg_mean_squared_error")
    grid_search.fit(X_train,y_train)
    return grid_search


## 2. MLFlow Working:

## Split data into training and test sets
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20)

from mlflow.models import infer_signature
signature=infer_signature(X_train,y_train)

## Define the hyperparameter grid
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

## start the MLFLOW Experiments

with mlflow.start_run():
    ## Perform hyperparameter tuning
    grid_search=hyperparameter_tuning(X_train,y_train,param_grid)
    ## Get the best model
    best_model=grid_search.best_estimator_
    ## Evaluate the best model
    y_pred=best_model.predict(X_test)
    mse=mean_squared_error(y_test,y_pred)
    ## Log best parameters and metrics
    mlflow.log_param("best_n_estimators",grid_search.best_params_['n_estimators'])
    mlflow.log_param("best_max_depth", grid_search.best_params_['max_depth'])
    mlflow.log_param("best_min_samples_split", grid_search.best_params_['min_samples_split'])
    mlflow.log_param("best_min_samples_leaf", grid_search.best_params_['min_samples_leaf'])
    mlflow.log_metric("mse",mse)
    ## Tracking url
    mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")
    tracking_url_type_store=urlparse(mlflow.get_tracking_uri()).scheme
    if tracking_url_type_store !='file':
        mlflow.sklearn.log_model(best_model,"model",registered_model_name="Best Randomforest Model")
    else:
        mlflow.sklearn.log_model(best_model,"model",signature=signature)
    print(f"Best Hyperparameters: {grid_search.best_params_}")
    print(f"Mean Squared Error: {mse}")

3. Go to the folder and run "mlflow ui" and then set "Tracking URI" in the code

## If we didn't set "mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")". then "tracking_url_type_store=urlparse(mlflow.get_tracking_uri()).scheme", will return a value as "file", because we haven't set any URL

## If we set it, we will get scheme of the URI and that is what url parse does

### We wrote a else condition in the above mlflow code, when we didn't set URI too, in which we didn't say anything about registered_model

### We didn't set any experiment name; So it's name was "Default"


### B) Deep Learning ANN Model with MLFlow

### 1. Used HyperOpt Library - Used to do Hyperparameter Tuning in ANN

There was previous version of Numpy that was incompatible; So we uninstalled it

## from hyperopt import STATUS_OK,Trials,fmin,hp,tpe

import mlflow
from mlflow.models import infer_signature

## 2. Load the dataset
data=pd.read_csv(
    "https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-white.csv",
    sep=";",
)

### This is a Classification Project, to predict Wine Quality

## 3. Split the data into training,validation and test sets

train,test=train_test_split(data,test_size=0.25,random_state=42)

train_x=train.drop(['quality'],axis=1).values

train_y=train[['quality']].values.ravel()

#### Usage of values.ravel in "train_y=train[['quality']].values.ravel()"

## train[['quality']].values - It is a 2D Array; 

## train_y=train[['quality']].values.ravel() - Gives a 1-D array and that is the reason we used it; Or else we need to reshape it using NumPy Array

## test dataset
test_x=test.drop(['quality'],axis=1).values
test_y=test[['quality']].values.ravel()

#### 4. We created further used Training Data into validation data from Train Data; We use Train and Valid data to check performance of model; We use test data as a New data and do predictions

## splitting this train data into train and validation

train_x,valid_x,train_y,valid_y=train_test_split(train_x,train_y,test_size=0.20,random_state=42)

signature=infer_signature(train_x,train_y)

### 4. Creating an ANN Model

def train_model(params,epochs,train_x,train_y,valid_x,valid_y,test_x,test_y):
    ## Define model architecture
    mean=np.mean(train_x,axis=0)
    var=np.var(train_x,axis=0)
    model=keras.Sequential(
        [
            keras.Input([train_x.shape[1]]),
            keras.layers.Normalization(mean=mean,variance=var),
            keras.layers.Dense(64,activation='relu'),
            keras.layers.Dense(1)
        ]
    )

mean=np.mean(train_x,axis=0) - For row wise it is going to find mean of all features

### This is required as we have to Perform Normalization when we are training our ANN, our Neural Network

### Next, we found out for Variance for row wise; Both we will use for our Layer Normalization

## Next we took Keras Sequential Model ; keras.Input([train_x.shape[1]] - We took all 11 Input Features and this Becomes our Input Layer

### Then we do Layer Normalization - keras.layers.Normalization(mean=mean,variance=var)

### Creating an Hidden Layer - keras.layers.Dense(64,activation='relu')

### Creating OutPut Layer - keras.layers.Dense(1)

We created an Simple ANN, with not much of Hidden Layers and all

### 5. Compiling the Model:

model.compile(optimizer=keras.optimizers.SGD(
        learning_rate=params["lr"],momentum=params["momentum"]
    ),
    loss="mean_squared_error",
    metrics=[keras.metrics.RootMeanSquaredError()]
    )

### For SGD, we need to give both Learning Rate and also Momentum

### learning_rate=params["lr"] - Params - By what parameter you want to try different different experiments; We can try with different different Hidden Layers; But it Takes Time; So we will try with different Learning Rates; Model will be trained with different different Learning Rates;

### Momentum is also another Parameter which we want to try with different values for better outcome;

### Other reason is also, we want to Log some experiments; That is why we use HyperOpt too, it will try with each and every Parameters we have given over there

### 6. Train the ANN model with lr and momentum params wwith MLFLOW tracking
    
with mlflow.start_run(nested=True):
        model.fit(train_x,train_y,validation_data=(valid_x,valid_y),
                  epochs=epochs,
                  batch_size=64)
        ## Evaluate the model
        eval_result=model.evaluate(valid_x,valid_y,batch_size=64)
        eval_rmse=eval_result[1]
        ## Log the parameters and results
        mlflow.log_params(params)
        mlflow.log_metric("eval_rmse",eval_rmse)
        ## log the model
        mlflow.tensorflow.log_model(model,"model",signature=signature)
        return {"loss": eval_rmse, "status": STATUS_OK, "model": model}

### It should also be in the Same function - Train the ANN model with lr and momentum params wwith MLFLOW tracking

### mlflow.start_run(nested=True) - Since we want to try with different different parameters, we have a option caled nested=True to do that; It will go inside and inside and follow a Nested Structure

### Training -    model.fit(train_x,train_y,validation_data=(valid_x,valid_y), epochs=epochs, batch_size=64)

### Evaluate the model - eval_result=model.evaluate(valid_x,valid_y,batch_size=64);

### eval_rmse=eval_result[1] - It will be in the form of a list, so we take first index to get RMSE; We want to track this and also Learning Rate, Momentum, etc.. 

## Log the parameters and results - mlflow.log_params(params); mlflow.log_metric("eval_rmse",eval_rmse)

## log the model using TensorFlow - mlflow.tensorflow.log_model(model,"model",signature=signature)

### Mlflow also Supports Tensorflow

### We used a Tensor Model and Keras is a wrap on Top of it; Signature defines the Schema of Input and Output

## It knows that this entire model is trained on Keras and Tensorflow and knows what are all the artifacts that needs to be create for this particular model

## Finally we returned everything - return {"loss": eval_rmse, "status": STATUS_OK, "model": model}

### We did everything step by step - Created ANN, added Normalization, compiled it, trained model, and while training and compiling we set different parameters for learning rate and momentum; And later Logger all Parameters and Metrics


### 6. Based on HyperOpt created an Objective Function:

def objective(params):
    # MLflow will track the parameters and results for each run
    result = train_model(
        params,
        epochs=3,
        train_x=train_x,
        train_y=train_y,
        valid_x=valid_x,
        valid_y=valid_y,
        test_x=test_x,
        test_y=test_y,
    )
    return result

**We gave our Training Mode, parameters, epochs, train,test data and finally returning the result**

### 7. Setting all Parameters:

space={
    "lr":hp.loguniform("lr",np.log(1e-5),np.log(1e-1)),
    "momentum":hp.uniform("momentum",0.0,1.0)

}

### From HyperOpt we used hp; It has for loguniform and uniform; It is like Hyperparameter Tuning; Learning Rate - It will be ranging from 10^-5 to 10^-1

### Momentum - "momentum":hp.uniform("momentum",0.0,1.0); 0 to 1

### 8. In Training we wrote start_run; Now we will write one more to run on top of it

mlflow.set_experiment("wine-quality")
with mlflow.start_run():
    # Conduct the hyperparameter search using Hyperopt
    trials=Trials()
    best=fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=4,
        trials=trials
    )
    # Fetch the details of the best run
    best_run = sorted(trials.results, key=lambda x: x["loss"])[0]
    # Log the best parameters, loss, and model
    mlflow.log_params(best)
    mlflow.log_metric("eval_rmse", best_run["loss"])
    mlflow.tensorflow.log_model(best_run["model"], "model", signature=signature)
    # Print out the best parameters and corresponding loss
    print(f"Best parameters: {best}")
    print(f"Best eval rmse: {best_run['loss']}")

## Start Run - To start the Experiment

### trials=Trials() - This performs HyperParameter Tuning

### best=fmin(fn=objective,space=space,algo=tpe.suggest, max_evals=4,trials=trials) - Function name - Objective; space - Hyperparameters we want to try with;  type.suggest - It will be using internally all kind different kind of algorithms itself and maximum evaluation; And we gave maximum evaluation = 4;  Trials - We gave our trials; We defined these for Hyperparameter Search; 

### best_run = sorted(trials.results, key=lambda x: x["loss"])[0] - Then we found the best details using this; We sort it using Loss function and find out which loss is less; We will be taking it as best run

### Trying to run that best (We defined) - mlflow.log_params(best)





###### BentoML:

After creating model, we create API's, integrate with Web Application it can be flask, streamlit, We also have to dockerization; We also have to do all those separately

What if we can do all those using single thing

**BentoML: Helps to do all till moving to Production**

**It helps in Inference Optimization;**

#### Has **Build Once; Deploy Anywhere; Bento is the file name which we can deploy in Any Cloud**

#### How it works: Define a model, save, Create a serve (Service will be responsible in Optimizing the inferences), Build a Bento, Deploy the Bento

#### We will create it and see how it creates API, Swaggger API's

For BentoML Python version should be greater than 3.8

**1. Creating venv - conda create -p venv python==3.9 -y** and  activate environment "conda activate venv"

2. Creating requirements.txt with bentoml==1.0.25, scikit-learn and running using "pip install -r requirements.txt

## With Model Training we can use BentoML and we can use app.py which does the prediction

3. Trained a Model using SVM; Didn't did train test split; Just trained the model

### Then we save the model to BentoML Local model store; Whenever we install BentoML, it will create a local repository in some drive; In that drive we will go and save particular classifier

### We will also be able to see versioning of model 

## 4. Save model to the BentoML Local Model Store
saved_model=bentoml.sklearn.save_model("iris_clf,clf")
print(f"Model Saved : {saved_model}")

### 5. If we ran using "python train.py", we will get

Model Saved : Model(tag="iris_clf:qyw5rauaa6zzrdch")

We will save that id in notebook to do inferencing 

6. "bentoml models list" - Shows models list; It also supports versioning

### It stores model in some local store - To see it - C - Users - user - bentoml - models - We can see models with pkl and yaml files

### It is the local bentoml model store; It will be by default in C Drive will get created

### How to read this model and do prediction

7. Created a test.py and ran

import bentoml 

iris_clf_runner=bentoml.sklearn.get("iris_clf:latest").to_runner()

iris_clf_runner.init_local()

print(iris_clf_runner.predict.run([[5.9,3,5.1,1.8]))

8. **Running "python test.py" - Got output as 2; Means it is classifying as Category 2**

### We didn't understand how BentoML helps to create API, Swagger UI; For that we need to write lot of codes; With respect to Flask, need to create API, write post request, get request as such 

### But in BentoML we don't need to do those; To get optimal model inferencing, we need to create "service.py"; Service.py will be responsible in creating all API's and helps to create swagger UI too 

import numpy as np

import bentoml

from bentoml.io import NumpyNdarray (Wrapper on top of numpy)

iris_clf_runner=bentoml.sklearn.get("iris_clf:latest").to_runner()

## If multiple models, say one for converting categorical to numerical, we can ran it, and then classifer, etc.. same way we can run in those orders in "runners=", can be given in the format of list

# Create the BentoService
svc = bentoml.Service("iris_classifier", runners=[iris_clf_runner])

# Define API endpoint
@svc.api(input=NumpyNdarray(), output=NumpyNdarray())
def classify(input_series: np.ndarray) -> np.ndarray:
    return iris_clf_runner.predict.run(input_series)

### 9. To run the file: bentoml serve service.py:svc --reload (We can reload based on any number of changes happening; svc - name we are giving)

**It ran in 0.0.0.0:3000**

#### Ran "http://localhost:3000/" in Browser

#### It takes to Swagger UI, which provides entire information; We can create any API's using code; It will be fully in form of Application JSON

**We went to POST and gave 4 Parameters; "[1,2.3,4.2,1.0]"; Parameters of Iris Flowe; We got output as "1" (as Response Body)**

### We can also use it with "POSTMAN"

### Use a Post Request and type "127.0.0.1:3000/classify"; Select "Body" and change it to "JSON" and give "[
  [1,2.3,4.2,1.0]
]"

If we click "Send", we will get same output

## We used "classify" because it is "POST/classify" in the website

If using Flask, we need to create it, use Swagger UI; But with BentoML, we were able to do it

### Everything is fine till here, for Local

### But we need to Package this application and send to Deployment Purpose

### We need to create a Bento, it will have all information about code, packages; It package application fully for production deployment

##### 10. Create "bentfile.yaml" (Naming should only be this, because once we write bento build, it will look for "bentofile.yaml" and wrote following code

service: "service.py:svc"
labels:
  owner:bentoml-team
  project:gallery
include:
- "*.py"
python:
  packages:
    - scikit-learn 
    - pandas

**Owner name we can give anything**

#### 11. Next Command: "bentoml build" (Once we write bento build, it will look for "bentofile.yaml")

#### We can also Containerize the Bento using Dockers

#### **12. bentoml list - See all the Bentos we have created**

**In bentoml folder in C Drive, we can see folder called "bentos"; It has the entire package**
