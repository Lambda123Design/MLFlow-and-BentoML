# MLFlow-and-BentoML

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

2. Creating requirements.txt with bentoml==1.0.25, scikit-learn
