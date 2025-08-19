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
