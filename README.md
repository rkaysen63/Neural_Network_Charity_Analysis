# Neural_Network_Charity_Analysis

<p align="center">
  <a href="#">Neural Networks</a>
  <br/><br/> 
  <img src="Images/jj-ying-8bghKxNU1j0-unsplash.jpg" width="700">
</p>
  
## Table of Contents
* [Overview](https://github.com/rkaysen63/Neural_Network_Charity_Analysis/blob/master/README.md#overview)
* [Resources](https://github.com/rkaysen63/Neural_Network_Charity_Analysis/blob/master/README.md#resources)
* [Results](https://github.com/rkaysen63/Neural_Network_Charity_Analysis/blob/master/README.md#results)
* [Summary](https://github.com/rkaysen63/Neural_Network_Charity_Analysis/blob/master/README.md#summary)

## Resources:    
* Data: charity_data.csv
* Tools: 
  * Python
  * Colaboratory (Colab) notebook for writing code
  * Jupyter Notebook
* https://stackoverflow.com/questions/59069058/save-model-every-10-epochs-tensorflow-keras-v2
* Opening image courtesy of: Photo by JJ Ying on Unsplash (Nice photo JJ Ying!) 
* Lesson Plan: UTA-VIRT-DATA-PT-02-2021-U-B-TTH, Module 19 Challenge

## Overview:
Alphabet Soup is a charitable foundation that has funded over 34,000 organizations over the years.  The purpose of the funding is to assist organizations with their philanthropic projects.  The data collected shows that each organization in their database has either made an impact, i.e. "IS SUCCESSFUL", or not.  The purpose of this analysis is to create a model from the existing metadata about each organization that will predict whether or not a future applicant will be successful.  In other words, the purpose is to create a model that will help Alphabet Soup determine which future applications should be accepted or rejected.

## Results:

### Deliverable 1
<p align="center">
  <a href="#">application_df:  Data is loaded into a DataFrame using Pandas</a>
  <br/><br/> 
  <img src="Images/application_df1.png" width="700">
</p>

* Drop the non-beneficial ID columns, 'EIN' and 'NAME'.<br/><br/> 
  `application_df = application_df.drop(columns = ["EIN", "NAME"])`<br/><br/> 

<p align="center">
  <img src="Images/application_df2.png" width="700">
</p>  
  
* Group together columns with more than 10 unique values.<br/><br/> 

      # Determine the number of unique values in each column.    
      application_df.nunique()

<p align="center">
  <img src="Images/application_unique.png" width="200">
</p>  
  
      # Look at APPLICATION_TYPE value counts for binning.
      application_type_counts = application_df.APPLICATION_TYPE.value_counts()
      application_type_counts

<p align="center">
  <img src="Images/app_type_counts1.png" width="250">
</p>  
        
      # Visualize the value counts of APPLICATION_TYPE.
      application_type_counts.plot.density()

<p align="center">
  <img src="Images/app_type_density.png" width="400">
</p>  
        
    # Determine which values to replace if counts are less than ? (19.3.3)
    replace_application = list(application_type_counts[application_type_counts < 500].index)

    # Replace in dataframe
    for app in replace_application:
        application_df.APPLICATION_TYPE = application_df.APPLICATION_TYPE.replace(app,"Other")
    
    # Check to make sure binning was successful
    application_df.APPLICATION_TYPE.value_counts()

<p align="center">
  <img src="Images/app_type_counts2.png" width="250">
</p>  

    # Look at CLASSIFICATION value counts for binning.
    classification_counts = application_df.CLASSIFICATION.value_counts()
    classification_counts

<p align="center">
  <img src="Images/cls_counts1.png" width="300">
</p>  
        
    # Visualize the value counts of CLASSIFICATION.
    classification_counts.plot.density()

<p align="center">
  <img src="Images/cls_density.png" width="400">
</p>  
        
    # Determine which values to replace if counts are less than ..? (19.3.3)
    replace_class = list(classification_counts[classification_counts < 1000].index)

    # Replace in dataframe
    for cls in replace_class:
        application_df.CLASSIFICATION = application_df.CLASSIFICATION.replace(cls,"Other")

    # Check to make sure binning was successful
    application_df.CLASSIFICATION.value_counts()

<p align="center">
  <img src="Images/cls_counts2.png" width="250">
</p> 

* Encode the categorical variables using one-hot encoding.<br/><br/> 

      # Generate our categorical variable lists (19.4.2)
      application_cat = application_df.dtypes[application_df.dtypes == "object"].index.tolist()
      application_cat
      
<p align="center">
  <img src="Images/app_cat.png" width="200">
</p> 
    
    # Create a OneHotEncoder instance (19.4.3)
    enc = OneHotEncoder(sparse=False)

    # Fit and transform the OneHotEncoder using the categorical variable list
    encode_df = pd.DataFrame(enc.fit_transform(application_df[application_cat]))

    # Add the encoded variable names to the dataframe
    encode_df.columns = enc.get_feature_names(application_cat)
    encode_df.head()

<p align="center">
  <img src="Images/encode_df.png" width="700">
</p>  
  
    # Merge one-hot encoded features and drop the originals
    application_df = application_df.merge(encode_df,left_index=True, right_index=True)
    application_df = application_df.drop(application_cat,axis=1)
    application_df.head()  

<p align="center">
  <img src="Images/application_df3.png" width="700">
</p>  
  
* Split the preprocessed data into features and target arrays.<br/><br/> 

      #  Target
      y = application_df["IS_SUCCESSFUL"].values
      # Features
      X = application_df.drop(["IS_SUCCESSFUL"],axis=1).values
    
* Split the preprocessed data into training and testing datasets. <br/><br/> 
  `X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)`

* Standardize the numerical values using the StandardScaler() module. <br/><br/> 

      # Create a StandardScaler instances
      scaler = StandardScaler()

      # Fit the StandardScaler
      X_scaler = scaler.fit(X_train)

      # Scale the data
      X_train_scaled = X_scaler.transform(X_train)
      X_test_scaled = X_scaler.transform(X_test)

### Deliverable 2

* Define the neural network model using Tensorflow Keras.<br/><br/> 

      nn = tf.keras.models.Sequential()

      # First hidden layer
      nn.add(tf.keras.layers.Dense(units=80, activation="relu", input_dim=43))

      # Second hidden layer
      nn.add(tf.keras.layers.Dense(units=30, activation="relu"))

      # Output layer
      nn.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))     

      # Check the structure of the model
      nn.summary()
  
<p align="center">
  <a href="#">"nn" Model Structure</a>
  <br/><br/> 
  <img src="Images/Del_2_Model_Sequential.png" width="500">
</p> 

* Train the model and save the model's weights every 5 epochs.

      # Import checkpoint dependencies
      import os
      from tensorflow.keras.callbacks import ModelCheckpoint, Callback

      # Define the checkpoint path and filenames
      os.makedirs("checkpoints/",exist_ok=True)
      checkpoint_path = "checkpoints/weights.{epoch:02d}.hdf5"

      # Compile the model
      nn.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

      # Create a callback that saves the model's weights every 5 epochs

      batch_size=32
      steps_per_epoch = int(y_train.size / batch_size)
      period = 5

      cp_callback = ModelCheckpoint(
          filepath=checkpoint_path,
          verbose=1,
          save_weights_only=True,
          save_freq= period * steps_per_epoch)

      # Train the model
      fit_model = nn.fit(X_train_scaled,y_train,batch_size=32,epochs=100,callbacks=[cp_callback])

<p align="center">
  <img src="Images/Del_2_fit_model.png" width="700">
</p> 

* Display output of the model's loss and accuracy.<br/><br/> 

      # Evaluate the model using the test data
      model_loss, model_accuracy = nn.evaluate(X_test_scaled,y_test,verbose=2)
      print(f"Loss: {model_loss}, Accuracy: {model_accuracy}")

<p align="center">
  <img src="Images/Del_2_evaluate_model.png" width="400">
</p> 

* Save results are to an HDF5 file.<br/><br/> 

      # Export our model to HDF5 file
      nn.save("trained_application.h5")

### Deliverable 3

* Remove noisy variables from features.
<br/><br/> 
  In the original preprocessing, ID columns 'EIN' and 'NAMES' were dropped as unnecessary. Then the DataFrame columns were checked for number of unique values, `nunique`, in order to determine whether or not to bucket some of the data.  Since both the "CLASSIFICATION" and "APPLICATION_TYPE" columns had more than 10 unique values each, their `value_counts` were were visualized in a density plot for each.  A set point for each columm was established such that any `value_counts` less than the set point would be bucketed into an "Other" category.  Even with the bucketing, model's loss was 0.56 and the accuracy was only 73% when the model was evaluated.  For this reason, additional preprocessing steps were taken in an attempt to reduce the loss boost the accuracy of the model.

  1) First the DataFrame was re-loaded and this time only 'EIN' was dropped to see if binning 'NAMES' could improve optimization.
  2) Then a variable to hold the `value_counts` of the names was created.
  
          name_counts = application_df.NAME.value_counts()
 
  3) The name_counts were plotted in a density curve.

<p align="center">
  <img src="Images/Del_3_names_density.png" width="400">
</p> 

  4) Five or less name_counts were then bucketed into an "Other" category.

          # Replace if counts are less than or equal to 5.
          replace_name = list(name_counts[name_counts <= 5].index)

          # Replace in dataframe
          for name in replace_name:
              application_df.NAME = application_df.NAME.replace(name,"Other")

          # Check to make sure binning was successful
          application_df.NAME.value_counts()

<p align="center">
  <img src="Images/Del_3_names_value_counts.png" width="300">
</p> 

  5) "CLASSIFICATION" was binned as before.
  6) "APPLICATION_TYPE" was binned as well, but the `value_counts` set point to "Other" category was larger, thereby increasing the "Other" category from 276 to 804 `value_counts`.
  7) After reviewing the SPECIAL_CONSIDERATIONS `value_counts` for binning, I decided to drop the column since less than one percent (27/34299 * 100) of the applicants had special considerations.  

          spec_counts = application_df.SPECIAL_CONSIDERATIONS.value_counts()
          application_df = application_df.drop(columns = ["SPECIAL_CONSIDERATIONS"])
          print(application_df.shape)
          application_df.head()

  8) And finally, the categorical data was encoded as before with OneHotEncoder, merged back into the original DataFrame and the original categorical data dropped.

<p align="center">
  <img src="Images/Del_3_mergedDF.png" width="700">
</p> 

* Optimize the model by adding neurons to hidden layers, adding additional hidden layers, and changing the activation function of hidden or output layers.
<br/><br/> 
  1) The newly re-preprocessed data is split into features and target arrays. There were 43 features after the data was preprocessed the first round, but after re-preprocessing there are 395 features.
  2) The features and target arrays are split into training and testing datasets.
  3) X_train and X_test datasets are scaled.  
  
          # Split our preprocessed data into our features and target arrays
          #  Target
          y = application_df["IS_SUCCESSFUL"].values
          # Features
          X = application_df.drop(["IS_SUCCESSFUL"],axis=1).values

          # Split the preprocessed data into a training and testing dataset
          X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

          # Create a StandardScaler instances
          scaler = StandardScaler()

          # Fit the StandardScaler
          X_scaler = scaler.fit(X_train)

          # Scale the data
          X_train_scaled = X_scaler.transform(X_train)
          X_test_scaled = X_scaler.transform(X_test)

          number_input_features = len(X_train[0])
          number_input_features
      
  Create a function to allow kerastuner to decide the number of hidden layers, number of neurons in each layer and the activation functions of each layer.

      # Create a method that creates a new Sequential model with hyperparameter options
      def create_model(hp):
          nn_model = tf.keras.models.Sequential()

          # Allow kerastuner to decide which activation function to use in hidden layers
          activation = hp.Choice('activation',['relu','tanh','sigmoid'])

          # Allow kerastuner to decide number of neurons in first layer
          nn_model.add(tf.keras.layers.Dense(units=hp.Int('first_units',
              min_value=1,
              max_value=100,
              step=2), activation=activation, input_dim=395))

          # Allow kerastuner to decide number of hidden layers and neurons in hidden layers
          for i in range(hp.Int('num_layers', 1, 6)):
              nn_model.add(tf.keras.layers.Dense(units=hp.Int('units_' + str(i),
                  min_value=1,
                  max_value=40,
                  step=2),
                  activation=activation))

          nn_model.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))

          # Compile the model
          nn_model.compile(loss="binary_crossentropy", optimizer='adam', metrics=["accuracy"])

          return nn_model

  Run the kerastuner search for best hyperparameters.
  
      # Import the kerastuner library
      from tensorflow import keras
      import keras_tuner as kt

      tuner = kt.Hyperband(
          create_model,
          objective="val_accuracy",
          max_epochs=20,
          hyperband_iterations=2)
      # Run the kerastuner search for best hyperparameters
      tuner.search(X_train_scaled,y_train,batch_size=64,epochs=20,validation_data=(X_test_scaled,y_test))

<p align="center">
  <img src="Images/Del_3_kt_search.png" width="400">
</p> 

      # Tuner results summary shows 10 best trials
      tuner.results_summary()
      
  Check the best model's structure.
  
      nn = tuner.get_best_models(num_models=1)[0]
      nn.summary()

<p align="center">
  <img src="Images/Del_3_kt_nn_structure.png" width="500">
</p> 

* Compile and train the model.  Save the model's weights every 5 epochs.

      # Import checkpoint dependencies
      import os
      from tensorflow.keras.callbacks import ModelCheckpoint, Callback

      # Define the checkpoint path and filenames
      os.makedirs("checkpoints2/",exist_ok=True)
      checkpoint_path = "checkpoints2/weights.{epoch:02d}.hdf5"

      # Compile the model
      nn.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

      # Create a callback that saves the model's weights every 5 epochs
      # https://stackoverflow.com/questions/59069058/save-model-every-10-epochs-tensorflow-keras-v2

      batch_size=32
      steps_per_epoch = int(y_train.size / batch_size)
      period = 5

      cp_callback = ModelCheckpoint(
          filepath=checkpoint_path,
          verbose=1,
          save_weights_only=True,
          save_freq= period * steps_per_epoch)

      # Train the model
      fit_model = nn.fit(X_train_scaled,y_train,batch_size=32,epochs=100,callbacks=[cp_callback])

<p align="center">
  <img src="Images/Del_3_fit_model.png" width="600">
</p> 

    # Evaluate the model using the test data
    model_loss, model_accuracy = test_model.evaluate(X_test_scaled,y_test,verbose=2)
    print(f"Loss: {model_loss}, Accuracy: {model_accuracy}")
 
<p align="center">
  <img src="Images/Del_3_evaluate_model.png" width="600">
</p>           

* Export nn model to HDF5 file.

      nn.save("AlphabetSoupCharity_Optimization.h5")

### Deliverable 4

* Data Preprocessing
  * The target of the model is the "IS_SUCCESSFUL" category.
  * Features:  "NAME", "APPLICATION_TYPE ", "AFFILIATION", "CLASSIFICATION", "USE_CASE", "ORGANIZATION", "STATUS", "INCOME_AMT", "ASK_AMT" 
  * "EIN" is neither a target nor a feature and was removed from the input data.  In addition, "SPECIAL_CONSIDERATIONS" was a noisy variable and for this reason was also removed from the input list.
* Compiling, Training, and Evaluating the Model
  * The number of layers, neurons per layer and activation functions were selected by the kerastuner when it was run to search for the best hyperparameters.
  * Using the kerastuner, I was able to exceed the target accuracy of 75%.
  * In order to increase model performance, I preprocessed the data to bin "NAME", "APPLICATION_TYPE", and "CLASSIFICATION" columns.  in addition, I dropped the "SPECIAL_CONSIDERATIONS" column since less than one percent of the applicants had special considerations.  I then applied trial and error to estimate number of hidden layers, neurons per layer and activation functions.  I was not able to reach the target accuracy nor improve the loss.  I finally tried the kerastuner function and was able to allow the kerastuner to optimize the model.
  
## Summary:
Summarize the overall results of the deep learning model. Include a recommendation for how a different model could solve this classification problem, and explain your recommendation.
  
The results of the deep learning model were moderately successful.  The purpose of the model is help Alphabet Soup determine which applicants to approve based on their likelihood of success, or ability to make an impact with their project.  The model created by the neural network will predict potentially successful applicants with nearly 80% accuracy.  Since this large dataset does include the results, "IS_SUCCESSFUL", a supervised learning model may be able to produce even better results.

[Back to the Table of Contents](https://github.com/rkaysen63/Neural_Network_Charity_Analysis/blob/master/README.md#table-of-contents)
