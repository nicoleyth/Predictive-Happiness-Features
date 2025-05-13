[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/8ieAiaJ9)
[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-2e0aaae1b6195c2367325f4f02e2d04e9abb55f0b24a779b69b11b9e10269abc.svg)](https://classroom.github.com/online_ide?assignment_repo_id=19077547&assignment_repo_type=AssignmentRepo)
## CS260 Project: Lab Notebook
Lab Notebook Doc: [here](https://docs.google.com/document/d/14IvG18oSjFg0IV3tYVY0LBnrPfRlirCS6cI0keVszcU/edit?usp=sharing)

Presentation Slides Link: [here](https://docs.google.com/presentation/d/1mpV03OnL7WmnJnQYFnPgZEjtnZjzBL2WlZENJixgbU0/edit?usp=sharing)
(pdf version of slides uploaded in Github repo)

NOTE: We conducted experimentation grouping the features and removing the feature groupings. The current main.py file, when run, produces results for them feature are grouped into larger categories. To view results for features without grouping, comment line 31 in main.py file. The results for features before grouping are also stored in 'before_feat_combination' folder under 'figures'.  

########## Deadlines ##########

4/30 Presentation Slides
5/13 Code & Lab Notebook

########## To-Dos ##########
- Separate Datase
    - Extract dataset by year (dataset for just 2001, 2002, 2003, …, etc.)
    - Find entropy and gain from 2001
- Code/Calculations
    - Use ^^ to predict 2002-2024 entropy → then calculate gain with this value
    - For all years, calculate actual entropy for feature and gain.
    - Plot predicted and actual gain values.
    - Find the residuals (?)
        - Determine a way to compare actual and predicted values - more clarifications TBD
        - Create a visualization of the differences
    - Apply the predictive method and visualization for all other methods
- Presentation Slides (4/28)
- Iterate process for all years (select best term as predicted entropy value)
- Compare the average RSS for each year
- Try combining feature terms (increase entropy value)
- Clean output to only show relevant final output information
- Attach presentation slides and lab notebook to github

########## Log Entries ##########

Nicole & Aiden: 03-28-25 (1.5 hours)
- Searched for potential datasets. Selected “World Happiness Report”.
- Brainstormed project ideas/plans.
- Worked on dataset description on project proposal. 

Nicole & Aiden: 04-01-25 (1 hour)
- Polished project ideas.
- Ran idea by Professor Mathieson, finalized project plan.

Nicole & Aiden: 04-02-25 (4 hours)
- Completed project proposal and submitted to Professor Mathieson.

Aiden & Nicole: 04-13-25 (1 hour)
- Inputted the entropy file (from lab 6) and added the dataset from Kaggle.
- Discussed deadlines, tasks needed to be completed.

Aiden: 04-18-25 (1 hour)
- Calculated entropy (predicted and actual) for data, gain, and used both of the datasets to find the residual (2006-2024).
- First checked which years exist in the dataset.
- Created convert_labels() function, to convert y_label into binary 0/1 using threshold of 5 for dataset scaled between 0-10.

Nicole: 04-21-25 (1.5 hours)
- Worked on code efficiency, applied functions (top-down design).
- Fixed errors, makes sure the code is functioning correctly.

Nicole: 04-22-25 (1.5 hours)
- Function to accommodate continuous features (working on!).
- Working on finding the top best features from training data. - ended while in debugging process.
    - Our training data is set to the earliest year.
- Question to consider: 
    - Should we select a new dataset? All dataset features are continuous, unless we convert everything to discrete data. 
    - The y_label data is also continuous, how should we make this discrete? Should we use a threshold of exactly ½?

Nicole & Aiden: 04-22-25 (1 hour)
- Discussed how to convert continuous features to discrete features with Professor Mathieson during our lab session.
- Fixed “list[x]” does not exist error.

Aiden: 04-27-25 (1 hour)
- Tried to fix errors in main.py.
- Fixed IndexError: list index out of range.
- Fixed ValueError: ('Lengths must match to compare', (4000,), (174,)).

Nicole: 04-28-25 (1 hour)
- Included the best feature function to sort gain values and calculate the best values.
- Calculated the predicted entropy (training data).
- Iterate through all years and recalculate entropy and gain values.
    - Gain values were initially calculated using the non-predicted entropy and didn’t accommodate for the specific feature that we selected as the best feature.
- Something to think about: is_continuous, do we need to find continuous thresholds?

Nicole: 04-30-25 (1 hour)
- Created a figure to visualize the comparison between actual and predicted gain by year.
- Calculated residuals and created residuals scatter plot.
- Started to implement predicted and actual gains for all years (to compare difference).

Aiden: 04-30-25 (1 hour)
- Fixed figure to visualize difference between actual and predicted gain by year (RSS plot).
- Created a best fit line for the points which were consistent (disregarding outliners).

Nicole: 04-30-25 (3 hours)
- Reimplemented continuous feature function - to identify thresholds for datasets.
- Found error in gain calculations in Partition.py file - successfully fixed gain calculation logic.
- Changed actual vs predicted gain scatter plot into bar graph (better visualization).
- Created a figure comparing predicted gain with the gain value of the best feature per year.
- Created a figure showing the gain values of the initial (prediction) dataset - this is to help visualize the best gain value during the presentation.

Nicole: 05-12-25 (1.5 hours)
- Cleaned up code, included some more comments, and simplified code with more functions.
- Helper functions: 
    - clean_data(): Loads dataset into pd, remove missing y-values (happiness score), reorganize data into list of dictionaries ordered by year, return years_data and features list
    - set_data(): remove excluded columns, set X_base and y_base matrices, initialize list of Example objects, returns X_base, y_base, and examples

Nicole: 05-13-25 (4 hours)
- Iterate through the entire project process multiple times, for each iteration, set a different year as the ‘training’ dataset. 
    - To shorten the length of the code, I converted multiple calculation and plotting procedures into functions. 
        - predict_gain(year, years_data, pred_entropy, features): calculates the predicted gain by calculating actual_entropy and using parameter variables
        - calc_actual_gain(year, years_data, predicted_gain, features): calculates the actual gain without the predicted entropy and returns dictionaries of actual gains and year_best_features with key value set as year.
        - calc_rss(year, years_data, actual_gain, predicted_gain): calculates the rss between actual and predicted gain
        - plot_rss(rss, year, other_years, line): plots scatter plot as the name specifies
        - plot_gain_comparison_w_feat(year, year_best_feature, predicted_gain, best_pred_feat): plot bar graph comparing predicted with actual gain values. 
- Comparing the average RSS values for each training year to identify the best predictive feature
- Created a bar plot showing RSS values (between actual and predicted gain values) with its selected best feature per year
- Cleaned print outputs, only printing relevant output information in an organized format. 

Nicole: 05-13-25 (1 hour)
- Grouped features into categories (hopefully larger entropy values = better accuracy)
    - All the datasets are combined. For feature values, I multiplied the feature values in each category (essentially the data value for each feature are weights for other features under the same category). 
    - New dataset replaced years_data and features at the beginning of the code (line 30)
- I reran the code with the new dataset and generated new graphs for comparison. 
    - Former graphs are moved into a new folder under figures/before_feat_combination


########## References ##########

Yadav, Khushi. “World Happiness Report”. Kaggle. https://www.kaggle.com/datasets/khushikyad001/world-happiness-report/data 


########## AI Code Generator Use ##########

For the project, we used ChatGPT 4.0 to help us through part of the programming process. Using ChatGPT really helped with the pacing of the project, but there were limitations to its coding effectiveness. There were times when it was much better to use our own logic and online research to complete a procedure because ChatGPT was not following specific instructions, and the code did not work as it was supposed to. One example of this was when creating a bar plot comparing the average RSS per year. For some particular reason, ChatGPT generated many variations of the code, but none of them produced the output that we desired. As a result, we spent more than 20 minutes playing around with ChatGPT when we could have written the code ourselves in less than 5 minutes. As a result of many similar experiences, we had to learn how to balance when to use ChatGPT and when to rely on ourselves to code parts of the procedures. 

We realized that ChatGPT was really helpful when generating code for simple functions, such as producing a scatter plot given x-values and y-values, and converting the y-labels function. It is also helpful at times to catch mistakes in the code, such as having a function and a variable with the same name present in the code. It is also helpful in explaining what certain errors are and what is going on in certain parts of the code. However, for more complicated logic, ChatGPT was rather inefficient, and we had to rely on ourselves. Such examples include implementing the Example and Partition classes. Additionally, for more complicated and systematic errors that require processing through many layers of logic, ChatGPT was inefficient. Lastly, it was much preferred to manually research errors and quick fixes rather than asking ChatGPT because ChatGPT gave a series of suggestions to try, which is much more inefficient than directly searching for the solution we want. Overall, I would say ChatGPT was a helpful tool, but it played a small role throughout the implementation process of our project. 