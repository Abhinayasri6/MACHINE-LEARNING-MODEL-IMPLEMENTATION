# MACHINE-LEARNING-MODEL-IMPLEMENTATION

1*COMPANY*:CODTECH IT SOLUTIONS

*NAME*:BASANI ABHINAYASRI

*INTERN ID*:CT04DH1512

*DOMAIN*:PYTHON PROGRAMMING

*DURATION*:4 WEEKS

*MENTOR*:NEELA SANTHOSH

*DESCRIPTION*:This machine learning project centers on building a predictive model to classify iris flower species using the widely recognized Iris dataset. The dataset includes 150 samples of iris flowers, each characterized by four numerical features—sepal length, sepal width, petal length, and petal width—and associated with one of three distinct species: Iris-setosa, Iris-versicolor, and Iris-virginica. Due to its simplicity, cleanliness, and balanced class distribution, the Iris dataset has become a staple in machine learning education and prototyping. In this implementation, we use the Logistic Regression algorithm from the scikit-learn library to perform multi-class classification. The process begins with importing core Python libraries for data analysis and visualization, including pandas, NumPy, matplotlib, and seaborn, alongside scikit-learn tools for model training and evaluation.
Since the dataset is already preloaded and numeric, preprocessing requirements are minimal. There's no need for feature extraction techniques like vectorization or normalization for this basic example. The features and labels are directly assigned to variables X and y, and the data is split into training and testing sets using train_test_split, reserving 25% for evaluation. This helps us train the model on a representative subset while ensuring reliable performance assessment on unseen data.
Next, a Logistic Regression model is instantiated and trained on the training data. The max_iter parameter is increased to 200 to ensure convergence, as the default value of 100 may sometimes be insufficient for multi-class problems. Logistic Regression, despite its name, is a powerful linear classification algorithm and serves as an excellent starting point for baseline modeling. Once trained, the model makes predictions on the test data. These predictions are compared with the true labels to evaluate performance.
The evaluation results include a classification report detailing precision, recall, F1-score, and support for each iris class, helping us understand how well the model discriminates between different flower species. Additionally, an overall accuracy score is printed, which quantifies the percentage of correct predictions. To visualize model performance, a confusion matrix is generated and plotted using seaborn's heatmap function. This matrix offers a granular view of the number of correctly and incorrectly classified samples across the three species, allowing us to identify potential patterns of misclassification.
In summary, this project presents a straightforward yet effective implementation of a predictive machine learning model using Logistic Regression. The Iris dataset’s well-defined structure makes it an ideal candidate for beginners to explore core machine learning concepts. 

*OUTPUT*:

<img width="1920" height="1080" alt="Image" src="https://github.com/user-attachments/assets/e8984fd3-1069-4e72-9937-c6f37ec3242f" />


